# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end pure-PyTorch inference pipeline for TRIBE v2.

This is the user-facing entry point of the ``tribev2.fast`` subpackage.
It:

1. Loads the brain-model checkpoint and the modality-extractor configs
   from a local directory or HuggingFace Hub repo.
2. Builds three pure-PyTorch extractors (text / audio / video) - no
   ``neuralset.extractors``, no ``exca`` infra, no SLURM bookkeeping.
3. Pre-processes events (audio extraction from video, ASR -> words +
   context) using the **CPU-bound** transforms from the existing
   ``tribev2.demo_utils.get_audio_and_text_events`` helper - those are
   already lightweight and not the bottleneck.
4. Runs feature extraction **in parallel across visible GPUs** (one
   extractor per GPU via threads + ``torch.cuda.device``).  Cached
   features are written to ``cache_dir`` so repeat runs skip the work.
5. Slices cached per-event features into per-segment ``(L, D, T)``
   tensors via ``tribev2.fast.dataset.FastSegmentDataset``.
6. Hands the dataset to ``tribev2.inference.InferenceRunner`` for the
   optimised model forward (autocast, pinned host buffer, async copies,
   multi-GPU strided sharding).
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from neuralset.events.utils import standardize_events

from tribev2.demo_utils import (
    VALID_SUFFIXES,
    TextToEvents,
    get_audio_and_text_events,
)
from tribev2.fast.dataset import FastBatch, FastSegmentDataset, fast_collate
from tribev2.fast.extractors import (
    AudioExtractor,
    ExtractorConfig,
    TextExtractor,
    VideoExtractor,
    _BaseFastExtractor,
)
from tribev2.fast.segments import (
    events_from_dataframe,
    list_segments,
)
from tribev2.inference import (
    DeviceLike,
    DtypeLike,
    InferenceRunner,
    _select_devices,
    _select_dtype,
)
from tribev2.model import FmriEncoder

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Lightweight TribeModel-shaped adapter for InferenceRunner
# --------------------------------------------------------------------------- #


class _BrainModelAdapter:
    """Minimal duck-type that ``InferenceRunner.__init__`` understands.

    ``InferenceRunner`` reads ``tribe_model._model``, ``.data.TR``,
    ``.data.batch_size``, and ``.remove_empty_segments``.  We synthesise
    a tiny holder so we don't have to instantiate ``TribeExperiment``.
    """

    class _DataShim:
        def __init__(self, TR: float, batch_size: int):
            self.TR = TR
            self.batch_size = batch_size

        # Stub - never called in the dataset-precomputed path.
        def get_loaders(self, *_a, **_kw):
            raise RuntimeError(
                "FastTribePipeline pre-builds the dataset; "
                "get_loaders should not be reached"
            )

    def __init__(self, model: torch.nn.Module, *, TR: float, batch_size: int):
        self._model = model
        self.data = _BrainModelAdapter._DataShim(TR=TR, batch_size=batch_size)
        self.remove_empty_segments = True


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #


class FastTribePipeline:
    """Pure-PyTorch single-PC inference pipeline for TRIBE v2.

    Construct via :meth:`from_pretrained`.  Use as a context manager so
    the GPU model + extractor replicas are freed deterministically.
    """

    # ---- construction ------------------------------------------------ #

    def __init__(
        self,
        *,
        brain_model: torch.nn.Module,
        text_config: ExtractorConfig | None,
        audio_config: ExtractorConfig | None,
        video_config: ExtractorConfig | None,
        TR: float,
        batch_size: int,
        cache_dir: Path,
        devices: list[torch.device],
        dtype: torch.dtype,
        num_workers: int | None,
        prefetch_factor: int,
        pin_memory: bool,
        persistent_workers: bool,
        segment_aggregation: str,
        duration_trs: int,
        overlap_trs: int,
        stride_drop_incomplete: bool,
        features_to_use: list[str],
    ):
        self.brain_model = brain_model
        self.devices = devices
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.TR = TR
        self.batch_size = batch_size
        self.segment_aggregation = segment_aggregation
        self.duration_trs = duration_trs
        self.overlap_trs = overlap_trs
        self.stride_drop_incomplete = stride_drop_incomplete
        self.features_to_use = features_to_use

        # One extractor per requested modality; constructed lazily on
        # the assigned device inside _extract_in_parallel.
        self._extractor_configs: dict[str, ExtractorConfig] = {}
        if text_config is not None and "text" in features_to_use:
            self._extractor_configs["text"] = text_config
        if audio_config is not None and "audio" in features_to_use:
            self._extractor_configs["audio"] = audio_config
        if video_config is not None and "video" in features_to_use:
            self._extractor_configs["video"] = video_config

        # The InferenceRunner wraps the brain model + handles forward.
        adapter = _BrainModelAdapter(
            brain_model, TR=TR, batch_size=batch_size
        )
        runner_kwargs: dict[str, tp.Any] = dict(
            devices=devices,
            dtype=dtype,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            free_after=False,
        )
        if num_workers is not None:
            runner_kwargs["num_workers"] = num_workers
        self._runner = InferenceRunner(adapter, **runner_kwargs)
        self._closed = False

    def __enter__(self) -> "FastTribePipeline":
        return self

    def __exit__(self, *exc: tp.Any) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._runner.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._closed = True

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        *,
        checkpoint_name: str = "best.ckpt",
        cache_dir: str | Path = "./cache",
        devices: DeviceLike | list[DeviceLike] | None = None,
        dtype: DtypeLike = "auto",
        batch_size: int | None = None,
        num_workers: int | None = None,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        text_model: str | None = None,
    ) -> "FastTribePipeline":
        """Load checkpoint + config and build a fast inference pipeline.

        ``checkpoint_dir`` is either a local directory containing
        ``config.yaml`` + ``<checkpoint_name>``, or a HuggingFace Hub
        repo id (e.g. ``"facebook/tribev2"``).
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        device_list = _select_devices(devices)
        dtype_resolved = _select_dtype(dtype, device_list[0])

        ckpt_path, config_path = _resolve_paths(checkpoint_dir, checkpoint_name)
        import platform
        if platform.system() == "Windows":
            import pathlib
            _orig_posix = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
        try:
            with open(config_path, "r") as f:
                full_config = yaml.load(f, Loader=yaml.UnsafeLoader)
        finally:
            if platform.system() == "Windows":
                pathlib.PosixPath = _orig_posix

        data_cfg = full_config.get("data", {})
        layers_to_use = data_cfg.get("layers_to_use")
        layer_aggregation = data_cfg.get("layer_aggregation", "group_mean")
        TR = 1.0 / float(data_cfg.get("neuro", {}).get("frequency", 1.0))
        bs = (
            batch_size
            if batch_size is not None
            else int(data_cfg.get("batch_size", 8))
        )
        features_to_use = list(data_cfg.get("features_to_use", []))
        segment_aggregations = {
            modality: data_cfg.get(f"{modality}_feature", {}).get("aggregation", "sum")
            for modality in ("text", "audio", "video")
        }
        # Default to text aggregation for the segment-level mode (text is
        # the dominant static modality).  The original Data class applies
        # the per-extractor aggregation; here we pick one global setting
        # since slice_*_for_segment uses the same rule for all modalities.
        seg_agg = segment_aggregations.get("text", "sum")

        text_cfg = _make_extractor_config(
            data_cfg.get("text_feature"),
            layers_to_use=layers_to_use,
            layer_aggregation=layer_aggregation,
            dtype=dtype_resolved,
        )
        if text_cfg is not None:
            text_cfg.batch_size = min(text_cfg.batch_size, 4)
        if text_model is not None and text_cfg is not None:
            text_cfg = ExtractorConfig(
                model_name=text_model,
                layers=text_cfg.layers,
                layer_aggregation=text_cfg.layer_aggregation,
                token_aggregation=text_cfg.token_aggregation,
                frequency=text_cfg.frequency,
                batch_size=text_cfg.batch_size,
                dtype=text_cfg.dtype,
            )
        audio_cfg = _make_extractor_config(
            data_cfg.get("audio_feature"),
            layers_to_use=layers_to_use,
            layer_aggregation=layer_aggregation,
            dtype=dtype_resolved,
        )
        video_cfg = _make_extractor_config(
            data_cfg.get("video_feature"),
            layers_to_use=layers_to_use,
            layer_aggregation=layer_aggregation,
            dtype=dtype_resolved,
        )
        if video_cfg is not None:
            video_cfg.batch_size = 1

        # Build the brain model from the checkpoint.
        logger.info("Loading brain-model checkpoint from %s", ckpt_path)
        ckpt = torch.load(
            ckpt_path, map_location="cpu", weights_only=True, mmap=True
        )
        build_args = ckpt["model_build_args"]
        state_dict = {
            k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()
        }
        del ckpt
        brain_model_config = full_config["brain_model_config"]
        brain_cfg = FmriEncoder(**brain_model_config)
        if brain_cfg.subject_layers is not None:
            brain_cfg.subject_layers.average_subjects = True
            brain_cfg.subject_layers.n_subjects = 0
        model = brain_cfg.build(**build_args)
        model.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict
        model.to(device_list[0]).eval()

        return cls(
            brain_model=model,
            text_config=text_cfg,
            audio_config=audio_cfg,
            video_config=video_cfg,
            TR=TR,
            batch_size=bs,
            cache_dir=cache_dir,
            devices=device_list,
            dtype=dtype_resolved,
            num_workers=num_workers if num_workers is not None else (0 if os.name == "nt" else 2),
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            segment_aggregation=seg_agg,
            duration_trs=int(data_cfg.get("duration_trs", 40)),
            overlap_trs=int(data_cfg.get("overlap_trs_train", 0)),
            stride_drop_incomplete=bool(
                data_cfg.get("stride_drop_incomplete", False)
            ),
            features_to_use=features_to_use,
        )

    # ---- event preprocessing ---------------------------------------- #

    def get_events_dataframe(
        self,
        text_path: str | None = None,
        audio_path: str | None = None,
        video_path: str | None = None,
    ) -> pd.DataFrame:
        """Same surface as :meth:`TribeModel.get_events_dataframe`."""
        provided = {
            n: v
            for n, v in (
                ("text_path", text_path),
                ("audio_path", audio_path),
                ("video_path", video_path),
            )
            if v is not None
        }
        if len(provided) != 1:
            raise ValueError(
                f"Exactly one of text_path, audio_path, video_path must be "
                f"provided, got: {list(provided.keys()) or 'none'}"
            )
        name, value = next(iter(provided.items()))
        path = Path(value)
        if path.suffix.lower() not in VALID_SUFFIXES[name]:
            raise ValueError(
                f"{name} must end with one of {sorted(VALID_SUFFIXES[name])}, "
                f"got '{path.suffix}'"
            )
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")

        if text_path is not None:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError(f"Text file is empty: {path}")
            return TextToEvents(
                text=text,
                infra={"folder": str(self.cache_dir), "mode": "retry"},
            ).get_events()

        event_type = "Audio" if audio_path is not None else "Video"
        event = {
            "type": event_type,
            "filepath": str(path),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        return get_audio_and_text_events(pd.DataFrame([event]))

    # ---- the main loop ---------------------------------------------- #

    def predict(
        self,
        events: pd.DataFrame,
        *,
        verbose: bool = True,
        remove_empty_segments: bool = True,
    ) -> tuple[np.ndarray, list]:
        """Extract features in parallel, slice into segments, run forward."""
        if self._closed:
            raise RuntimeError("FastTribePipeline has been closed")

        events = standardize_events(events)
        events = self._add_dummy_categorical_events(events)
        fast_events = events_from_dataframe(events)

        # Move brain model to CPU during extraction to free VRAM.
        brain_device = next(self.brain_model.parameters()).device
        self.brain_model.cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        extractors = self._build_extractors_on_devices()
        try:
            self._extract_in_parallel(extractors, fast_events, verbose=verbose)
        finally:
            for ext in extractors.values():
                ext.free()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.brain_model.to(brain_device)

        # Build segments and dataset.
        duration = self.duration_trs * self.TR
        stride = (self.duration_trs - self.overlap_trs) * self.TR
        segments = list_segments(
            fast_events,
            duration=duration,
            stride=stride,
            stride_drop_incomplete=self.stride_drop_incomplete,
        )
        if not segments:
            n_vertices = int(self.brain_model.n_outputs)
            return np.empty((0, n_vertices), dtype=np.float32), []

        dataset = FastSegmentDataset(
            events=fast_events,
            segments=segments,
            extractors=extractors,
            segment_aggregation=self.segment_aggregation,
        )
        # Debug: log cached feature shapes
        for name, ext in extractors.items():
            for ev in fast_events:
                if ext.has_cached(ev):
                    arr = ext.load_cached(ev)
                    logger.info("Cached %s feature for %s: shape=%s", name, ev.type, arr.shape)
                    break
        # Hand to InferenceRunner with our collate.
        return self._runner.predict(
            dataset=dataset,
            collate_fn=fast_collate,
            verbose=verbose,
            remove_empty_segments=remove_empty_segments,
        )

    # ---- internals --------------------------------------------------- #

    def _build_extractors_on_devices(self) -> dict[str, _BaseFastExtractor]:
        """Round-robin assign each modality extractor to a CUDA device."""
        gpu_devices = [d for d in self.devices if d.type == "cuda"]
        if not gpu_devices:
            gpu_devices = [self.devices[0]]
        out: dict[str, _BaseFastExtractor] = {}
        modality_to_class = {
            "text": TextExtractor,
            "audio": AudioExtractor,
            "video": VideoExtractor,
        }
        for i, (modality, cfg) in enumerate(self._extractor_configs.items()):
            cls = modality_to_class[modality]
            dev = gpu_devices[i % len(gpu_devices)]
            out[modality] = cls(cfg, cache_dir=self.cache_dir, device=dev)
        return out

    def _extract_in_parallel(
        self,
        extractors: dict[str, _BaseFastExtractor],
        events: list,
        *,
        verbose: bool,
    ) -> None:
        items = list(extractors.items())
        if not items:
            return
        gpu_devices = [d for d in self.devices if d.type == "cuda"]
        if len(gpu_devices) <= 1 or len(items) == 1:
            for name, ext in items:
                logger.info("Extracting %s on %s", name, ext.device)
                ext.prepare(events)
                ext.free()  # free as we go on single-GPU to keep VRAM low
            return
        log_lock = threading.Lock()

        def _run_one(name: str, ext: _BaseFastExtractor) -> None:
            gpu_idx = ext.device.index if ext.device.index is not None else 0
            with torch.cuda.device(gpu_idx):
                with log_lock:
                    logger.info("Extracting %s on cuda:%d", name, gpu_idx)
                ext.prepare(events)

        with ThreadPoolExecutor(
            max_workers=min(len(items), len(gpu_devices)),
            thread_name_prefix="fast-extract",
        ) as pool:
            futures = [
                pool.submit(_run_one, name, ext) for name, ext in items
            ]
            for f in futures:
                f.result()

    def _add_dummy_categorical_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """One CategoricalEvent per timeline carrying the segment bounds."""
        dummy: list[dict[str, tp.Any]] = []
        for timeline_name, timeline in events.groupby("timeline"):
            if "split" in timeline.columns:
                splits = timeline.split.dropna().unique()
                split = splits[0] if len(splits) else "all"
            else:
                split = "all"
            dummy.append(
                {
                    "type": "CategoricalEvent",
                    "timeline": timeline_name,
                    "start": timeline.start.min(),
                    "duration": timeline.stop.max() - timeline.start.min(),
                    "split": split,
                    "subject": timeline.subject.unique()[0],
                }
            )
        out = pd.concat([events, pd.DataFrame(dummy)])
        return standardize_events(out)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _resolve_paths(
    checkpoint_dir: str | Path, checkpoint_name: str
) -> tuple[Path, Path]:
    cp = Path(checkpoint_dir)
    if cp.exists():
        return cp / checkpoint_name, cp / "config.yaml"
    from huggingface_hub import hf_hub_download

    repo_id = str(checkpoint_dir).replace("\\", "/")
    return (
        Path(hf_hub_download(repo_id, checkpoint_name)),
        Path(hf_hub_download(repo_id, "config.yaml")),
    )


def _make_extractor_config(
    feature_cfg: dict | None,
    *,
    layers_to_use: list[float] | None,
    layer_aggregation: str | None,
    dtype: torch.dtype,
) -> ExtractorConfig | None:
    """Translate a YAML extractor block into our :class:`ExtractorConfig`."""
    if feature_cfg is None:
        return None
    # The YAML carries plenty of cluster / infra noise we ignore.
    layers = layers_to_use or feature_cfg.get("layers")
    if layers is None:
        return None
    # Some extractor configs nest the model name under image (video).
    model_name = feature_cfg.get("model_name") or feature_cfg.get("image", {}).get(
        "model_name"
    )
    if model_name is None:
        return None
    return ExtractorConfig(
        model_name=str(model_name),
        layers=list(layers),
        layer_aggregation=layer_aggregation or "group_mean",
        token_aggregation=feature_cfg.get("token_aggregation", "mean"),
        frequency=float(feature_cfg.get("frequency", 2.0)),
        batch_size=int(feature_cfg.get("batch_size", 4)),
        dtype=dtype,
    )
