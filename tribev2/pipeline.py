# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end single-PC inference pipeline for TRIBE v2.

This module is a from-scratch rewrite of the inference path that
replaces :class:`tribev2.demo_utils.TribeModel` for single-machine use:

* No SLURM / cluster machinery (the existing ``infra`` config is
  rewritten to local mode at construction time).
* No PyTorch Lightning Trainer / FSDP for either feature extraction or
  the model forward.
* Disk cache for extracted features is **kept** by default (resumable
  across runs) but the SLURM / multi-node scaffolding around it is
  stripped.
* Feature extractors (text / audio / video) run **in parallel across
  the available GPUs** via one thread per extractor, using
  :func:`torch.cuda.device` to pin each thread to a target GPU
  (the extractor's ``device`` field is a ``Literal["auto","cpu","cuda",...]``
  that doesn't accept ``"cuda:N"``, so per-thread default-device
  switching is the cleanest way to map them onto distinct GPUs).
* The model forward is delegated to :class:`tribev2.inference.InferenceRunner`
  with a pre-built dataset, so the optimised inference loop
  (autocast, pinned-host buffer, async copies, on-device keep-mask
  index_select, multi-GPU strided sharding) is reused as-is.

Typical usage::

    from tribev2 import TribePipeline

    pipe = TribePipeline.from_pretrained("facebook/tribev2")
    events = pipe.get_events_dataframe(video_path="clip.mp4")
    preds, segments = pipe.predict(events)
    pipe.close()
"""

from __future__ import annotations

import gc
import logging
import threading
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import neuralset as ns
import numpy as np
import pandas as pd
import torch
import yaml
from exca import ConfDict
from neuralset.events.etypes import EventTypesHelper
from neuralset.events.utils import standardize_events

from tribev2.demo_utils import (
    VALID_SUFFIXES,
    TextToEvents,
    get_audio_and_text_events,
)
from tribev2.inference import (
    DeviceLike,
    DtypeLike,
    InferenceRunner,
    _select_devices,
    _select_dtype,
)
from tribev2.main import TribeExperiment, _free_extractor_model

logger = logging.getLogger(__name__)


# Per-extractor infra fields that become irrelevant in single-PC mode.
# Anything not listed here (notably ``folder``, ``keep_in_ram``,
# ``mode``, ``version``) is preserved so the on-disk cache layout stays
# compatible across runs.
_INFRA_FIELDS_TO_RESET: tuple[tuple[str, tp.Any], ...] = (
    ("cluster", None),
    ("slurm_partition", ""),
    ("slurm_constraint", ""),
    ("gpus_per_node", 0),
    ("max_jobs", 1),
    ("min_samples_per_job", 1),
)


def _strip_cluster_config(infra_obj: tp.Any) -> None:
    """Reset cluster-only fields on an exca infra object to local-mode values.

    Skips fields that the infra object does not expose (older exca
    versions).  Leaves caching fields alone.
    """
    for name, value in _INFRA_FIELDS_TO_RESET:
        if hasattr(infra_obj, name):
            try:
                setattr(infra_obj, name, value)
            except (TypeError, ValueError):
                # Field is frozen / has a stricter type - skip silently.
                pass


def _add_dummy_categorical_events(events: pd.DataFrame) -> pd.DataFrame:
    """Mirror the dummy-event injection in :meth:`Data.get_loaders`.

    A ``CategoricalEvent`` per timeline carries the segment-construction
    bounds (``start``, ``duration``, ``split``).  Without these the
    downstream :func:`ns.segments.list_segments` call has nothing to
    trigger on.
    """
    dummy_events: list[dict] = []
    for timeline_name, timeline in events.groupby("timeline"):
        if "split" in timeline.columns:
            splits = timeline.split.dropna().unique()
            assert len(splits) == 1, (
                f"Timeline {timeline_name} has multiple splits: {splits}"
            )
            split = splits[0]
        else:
            split = "all"
        dummy_events.append(
            {
                "type": "CategoricalEvent",
                "timeline": timeline_name,
                "start": timeline.start.min(),
                "duration": timeline.stop.max() - timeline.start.min(),
                "split": split,
                "subject": timeline.subject.unique()[0],
            }
        )
    out = pd.concat([events, pd.DataFrame(dummy_events)])
    return standardize_events(out)


def _filter_relevant_extractors(
    extractors: dict[str, tp.Any], events: pd.DataFrame
) -> dict[str, tp.Any]:
    """Drop extractors whose event types are not present in the dataframe."""
    present = set(events.type.unique())
    keep: dict[str, tp.Any] = {}
    for name, extractor in extractors.items():
        ev_types = EventTypesHelper(extractor.event_types).names
        if any(t in present for t in ev_types):
            keep[name] = extractor
        else:
            logger.warning(
                "Skipping extractor %s (no events of types %s)", name, ev_types
            )
    return keep


class TribePipeline:
    """Single-PC end-to-end inference pipeline for TRIBE v2.

    Construct via :meth:`from_pretrained`.  Holds an internal
    :class:`TribeExperiment` for config + checkpoint loading; the
    legacy ``predict``/``get_loaders`` paths on that object are not
    used.
    """

    def __init__(
        self,
        xp: TribeExperiment,
        *,
        cache_dir: Path,
        devices: list[torch.device],
        dtype: torch.dtype,
        batch_size: int | None = None,
        num_workers: int | None = None,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        remove_empty_segments: bool = True,
    ):
        self._xp = xp
        self.cache_dir = cache_dir
        self.devices = devices
        self.dtype = dtype
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.remove_empty_segments = remove_empty_segments

        # Configure each modality extractor for single-PC, local-only mode.
        # Disk cache stays at ``cache_dir`` so repeat runs are resumable.
        for modality in ("text", "audio", "video", "image"):
            extractor = getattr(xp.data, f"{modality}_feature", None)
            if extractor is None:
                continue
            if hasattr(extractor, "infra"):
                _strip_cluster_config(extractor.infra)
                if hasattr(extractor.infra, "folder"):
                    extractor.infra.folder = str(cache_dir)
                if hasattr(extractor.infra, "keep_in_ram"):
                    extractor.infra.keep_in_ram = True
            # video extractor wraps an inner image extractor with its own infra
            inner = getattr(extractor, "image", None)
            if inner is not None and hasattr(inner, "infra"):
                _strip_cluster_config(inner.infra)
                if hasattr(inner.infra, "folder"):
                    inner.infra.folder = str(cache_dir)
                if hasattr(inner.infra, "keep_in_ram"):
                    inner.infra.keep_in_ram = True

        # Inference runner reuses the optimised forward.  We pass
        # ``num_workers``/``batch_size`` overrides through.
        runner_kwargs: dict[str, tp.Any] = dict(
            devices=devices,
            dtype=dtype,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            free_after=False,
        )
        if batch_size is not None:
            runner_kwargs["batch_size"] = batch_size
        if num_workers is not None:
            runner_kwargs["num_workers"] = num_workers
        self._runner = InferenceRunner(xp, **runner_kwargs)
        self._closed = False

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> "TribePipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._runner.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._closed = True

    # -- construction ----------------------------------------------------

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
        remove_empty_segments: bool = True,
        config_update: dict | None = None,
    ) -> "TribePipeline":
        """Load a checkpoint + config and build a single-PC pipeline.

        ``checkpoint_dir`` is either a local directory containing
        ``config.yaml`` + ``<checkpoint_name>``, or a HuggingFace Hub
        repo id (e.g. ``"facebook/tribev2"``).  See
        :meth:`tribev2.demo_utils.TribeModel.from_pretrained` for the
        original behaviour - this method follows the same loading
        contract but without the legacy ``TribeModel`` wrapper.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        device_list = _select_devices(devices)
        dtype_resolved = _select_dtype(dtype, device_list[0])

        checkpoint_path = Path(checkpoint_dir)
        if checkpoint_path.exists():
            config_path = checkpoint_path / "config.yaml"
            ckpt_path = checkpoint_path / checkpoint_name
        else:
            from huggingface_hub import hf_hub_download

            repo_id = str(checkpoint_dir)
            config_path = hf_hub_download(repo_id, "config.yaml")
            ckpt_path = hf_hub_download(repo_id, checkpoint_name)

        with open(config_path, "r") as f:
            config = ConfDict(yaml.load(f, Loader=yaml.UnsafeLoader))

        # Strip the SLURM-shaped pieces of the config before TribeExperiment
        # validates them.  Mirrors what TribeModel.from_pretrained does.
        for modality in ("text", "audio", "video"):
            config[f"data.{modality}_feature.infra.folder"] = str(cache_dir)
            config[f"data.{modality}_feature.infra.cluster"] = None
        for param in (
            "infra.workdir",
            "data.study.infra_timelines",
            "data.neuro.infra",
            "data.image_feature.infra",
        ):
            config.pop(param)
        config["data.study.path"] = "."
        config["average_subjects"] = True
        config["checkpoint_path"] = str(config["infra.folder"]) + f"/{checkpoint_name}"
        if config_update is not None:
            config.update(config_update)

        xp = TribeExperiment(**config)

        logger.info("Loading model from %s", ckpt_path)
        ckpt = torch.load(
            ckpt_path, map_location="cpu", weights_only=True, mmap=True
        )
        build_args = ckpt["model_build_args"]
        state_dict = {
            k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()
        }
        del ckpt
        model = xp.brain_model_config.build(**build_args)
        model.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict
        model.to(device_list[0])
        model.eval()
        xp._model = model

        return cls(
            xp,
            cache_dir=cache_dir,
            devices=device_list,
            dtype=dtype_resolved,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            remove_empty_segments=remove_empty_segments,
        )

    # -- event preprocessing --------------------------------------------

    def get_events_dataframe(
        self,
        text_path: str | None = None,
        audio_path: str | None = None,
        video_path: str | None = None,
    ) -> pd.DataFrame:
        """Build an events DataFrame from exactly one input source.

        Mirrors :meth:`TribeModel.get_events_dataframe` so existing
        notebooks transfer without changes.
        """
        provided = {
            name: value
            for name, value in (
                ("text_path", text_path),
                ("audio_path", audio_path),
                ("video_path", video_path),
            )
            if value is not None
        }
        if len(provided) != 1:
            raise ValueError(
                f"Exactly one of text_path, audio_path, video_path must be "
                f"provided, got: {list(provided.keys()) or 'none'}"
            )
        name, value = next(iter(provided.items()))
        path = Path(value)
        suffix = path.suffix.lower()
        if suffix not in VALID_SUFFIXES[name]:
            raise ValueError(
                f"{name} must end with one of {sorted(VALID_SUFFIXES[name])}, "
                f"got '{suffix}'"
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

    # -- the actual inference -------------------------------------------

    def predict(
        self,
        events: pd.DataFrame,
        *,
        verbose: bool = True,
        remove_empty_segments: bool | None = None,
    ) -> tuple[np.ndarray, list]:
        """End-to-end inference: extract features in parallel, then forward.

        Returns predictions of shape ``(n_kept_segments, n_vertices)``
        and the matching list of segments, identical in contract to
        :meth:`TribeModel.predict`.
        """
        if self._closed:
            raise RuntimeError("TribePipeline has been closed; create a new one")
        if remove_empty_segments is None:
            remove_empty_segments = self.remove_empty_segments

        events = standardize_events(events)
        events = _add_dummy_categorical_events(events)

        data = self._xp.data
        extractors: dict[str, tp.Any] = {}
        for modality in data.features_to_use:
            ext = getattr(data, f"{modality}_feature", None)
            if ext is not None:
                extractors[modality] = ext
        if "Fmri" in events.type.unique():
            extractors["fmri"] = data.neuro
        extractors = _filter_relevant_extractors(extractors, events)

        # 1) Parallel feature extraction across GPUs.  Each extractor
        #    runs in its own thread; ``torch.cuda.device(idx)`` pins the
        #    thread-default CUDA device, so the extractor's internal
        #    ``model.to("cuda")`` lands on the right GPU.
        self._extract_in_parallel(extractors, events, verbose=verbose)

        # 2) subject_id is a cheap label-encoder, no GPU work.  Add it
        #    after extraction so it's part of the SegmentDataset.
        extractors["subject_id"] = data.subject_id

        # 3) Build segments + dataset (mirrors Data.get_loaders for split="all").
        TR: float = data.TR
        sel = np.ones(len(events), dtype=bool)
        segments = ns.segments.list_segments(
            events[sel],
            triggers=events[sel].type == "CategoricalEvent",
            stride=(data.duration_trs - data.overlap_trs_train) * TR,
            duration=data.duration_trs * TR,
            stride_drop_incomplete=data.stride_drop_incomplete,
        )
        if len(segments) == 0:
            logger.warning("No segments produced for the given events")
            n_vertices = int(self._xp._model.n_outputs)
            return np.empty((0, n_vertices), dtype=np.float32), []

        dataset = ns.dataloader.SegmentDataset(
            extractors=extractors,
            segments=segments,
            remove_incomplete_segments=False,
        )
        # Reuse the dataset's own collate fn if it exposes one; otherwise
        # let DataLoader fall back to the default collate.
        collate_fn = getattr(dataset, "collate_fn", None)
        if collate_fn is None:
            base_loader = dataset.build_dataloader(
                shuffle=False,
                num_workers=0,
                batch_size=data.batch_size,
            )
            collate_fn = getattr(base_loader, "collate_fn", None)

        # 4) Optimised model forward (autocast, pinned host buffer,
        #    async copies, multi-GPU sharding) over the pre-built dataset.
        return self._runner.predict(
            dataset=dataset,
            collate_fn=collate_fn,
            verbose=verbose,
            remove_empty_segments=remove_empty_segments,
        )

    # -- parallel extraction --------------------------------------------

    def _extract_in_parallel(
        self,
        extractors: dict[str, tp.Any],
        events: pd.DataFrame,
        *,
        verbose: bool,
    ) -> None:
        """Run each extractor's ``prepare`` on its own GPU concurrently.

        Strategy: assign extractors to GPUs round-robin.  Each extractor
        runs on its own thread; the thread sets the default CUDA device
        with ``torch.cuda.device(idx)`` so anything the extractor does
        with a bare ``"cuda"`` device lands on the right GPU.

        Single-GPU (or CPU) machines fall back to sequential execution
        - threading buys nothing there and complicates ordering.
        """
        gpu_devices = [d for d in self.devices if d.type == "cuda"]
        items = list(extractors.items())
        if not items:
            return

        if len(gpu_devices) <= 1 or len(items) == 1:
            # Sequential path: simpler, identical to the legacy order.
            for name, extractor in items:
                logger.info("Extracting %s on %s", name, self.devices[0])
                extractor.prepare(events)
                _free_extractor_model(extractor)
            return

        # Parallel path: pin one extractor per GPU (round-robin).
        log_lock = threading.Lock()

        def _run_one(name: str, extractor: tp.Any, gpu_idx: int) -> None:
            with torch.cuda.device(gpu_idx):
                with log_lock:
                    logger.info("Extracting %s on cuda:%d", name, gpu_idx)
                extractor.prepare(events)
                _free_extractor_model(extractor)

        with ThreadPoolExecutor(
            max_workers=min(len(items), len(gpu_devices)),
            thread_name_prefix="tribe-extract",
        ) as pool:
            futures = []
            for i, (name, extractor) in enumerate(items):
                gpu_idx = gpu_devices[i % len(gpu_devices)].index
                futures.append(pool.submit(_run_one, name, extractor, gpu_idx))
            for f in futures:
                f.result()  # surface any extractor exceptions

        # One final cache empty after all extractor models are freed.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
