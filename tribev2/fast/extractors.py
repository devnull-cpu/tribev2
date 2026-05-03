# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pure-PyTorch feature extractors for TRIBE v2 inference.

From-scratch replacements for ``neuralset.extractors.text.HuggingFaceText``,
``neuralset.extractors.audio.Wav2VecBert``, and
``neuralset.extractors.video.HuggingFaceVideo``.  Each extractor:

* Computes per-event features and writes one ``.npy`` cache file per
  event (resumable).
* Runs the underlying HuggingFace model under
  ``torch.inference_mode`` + optional ``torch.autocast`` (bf16/fp16).
* Has **zero per-batch CPU/GPU sync points**: the LLaMA / Wav2Vec /
  V-JEPA forward stays on device, layer + token aggregation runs on
  device, and there is one async D2H copy per batch into a pinned
  host buffer.

Output format (matches neuralset's ``_get_data`` so segment slicing in
``segments.py`` produces what ``FmriEncoderModel`` expects):

* Text (``Word`` event): ``(n_layers_out, D)``  -- static, no time axis.
* Audio: ``(n_layers_out, D, T_event)``  -- T_event ~= ``duration *
  frequency``.
* Video: ``(n_layers_out, D, T_event)``  -- same.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tribev2.fast.cache import FeatureCache, stable_uid
from tribev2.fast.segments import (
    FastEvent,
    aggregate_layers,
    aggregate_tokens,
    select_layer_indices,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Shared config + base
# --------------------------------------------------------------------------- #


@dataclass
class ExtractorConfig:
    """Common config for the three modality extractors."""

    model_name: str
    layers: list[float]
    layer_aggregation: tp.Literal["mean", "sum", "group_mean"] | None = "group_mean"
    token_aggregation: tp.Literal["mean", "sum", "max", "first", "last"] | None = "mean"
    frequency: float = 2.0
    batch_size: int = 4
    dtype: torch.dtype = torch.bfloat16
    quantize: tp.Literal[None, "int8", "int4", "fp8"] = None


def _autocast_dtype(device: torch.device, requested: torch.dtype) -> torch.dtype:
    """Pick a safe autocast dtype: requested if cuda + supported, else fp32."""
    if device.type != "cuda":
        return torch.float32
    if requested == torch.bfloat16:
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    return requested


ENABLE_COMPILE = True


def _quantization_kwargs(config: ExtractorConfig) -> dict:
    if config.quantize is None or config.quantize == "fp8":
        return {}
    try:
        from transformers import BitsAndBytesConfig
        if config.quantize == "int8":
            return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
        if config.quantize == "int4":
            return {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)}
    except ImportError:
        logger.warning("bitsandbytes not installed, skipping quantization")
    return {}


def _wrap_fp8(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """Convert model to FP8 inference via torchao quantize_ API (SM89+ TensorWise)."""
    try:
        from torchao.quantization import quantize_
        from torchao.quantization import Float8WeightOnlyConfig
        from torchao.float8 import CastConfig, ScalingGranularity
        config = Float8WeightOnlyConfig(
            weight_dtype=torch.float8_e4m3fn,
        )
        quantize_(model, config)
        logger.info("%s: FP8 weight-only via torchao (tensorwise)", name)
        return model
    except Exception as e:
        logger.info("%s: torchao FP8 weight-only failed: %s", name, e)
    try:
        from torchao.quantization import quantize_
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        config = Float8DynamicActivationFloat8WeightConfig(
            activation_dtype=torch.float8_e4m3fn,
            weight_dtype=torch.float8_e4m3fn,
        )
        quantize_(model, config)
        logger.info("%s: FP8 dynamic via torchao (tensorwise)", name)
        return model
    except Exception as e:
        logger.warning("%s: FP8 setup failed: %s", name, e)
    return model


def _compile_model(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """Compile model with TensorRT backend, falling back to inductor, then eager."""
    if not ENABLE_COMPILE:
        return model
    try:
        import torch_tensorrt  # noqa: F401
        compiled = torch.compile(model, backend="torch_tensorrt",
                                  options={"enabled_precisions": {torch.half, torch.bfloat16}})
        logger.info("%s: compiled with TensorRT backend", name)
        return compiled
    except Exception as e:
        logger.info("%s: TensorRT unavailable (%s), trying inductor", name, e)
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        logger.info("%s: compiled with inductor (reduce-overhead)", name)
        return compiled
    except Exception as e:
        logger.info("%s: torch.compile failed (%s), using eager mode", name, e)
        return model


class _BaseFastExtractor:
    """Shared cache + uid plumbing for all three extractors."""

    name: str = "base"

    def __init__(self, config: ExtractorConfig, *, cache_dir: str | Path):
        self.config = config
        self.cache = FeatureCache(cache_dir, extractor_uid=self._extractor_uid())

    def _extractor_uid(self) -> str:
        cfg = self.config
        # Cache key intentionally excludes batch_size and dtype - those
        # are compute-only knobs and don't affect the feature contract.
        # frequency is included for audio / video (changes time count).
        return stable_uid(
            self.name,
            cfg.model_name,
            cfg.layers,
            cfg.layer_aggregation,
            cfg.token_aggregation,
            cfg.frequency,
        )

    def has_cached(self, event: FastEvent) -> bool:
        return self.cache.has(event.uid())

    def load_cached(self, event: FastEvent) -> np.ndarray:
        return self.cache.load(event.uid())

    def save(self, event: FastEvent, array: np.ndarray) -> None:
        self.cache.save(event.uid(), array)


# --------------------------------------------------------------------------- #
# 1. Text: LLaMA / generic causal LM
# --------------------------------------------------------------------------- #


class TextExtractor(_BaseFastExtractor):
    """Per-word contextualised LM embeddings.

    Replaces ``neuralset.extractors.text.HuggingFaceText`` with a
    sync-free batched forward.

    Token selection: for a ``Word`` event with text ``w`` and context
    ``ctx``, we tokenize ``ctx`` (left-truncated), find the boundary by
    re-tokenizing ``ctx[:-len(w)].rstrip()``, then pick the trailing
    ``n_target = n_tokens - n_pads - n_prefix`` tokens (>= 1).  Same
    formula as ``text.py:444-451`` in neuralset.
    """

    name = "text"

    def __init__(
        self,
        config: ExtractorConfig,
        *,
        cache_dir: str | Path,
        device: torch.device | None = None,
    ):
        super().__init__(config, cache_dir=cache_dir)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model: torch.nn.Module | None = None
        self._tokenizer: tp.Any = None
        self._pad_id: int | None = None
        self._n_layers: int | None = None
        self._layer_indices: list[int] | None = None

    def _load_model(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        kwargs: dict[str, tp.Any] = {}
        if self.config.model_name.lower().startswith("microsoft/phi"):
            kwargs["trust_remote_code"] = True
        tok = AutoTokenizer.from_pretrained(
            self.config.model_name, truncation_side="left", **kwargs
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self._tokenizer = tok
        self._pad_id = int(tok.eos_token_id)
        # Load model in bf16 directly when on CUDA + bf16 capable, to
        # halve VRAM and skip the autocast cost.
        load_dtype = _autocast_dtype(self.device, self.config.dtype)
        quant_kwargs = _quantization_kwargs(self.config)
        model = AutoModel.from_pretrained(
            self.config.model_name,
            torch_dtype=load_dtype,
            **kwargs,
            **quant_kwargs,
        )
        if not quant_kwargs:
            model.to(self.device)
        model.eval()
        if self.device.type == "cuda":
            model = _compile_model(model, "Text")
            if self.config.quantize == "fp8":
                model = _wrap_fp8(model, "Text")
        self._model = model
        # n_model_layers includes the embedding output (hidden_states[0])
        # plus one entry per transformer block.
        n_blocks = getattr(model.config, "num_hidden_layers", None)
        if n_blocks is None:
            n_blocks = getattr(model.config, "n_layer", None)
        if n_blocks is None:
            raise RuntimeError(
                f"Cannot infer hidden-layer count for {self.config.model_name}"
            )
        self._n_layers = int(n_blocks) + 1
        self._layer_indices = select_layer_indices(
            self.config.layers, self._n_layers
        )

    def free(self) -> None:
        """Drop the model from memory once features are cached."""
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Public entry point: extract every uncached word.
    # ------------------------------------------------------------------ #

    def prepare(self, events: list[FastEvent]) -> None:
        words = [e for e in events if e.type == "Word"]
        todo = [e for e in words if not self.has_cached(e)]
        if not todo:
            return
        if self._model is None:
            self._load_model()

        from tqdm import tqdm as _tqdm
        bs = self.config.batch_size
        n_batches = (len(todo) + bs - 1) // bs
        for batch_start in _tqdm(range(0, len(todo), bs), total=n_batches, desc="Text embeddings", unit="batch"):
            batch = todo[batch_start : batch_start + bs]
            self._process_batch(batch)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _process_batch(self, batch: list[FastEvent]) -> None:
        assert self._model is not None and self._tokenizer is not None
        assert self._pad_id is not None and self._layer_indices is not None
        target_words = [b.text or "" for b in batch]
        contexts = [b.context or b.text or "" for b in batch]
        if not all(contexts):
            raise ValueError(f"Empty context in batch: {target_words!r}")

        tok = self._tokenizer(
            contexts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # Move tokens to device with non_blocking; pin if possible.
        for k, v in tok.items():
            if isinstance(v, torch.Tensor):
                if not v.is_pinned() and v.device.type == "cpu":
                    try:
                        v = v.pin_memory()
                    except Exception:
                        pass
                tok[k] = v.to(self.device, non_blocking=True)

        # Forward.  AutoModel returns hidden_states tuple of length
        # n_model_layers when output_hidden_states=True.
        with torch.inference_mode():
            outputs = self._model(**tok, output_hidden_states=True)
        states = outputs.hidden_states  # tuple (n_layers,) of (B, T, D)
        hidden = torch.stack(states, dim=0)  # (L, B, T, D)
        del states, outputs

        input_ids = tok["input_ids"]  # (B, T)
        # Per-row pad count, computed on device.
        n_pads_per_row = (input_ids == self._pad_id).sum(dim=-1)  # (B,)
        n_tokens = input_ids.shape[-1]

        # Per-word: figure out how many trailing real tokens belong to the
        # target word, mean-pool them on device, then layer-aggregate on device.
        per_word_aggregated: list[torch.Tensor] = []  # each (n_layers_out, D)
        for i, (target, context) in enumerate(zip(target_words, contexts)):
            n_pads = int(n_pads_per_row[i].item())  # one scalar pull per row
            real_T = n_tokens - n_pads
            # Prefix-encoding to find target-word boundary (mirrors
            # neuralset.text.py:444-451).
            prefix = context[: -len(target)].rstrip() if len(target) else context
            if prefix:
                n_prefix = len(
                    self._tokenizer.encode(prefix, add_special_tokens=False)
                )
            else:
                n_prefix = 0
            n_target = max(1, real_T - n_prefix)
            # Slice: hidden has shape (L, B, T, D); we want
            # hidden[:, i, real_T - n_target : real_T, :]  -> (L, n_target, D)
            tok_slice = hidden[:, i, real_T - n_target : real_T, :]
            # Token aggregation on device.
            aggregated = _aggregate_tokens_2d(
                tok_slice, mode=self.config.token_aggregation
            )  # (L, D)
            layer_agg = aggregate_layers(
                aggregated, self._layer_indices, self.config.layer_aggregation
            )  # (L_out, D)
            per_word_aggregated.append(layer_agg.detach().clone())

        del hidden, tok, n_pads_per_row, input_ids
        torch.cuda.empty_cache() if self.device.type == "cuda" else None

        stacked = torch.stack(per_word_aggregated, dim=0)
        del per_word_aggregated
        host = torch.empty(stacked.shape, dtype=torch.float32, pin_memory=False)
        host.copy_(stacked.detach(), non_blocking=False)
        del stacked
        host_np = host.numpy()

        for i, ev in enumerate(batch):
            self.save(ev, host_np[i])

    @property
    def n_layers_out(self) -> int:
        if self._layer_indices is None:
            self._load_model()
        return _layer_out_count(
            self._layer_indices or [], self.config.layer_aggregation
        )


def _aggregate_tokens_2d(latents: torch.Tensor, mode: str | None) -> torch.Tensor:
    """Token aggregation for ``(L, T, D)`` -> ``(L, D)`` (or ``(L, T, D)``)."""
    if mode is None:
        return latents
    if mode == "mean":
        return latents.mean(dim=1)
    if mode == "sum":
        return latents.sum(dim=1)
    if mode == "max":
        return latents.max(dim=1).values
    if mode == "first":
        return latents[:, 0]
    if mode == "last":
        return latents[:, -1]
    raise ValueError(f"Unknown token aggregation: {mode}")


def _layer_out_count(layer_indices: list[int], aggregation: str | None) -> int:
    if not layer_indices:
        return 0
    if aggregation in (None,):
        return len(layer_indices)
    if aggregation == "group_mean":
        return max(1, len(layer_indices) - 1)
    return 1  # mean / sum collapse to a single layer-axis


# --------------------------------------------------------------------------- #
# 2. Audio: Wav2Vec2Bert / generic HuggingFace audio
# --------------------------------------------------------------------------- #


class AudioExtractor(_BaseFastExtractor):
    """Per-audio-event embeddings via a HuggingFace audio model.

    Replaces ``neuralset.extractors.audio.Wav2VecBert`` /
    ``HuggingFaceAudio``.  One forward per event (audio events arrive
    one at a time per file/clip - the model already batches over time).
    """

    name = "audio"

    def __init__(
        self,
        config: ExtractorConfig,
        *,
        cache_dir: str | Path,
        device: torch.device | None = None,
        norm_audio: bool = True,
    ):
        super().__init__(config, cache_dir=cache_dir)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.norm_audio = norm_audio
        self._model: torch.nn.Module | None = None
        self._feature_extractor: tp.Any = None
        self._n_layers: int | None = None
        self._layer_indices: list[int] | None = None

    def _load_model(self) -> None:
        from transformers import AutoFeatureExtractor, AutoModel

        load_dtype = _autocast_dtype(self.device, self.config.dtype)
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.config.model_name
        )
        quant_kwargs = _quantization_kwargs(self.config)
        model = AutoModel.from_pretrained(
            self.config.model_name, torch_dtype=load_dtype, **quant_kwargs
        )
        if not quant_kwargs:
            model.to(self.device)
        model.eval()
        if self.device.type == "cuda":
            model = _compile_model(model, "Audio")
            if self.config.quantize == "fp8":
                model = _wrap_fp8(model, "Audio")
        self._model = model
        n_blocks = getattr(model.config, "num_hidden_layers", None)
        if n_blocks is None:
            raise RuntimeError(
                f"Cannot infer hidden-layer count for {self.config.model_name}"
            )
        self._n_layers = int(n_blocks) + 1
        self._layer_indices = select_layer_indices(
            self.config.layers, self._n_layers
        )

    def free(self) -> None:
        self._model = None
        self._feature_extractor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def input_frequency(self) -> int:
        if self._feature_extractor is None:
            self._load_model()
        return int(self._feature_extractor.sampling_rate)

    def prepare(self, events: list[FastEvent]) -> None:
        # Audio extractor handles Audio AND Video events (it pulls the
        # audio track from videos - matching neuralset behaviour).
        relevant = [e for e in events if e.type in ("Audio", "Video")]
        todo = [e for e in relevant if not self.has_cached(e)]
        if not todo:
            return
        if self._model is None:
            self._load_model()
        from tqdm import tqdm as _tqdm
        for ev in _tqdm(todo, desc="Audio embeddings", unit="event"):
            self._process_event(ev)

    def _process_event(self, event: FastEvent) -> None:
        assert self._model is not None
        assert self._feature_extractor is not None
        assert self._layer_indices is not None
        wav = _read_wav_for_event(event, target_sr=self.input_frequency)
        if self.norm_audio:
            wav = (wav - wav.mean()) / (1e-8 + wav.std())

        max_chunk_s = 30.0
        chunk_samples = int(max_chunk_s * self.input_frequency)
        total_samples = wav.shape[0]
        target_T = max(1, _round_int(event.duration * self.config.frequency))
        autocast_dtype = _autocast_dtype(self.device, self.config.dtype)

        if total_samples <= chunk_samples * 1.5:
            chunks = [wav]
        else:
            chunks = [wav[i:i + chunk_samples] for i in range(0, total_samples, chunk_samples)]

        chunk_hiddens = []
        for chunk_wav in chunks:
            feats = self._feature_extractor(
                chunk_wav.numpy(),
                return_tensors="pt",
                sampling_rate=self.input_frequency,
                do_normalize=self.norm_audio,
            )
            ftensor = feats.get("input_features", feats.get("input_values"))
            ftensor = ftensor.to(self.device, dtype=torch.float32, non_blocking=True)
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=autocast_dtype,
                enabled=(self.device.type == "cuda" and autocast_dtype != torch.float32),
            ):
                outputs = self._model(ftensor, output_hidden_states=True)
            states = outputs.hidden_states
            h = torch.stack(states, dim=0).squeeze(1).transpose(-1, -2)  # (L, D, T_chunk)
            chunk_hiddens.append(h)
            del ftensor, outputs, states

        if len(chunk_hiddens) > 1:
            hidden = torch.cat(chunk_hiddens, dim=-1)  # (L, D, T_total)
            del chunk_hiddens
        else:
            hidden = chunk_hiddens[0]

        if hidden.shape[-1] != target_T:
            hidden = F.interpolate(
                hidden.float(),
                size=target_T,
                mode="linear",
                align_corners=False,
            )
        layer_agg = aggregate_layers(
            hidden, self._layer_indices, self.config.layer_aggregation
        )
        del hidden
        host = torch.empty(layer_agg.shape, dtype=torch.float32, pin_memory=False)
        host.copy_(layer_agg.detach(), non_blocking=False)
        self.save(event, host.numpy())

    @property
    def n_layers_out(self) -> int:
        if self._layer_indices is None:
            self._load_model()
        return _layer_out_count(
            self._layer_indices or [], self.config.layer_aggregation
        )


def _read_wav_for_event(event: FastEvent, *, target_sr: int) -> torch.Tensor:
    """Read audio for an event and resample to ``target_sr`` (mono, 1-D)."""
    if event.filepath is None:
        raise ValueError(f"Event has no filepath: {event}")
    if event.type == "Audio":
        import soundfile as sf

        data, sr = sf.read(event.filepath, always_2d=True, dtype="float32")
        wav = torch.from_numpy(data)  # (n_samples, n_channels)
    elif event.type == "Video":
        from moviepy import VideoFileClip

        with VideoFileClip(event.filepath) as clip:
            audio = clip.audio
            if audio is None:
                raise ValueError(f"Video has no audio track: {event.filepath}")
            sr = int(audio.fps)
            arr = audio.to_soundarray()
        wav = torch.tensor(arr, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported event type for audio: {event.type!r}")
    # Mono.
    if wav.dim() == 2:
        wav = wav.mean(dim=1)
    # Trim to the event's start..start+duration window in the source file.
    # FastEvent.start is in the source file's timeline; for whole-file
    # events the window is the full file (start=0, duration=file_duration).
    start_sample = int(round(event.start * sr))
    end_sample = start_sample + int(round(event.duration * sr))
    if 0 <= start_sample < wav.numel() and end_sample > start_sample:
        wav = wav[start_sample:end_sample]
    # Resample to target_sr if needed.
    if sr != target_sr:
        wav = _resample_1d(wav, sr, target_sr)
    return wav


def _resample_1d(wav: torch.Tensor, src_sr: int, tgt_sr: int) -> torch.Tensor:
    """Linear-phase resampling via julius (matches neuralset's choice)."""
    try:
        import julius

        return julius.resample_frac(wav, src_sr, tgt_sr)
    except ImportError:
        # Torchaudio fallback if julius is not installed.
        import torchaudio.functional as taf

        return taf.resample(wav, src_sr, tgt_sr)


def _round_int(x: float) -> int:
    return int(math.floor(x + 0.5))


# --------------------------------------------------------------------------- #
# 3. Video: V-JEPA2
# --------------------------------------------------------------------------- #


class VideoExtractor(_BaseFastExtractor):
    """Per-video-event V-JEPA2 embeddings.

    Replaces ``neuralset.extractors.video.HuggingFaceVideo`` for the
    ``model.predict_hidden_states`` path (V-JEPA2 / VideoMAE / X-CLIP).
    Output shape per event: ``(n_layers_out, D, T_event)`` where
    ``T_event = round(duration * frequency)``.

    Key optimisation vs neuralset: the per-window loop accumulates
    embeddings on-device into a single ``(T_event, L, D)`` tensor and
    we copy to host once at the end of the event - vs neuralset's
    per-window ``.cpu().numpy()`` (`video.py:306`).
    """

    name = "video"

    def __init__(
        self,
        config: ExtractorConfig,
        *,
        cache_dir: str | Path,
        device: torch.device | None = None,
        clip_duration: float = 4.0,
        num_frames: int = 64,
    ):
        super().__init__(config, cache_dir=cache_dir)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.clip_duration = clip_duration
        self.num_frames = num_frames
        self._model: torch.nn.Module | None = None
        self._processor: tp.Any = None
        self._n_layers: int | None = None
        self._layer_indices: list[int] | None = None

    def _load_model(self) -> None:
        from transformers import AutoModel

        load_dtype = _autocast_dtype(self.device, self.config.dtype)
        self._processor = None
        quant_kwargs = _quantization_kwargs(self.config)
        model = AutoModel.from_pretrained(
            self.config.model_name, torch_dtype=load_dtype, **quant_kwargs
        )
        if not quant_kwargs:
            model.to(self.device)
        model.eval()
        if self.device.type == "cuda":
            model = _compile_model(model, "Video")
            if self.config.quantize == "fp8":
                model = _wrap_fp8(model, "Video")
        self._model = model
        # Try to infer transformer-block count from the vision config.
        cfg = getattr(model, "config", None)
        n_blocks = None
        for attr in ("num_hidden_layers", "depth"):
            n_blocks = getattr(cfg, attr, None)
            if n_blocks is not None:
                break
        if n_blocks is None and hasattr(cfg, "vision_config"):
            n_blocks = getattr(cfg.vision_config, "num_hidden_layers", None)
        if n_blocks is None:
            raise RuntimeError(
                f"Cannot infer hidden-layer count for {self.config.model_name}"
            )
        self._n_layers = int(n_blocks) + 1
        self._layer_indices = select_layer_indices(
            self.config.layers, self._n_layers
        )
        # Default num_frames from model config if available.
        for attr_path in ("num_frames", ("vision_config", "num_frames")):
            obj = cfg
            if isinstance(attr_path, tuple):
                for a in attr_path:
                    obj = getattr(obj, a, None)
                    if obj is None:
                        break
            else:
                obj = getattr(cfg, attr_path, None)
            if isinstance(obj, int) and obj > 0:
                self.num_frames = obj
                break

    def free(self) -> None:
        self._model = None
        self._processor = None
        self._gpu_transform = None
        self._pinned_buf = None
        self._transfer_stream = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare(self, events: list[FastEvent]) -> None:
        relevant = [e for e in events if e.type == "Video"]
        todo = [e for e in relevant if not self.has_cached(e)]
        if not todo:
            return
        if self._model is None:
            self._load_model()
        from tqdm import tqdm as _tqdm
        for ev in _tqdm(todo, desc="Video events", unit="event"):
            self._process_event(ev)

    def _process_event(self, event: FastEvent) -> None:
        assert self._model is not None and self._layer_indices is not None
        if event.filepath is None:
            raise ValueError(f"Video event has no filepath: {event}")

        T_event = max(1, _round_int(event.duration * self.config.frequency))
        times = np.linspace(0, event.duration, T_event + 1)[1:]
        subtimes = [
            k / self.num_frames * self.clip_duration
            for k in reversed(range(self.num_frames))
        ]

        all_frame_times = []
        for t in times:
            for st in subtimes:
                all_frame_times.append(max(0.0, event.start + t - st))

        from decord import VideoReader
        vr = VideoReader(event.filepath)
        fps = vr.get_avg_fps()
        num_vr_frames = len(vr)
        frame_indices = [min(int(round(ft, 3) * fps), num_vr_frames - 1) for ft in all_frame_times]
        unique_indices = sorted(set(frame_indices))
        logger.info("Decord: extracting %d unique frames (%d requested) from %s",
                     len(unique_indices), len(frame_indices), event.filepath)
        batch = vr.get_batch(unique_indices).asnumpy()
        idx_to_frame = {fi: batch[i] for i, fi in enumerate(unique_indices)}
        del batch, vr
        all_frames = [idx_to_frame[fi] for fi in frame_indices]
        del idx_to_frame

        out_device: torch.Tensor | None = None
        autocast_dtype = _autocast_dtype(self.device, self.config.dtype)
        fwd_key = "pixel_values_videos" if "vjepa2" in self.config.model_name else "pixel_values"
        batch_size = self.config.batch_size

        from tqdm import tqdm as _tqdm
        for batch_start in _tqdm(range(0, len(times), batch_size), total=(len(times) + batch_size - 1) // batch_size, desc="Encoding video", unit="batch"):
            batch_end = min(batch_start + batch_size, len(times))
            batch_pv = []
            for k in range(batch_start, batch_end):
                start = k * self.num_frames
                frames_np = np.stack(all_frames[start:start + self.num_frames], axis=0)
                pv = self._encode_frames(frames_np)  # (1, T, C, H, W)
                batch_pv.append(pv.squeeze(0))  # (T, C, H, W)
            batched = torch.stack(batch_pv)  # (B, T, C, H, W)
            del batch_pv
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=autocast_dtype,
                enabled=(self.device.type == "cuda" and autocast_dtype != torch.float32),
            ):
                out = self._model(
                    **{fwd_key: batched}, output_hidden_states=True
                )
            del batched
            states = out.hidden_states
            hidden = torch.stack(states, dim=0)  # (L, B, n_tokens, D)
            for i in range(batch_end - batch_start):
                h = hidden[:, i]  # (L, n_tokens, D)
                h = aggregate_tokens(h, self.config.token_aggregation)  # (L, D)
                h = aggregate_layers(h, self._layer_indices, self.config.layer_aggregation)  # (L_out, D)
                if out_device is None:
                    out_device = torch.empty(
                        (T_event, *h.shape),
                        dtype=h.dtype,
                        device=self.device,
                    )
                out_device[batch_start + i] = h
            del hidden, states, out
        del all_frames

        assert out_device is not None
        # out_device: (T_event, L_out, D) or (T_event, D)
        # Cache format: (L_out, D, T_event) or (D, T_event)
        if out_device.dim() == 3:
            out_device = out_device.permute(1, 2, 0).contiguous()
        elif out_device.dim() == 2:
            out_device = out_device.permute(1, 0).contiguous()
        host = torch.empty(out_device.shape, dtype=torch.float32, pin_memory=False)
        host.copy_(out_device.detach(), non_blocking=False)
        self.save(event, host.numpy())

    _gpu_transform: torch.nn.Module | None = None
    _pinned_buf: torch.Tensor | None = None
    _transfer_stream: torch.cuda.Stream | None = None

    def _get_gpu_transform(self) -> torch.nn.Module:
        if self._gpu_transform is None:
            from torchvision import transforms as T
            self._gpu_transform = torch.nn.Sequential(
                T.Resize(292, antialias=True),
                T.CenterCrop(256),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ).to(self.device)
        return self._gpu_transform

    def _get_transfer_stream(self) -> torch.cuda.Stream:
        if self._transfer_stream is None:
            self._transfer_stream = torch.cuda.Stream()
        return self._transfer_stream

    def _encode_frames(self, frames_np: np.ndarray) -> torch.Tensor:
        """Convert raw uint8 frames (T, H, W, 3) into model input via GPU transforms."""
        shape = frames_np.shape
        if self._pinned_buf is None or self._pinned_buf.shape != shape:
            self._pinned_buf = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
        self._pinned_buf.copy_(torch.from_numpy(frames_np))
        stream = self._get_transfer_stream()
        with torch.cuda.stream(stream):
            t = self._pinned_buf.to(device=self.device, non_blocking=True, dtype=torch.float32)
            t = t.permute(0, 3, 1, 2) / 255.0  # NHWC → NCHW, rescale
            t = self._get_gpu_transform()(t)  # resize, crop, normalize — all on GPU
        torch.cuda.current_stream().wait_stream(stream)
        return t.unsqueeze(0)  # (1, T, 3, 256, 256)

    @property
    def n_layers_out(self) -> int:
        if self._layer_indices is None:
            self._load_model()
        return _layer_out_count(
            self._layer_indices or [], self.config.layer_aggregation
        )
