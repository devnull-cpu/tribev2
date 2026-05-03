# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Optimized single-PC inference pipeline for :class:`TribeModel`.

This is a standalone, pure-PyTorch alternative to
:meth:`tribev2.demo_utils.TribeModel.predict`.  It targets one machine
with one or more GPUs and removes the per-batch GPU/CPU sync points,
the unbounded prediction list, and the PyTorch-Lightning + FSDP
machinery that the legacy path drags in for multi-GPU.

Key optimisations
-----------------
* H2D copies use ``non_blocking=True`` together with a pinned-memory
  DataLoader and ``persistent_workers``.
* Forward pass runs under ``torch.inference_mode`` and ``torch.autocast``
  (``bf16`` on Ampere+, ``fp16`` on older CUDA, ``fp32`` on CPU).
* The "keep" mask is computed CPU-side from segment metadata *before*
  the forward, then applied on-device with ``index_select`` so the only
  D2H copy moves exactly the kept rows.
* Predictions are written into a single pre-allocated pinned host buffer
  (no Python list, no final ``np.concatenate``).
* Multi-GPU on the same machine: one model replica per device, one
  thread per device, deterministic strided sharding by parent segment
  index.  No multiprocessing, no IPC, no checkpoint reload.

Usage::

    from tribev2.demo_utils import TribeModel
    from tribev2.inference import InferenceRunner

    model = TribeModel.from_pretrained("facebook/tribev2")
    events = model.get_events_dataframe(video_path="clip.mp4")
    with InferenceRunner(model) as runner:
        preds, segments = runner.predict(events)
"""

from __future__ import annotations

import copy
import gc
import logging
import os
import typing as tp
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

if tp.TYPE_CHECKING:
    from tribev2.demo_utils import TribeModel

logger = logging.getLogger(__name__)

DeviceLike = tp.Union[str, torch.device]
DtypeLike = tp.Union[str, torch.dtype, None]


def _select_devices(devices: DeviceLike | list[DeviceLike] | None) -> list[torch.device]:
    if devices is None:
        if torch.cuda.is_available():
            return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        return [torch.device("cpu")]
    if isinstance(devices, (str, torch.device)):
        return [torch.device(devices)]
    return [torch.device(d) for d in devices]


def _select_dtype(dtype: DtypeLike, device: torch.device) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None or dtype == "auto":
        if device.type != "cuda":
            return torch.float32
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    if isinstance(dtype, str):
        attr = getattr(torch, dtype, None)
        if not isinstance(attr, torch.dtype):
            raise ValueError(f"Unknown torch dtype name: {dtype!r}")
        return attr
    raise TypeError(f"Unsupported dtype: {dtype!r}")


def _explode_segments(parent_segments: list, TR: float) -> list:
    """Per-TR sub-segments for a list of parent segments.

    Mirrors the inner loop of :meth:`TribeModel.predict` so this path
    produces sub-segments in the exact same order as the legacy one.
    """
    out = []
    for segment in parent_segments:
        for t in np.arange(0, segment.duration - 1e-2, TR):
            out.append(segment.copy(offset=t, duration=TR))
    return out


def _compute_keep_mask(sub_segments: list, remove_empty: bool) -> np.ndarray:
    if not remove_empty:
        return np.ones(len(sub_segments), dtype=bool)
    return np.fromiter(
        (len(s.ns_events) > 0 for s in sub_segments),
        dtype=bool,
        count=len(sub_segments),
    )


def _to_device(batch, device: torch.device, non_blocking: bool = True):
    """Best-effort non-blocking move of a ``SegmentData`` batch to *device*.

    Falls back to copying ``batch.data`` tensors in place if the batch's
    own ``.to`` does not accept ``non_blocking``.
    """
    try:
        return batch.to(device, non_blocking=non_blocking)
    except TypeError:
        moved = batch.to(device)
        return moved


class _StridedIndexSampler(Sampler[int]):
    """Sequential sampler over ``range(rank, n, world_size)``."""

    def __init__(self, n: int, rank: int, world_size: int):
        self._indices = list(range(rank, n, world_size))

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


class InferenceRunner:
    """Optimized inference runner for a loaded :class:`TribeModel`.

    Parameters
    ----------
    tribe_model:
        A model already loaded via :meth:`TribeModel.from_pretrained`.
    devices:
        ``None`` selects every visible CUDA device (or CPU if none).
        Otherwise a single device string / object, or a list of them.
    dtype:
        Compute dtype for the forward pass.  ``"auto"`` picks ``bf16``
        on Ampere+ (SM>=80), ``fp16`` on older CUDA, ``fp32`` on CPU.
    batch_size:
        Override ``tribe_model.data.batch_size`` for the inference loader.
    num_workers:
        Per-rank DataLoader worker count.  Defaults to
        ``min(8, cpu_count()//2 // world_size)`` so the total worker
        process count stays bounded when sharding across multiple GPUs.
    prefetch_factor, pin_memory, persistent_workers:
        Forwarded to :class:`torch.utils.data.DataLoader`.  Pinned memory
        is auto-disabled when the target device is CPU.
    free_after:
        If ``True``, move replicas to CPU and call
        ``torch.cuda.empty_cache`` when :meth:`close` runs.
    """

    def __init__(
        self,
        tribe_model: "TribeModel",
        *,
        devices: DeviceLike | list[DeviceLike] | None = None,
        dtype: DtypeLike = "auto",
        batch_size: int | None = None,
        num_workers: int | None = None,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        free_after: bool = False,
    ):
        if getattr(tribe_model, "_model", None) is None:
            raise RuntimeError(
                "tribe_model has no loaded weights; call TribeModel.from_pretrained first"
            )
        self.tribe_model = tribe_model
        self.devices = _select_devices(devices)
        self.dtype = _select_dtype(dtype, self.devices[0])
        self.batch_size = batch_size
        if num_workers is None:
            cpu_total = os.cpu_count() or 2
            per_rank = (cpu_total // 2) // max(1, len(self.devices))
            num_workers = min(8, max(0, per_rank))
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.free_after = free_after

        base_model = tribe_model._model
        self._replicas: dict[torch.device, torch.nn.Module] = {}
        primary = self.devices[0]
        base_model.to(primary).eval()
        self._replicas[primary] = base_model
        for d in self.devices[1:]:
            replica = copy.deepcopy(base_model).to(d).eval()
            self._replicas[d] = replica

        self._executor: ThreadPoolExecutor | None = None
        if len(self.devices) > 1:
            self._executor = ThreadPoolExecutor(max_workers=len(self.devices))

        logger.info(
            "InferenceRunner ready: devices=%s dtype=%s workers=%d",
            [str(d) for d in self.devices],
            self.dtype,
            self.num_workers,
        )

    def __enter__(self) -> "InferenceRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self.free_after:
            for d, m in list(self._replicas.items()):
                if d.type == "cuda":
                    m.to("cpu")
            self._replicas.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def predict(
        self,
        events: pd.DataFrame | None = None,
        *,
        dataset: tp.Any = None,
        collate_fn: tp.Callable | None = None,
        verbose: bool = True,
        remove_empty_segments: bool | None = None,
    ) -> tuple[np.ndarray, list]:
        """Run inference and return ``(preds, segments)`` matching the legacy contract.

        Either pass ``events`` (the runner will build the dataset via
        ``tribe_model.data.get_loaders``) or pass a pre-built ``dataset``
        and ``collate_fn`` (used by :class:`TribePipeline` to skip the
        legacy ``Data.get_loaders`` orchestration).
        """
        if remove_empty_segments is None:
            remove_empty_segments = getattr(
                self.tribe_model, "remove_empty_segments", True
            )

        TR: float = self.tribe_model.data.TR
        primary_model = self._replicas[self.devices[0]]
        n_vertices = int(primary_model.n_outputs)
        n_output_timesteps = int(primary_model.n_output_timesteps)

        if dataset is None:
            if events is None:
                raise ValueError("Pass either events or a pre-built dataset")
            base_loader = self.tribe_model.data.get_loaders(
                events=events, split_to_build="all"
            )["all"]
            dataset = base_loader.dataset
            if collate_fn is None:
                collate_fn = getattr(base_loader, "collate_fn", None)
            base_batch_size = base_loader.batch_size
        else:
            base_batch_size = self.tribe_model.data.batch_size
        ds_segments = getattr(dataset, "segments", None)
        if ds_segments is None:
            raise AttributeError(
                "Underlying dataset has no .segments attribute; "
                "InferenceRunner relies on it to pre-compute output offsets."
            )
        parent_segments = list(ds_segments)
        n_parents = len(parent_segments)
        if n_parents == 0:
            return np.empty((0, n_vertices), dtype=np.float32), []

        all_sub_segments = _explode_segments(parent_segments, TR)
        # Each parent must produce exactly n_output_timesteps sub-segments
        # for the (B, D, T) -> (B*T, D) flatten to align with the keep mask.
        expected = n_parents * n_output_timesteps
        if len(all_sub_segments) != expected:
            raise RuntimeError(
                f"Sub-segment count mismatch: got {len(all_sub_segments)}, "
                f"expected n_parents*n_output_timesteps = {expected}. "
                "Check Data.duration_trs vs the model's n_output_timesteps."
            )
        keep_mask = _compute_keep_mask(all_sub_segments, remove_empty_segments)

        # Per-parent contiguous output rows: parent i owns rows
        # [parent_kept_offsets[i] : parent_kept_offsets[i+1]] in the
        # final output array.  This holds across all ranks because
        # each parent's predictions are contiguous in the flat layout.
        per_parent_keep = keep_mask.reshape(n_parents, n_output_timesteps)
        per_parent_kept_counts = per_parent_keep.sum(axis=1).astype(np.int64)
        parent_kept_offsets = np.zeros(n_parents + 1, dtype=np.int64)
        np.cumsum(per_parent_kept_counts, out=parent_kept_offsets[1:])
        n_kept_total = int(parent_kept_offsets[-1])
        n_samples_total = int(keep_mask.size)

        if n_kept_total == 0:
            logger.info(
                "Predicted 0 / %d segments (0.0%% kept)", n_samples_total
            )
            return np.empty((0, n_vertices), dtype=np.float32), []

        # Single pinned host buffer shared across ranks (each rank writes
        # to a disjoint set of parent-owned slices).
        out_pinned = torch.empty(
            (n_kept_total, n_vertices),
            dtype=torch.float32,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
        )
        segs_out: list = [None] * n_kept_total
        # Place segments synchronously up-front - they are CPU objects with no
        # dependence on the GPU forward.
        kept_global_idx = np.flatnonzero(keep_mask)
        for write_pos, sub_idx in enumerate(kept_global_idx):
            segs_out[write_pos] = all_sub_segments[int(sub_idx)]

        world_size = len(self.devices)

        def _run_rank(rank: int) -> int:
            device = self.devices[rank]
            model = self._replicas[device]
            sampler = _StridedIndexSampler(n_parents, rank=rank, world_size=world_size)
            bs = (
                self.batch_size
                if self.batch_size is not None
                else (base_batch_size or self.tribe_model.data.batch_size)
            )
            use_pin = self.pin_memory and device.type == "cuda"
            nw = self.num_workers
            loader_kwargs: dict[str, tp.Any] = dict(
                dataset=dataset,
                batch_size=bs,
                sampler=sampler,
                num_workers=nw,
                pin_memory=use_pin,
                drop_last=False,
                collate_fn=collate_fn,
            )
            if nw > 0:
                loader_kwargs["persistent_workers"] = self.persistent_workers
                loader_kwargs["prefetch_factor"] = self.prefetch_factor
            loader = DataLoader(**loader_kwargs)

            rank_indices = list(range(rank, n_parents, world_size))
            cursor = 0
            written = 0
            autocast_enabled = device.type == "cuda" and self.dtype != torch.float32
            pbar = tqdm(
                total=len(loader),
                disable=not verbose,
                desc=f"rank{rank}" if world_size > 1 else "predict",
                position=rank if world_size > 1 else 0,
                leave=(rank == 0),
            )
            with torch.inference_mode():
                for batch in loader:
                    n_in_batch = len(batch.segments)
                    parent_indices = rank_indices[cursor : cursor + n_in_batch]
                    cursor += n_in_batch

                    # Build local keep mask for this batch's parents from
                    # the pre-computed global mask - no per-segment .ns_events
                    # access here.
                    local_keep = per_parent_keep[parent_indices].reshape(-1)
                    local_kept_idx = np.flatnonzero(local_keep)
                    if local_kept_idx.size == 0:
                        pbar.update(1)
                        continue

                    kept_idx_t = torch.from_numpy(local_kept_idx).to(
                        device, non_blocking=True
                    )
                    batch = _to_device(batch, device, non_blocking=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=self.dtype,
                        enabled=autocast_enabled,
                    ):
                        y = model(batch)  # (B, D, T)
                    # (B, D, T) -> (B*T, D), keep stride matches sub-segment order
                    y = y.permute(0, 2, 1).reshape(-1, n_vertices)
                    y_kept = y.index_select(0, kept_idx_t)

                    # Slice writes per parent so disjointness across ranks holds.
                    src_pos = 0
                    for local_b, parent_idx in enumerate(parent_indices):
                        n_kept_p = int(per_parent_kept_counts[parent_idx])
                        if n_kept_p == 0:
                            continue
                        dst_lo = int(parent_kept_offsets[parent_idx])
                        dst_hi = dst_lo + n_kept_p
                        out_pinned[dst_lo:dst_hi].copy_(
                            y_kept[src_pos : src_pos + n_kept_p],
                            non_blocking=True,
                        )
                        src_pos += n_kept_p
                        written += n_kept_p
                    pbar.update(1)
            pbar.close()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return written

        if world_size == 1:
            _run_rank(0)
        else:
            assert self._executor is not None
            futures = [self._executor.submit(_run_rank, r) for r in range(world_size)]
            for f in futures:
                f.result()

        preds = out_pinned.numpy()
        # Ensure caller owns a regular (non-pinned) array; pinned buffer is
        # backed by the runner and would be reused on next predict() if we
        # ever cache it - safer to hand back a copy.  The cost is one
        # contiguous memcpy of ~tens of MB.
        preds = np.ascontiguousarray(preds)
        kept_pct = 100.0 * n_kept_total / max(n_samples_total, 1)
        logger.info(
            "Predicted %d / %d segments (%.1f%% kept)",
            n_kept_total,
            n_samples_total,
            kept_pct,
        )
        return preds, segs_out


def predict(
    tribe_model: "TribeModel",
    events: pd.DataFrame,
    **runner_kwargs: tp.Any,
) -> tuple[np.ndarray, list]:
    """One-shot convenience wrapper around :class:`InferenceRunner`."""
    verbose = runner_kwargs.pop("verbose", True)
    remove_empty_segments = runner_kwargs.pop("remove_empty_segments", None)
    with InferenceRunner(tribe_model, **runner_kwargs) as runner:
        return runner.predict(
            events,
            verbose=verbose,
            remove_empty_segments=remove_empty_segments,
        )
