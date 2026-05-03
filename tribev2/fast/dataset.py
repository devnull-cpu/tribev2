# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SegmentDataset replacement for the fast inference pipeline.

Reads per-event features from ``FeatureCache`` and assembles per-segment
``(L, D, T_seg)`` tensors via :func:`segments.slice_static_for_segment`
and :func:`segments.slice_temporal_for_segment`.

The output of :class:`FastSegmentDataset.__getitem__` and the collated
batch produced by :func:`fast_collate` are duck-typed to match
``neuralset.dataloader.SegmentData`` so the existing
``InferenceRunner`` (and the model's ``forward``) consume them
unchanged: ``batch.data: dict[str, Tensor]``, ``batch.segments: list``.
"""

from __future__ import annotations

import logging
import typing as tp
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from tribev2.fast.extractors import (
    AudioExtractor,
    TextExtractor,
    VideoExtractor,
    _BaseFastExtractor,
)
from tribev2.fast.segments import (
    FastEvent,
    FastSegment,
    slice_static_for_segment,
    slice_temporal_for_segment,
)

logger = logging.getLogger(__name__)


@dataclass
class FastBatch:
    """Duck-typed replacement for ``neuralset.dataloader.SegmentData``."""

    data: dict[str, torch.Tensor]
    segments: list[FastSegment]

    def to(self, device: torch.device, *, non_blocking: bool = False) -> "FastBatch":
        moved = {
            k: (
                v.to(device, non_blocking=non_blocking)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in self.data.items()
        }
        return FastBatch(data=moved, segments=self.segments)


class FastSegmentDataset(Dataset):
    """Builds per-segment per-modality tensors from cached features.

    ``__getitem__`` returns a ``FastBatch`` with batch dim 1; the
    default collate ``fast_collate`` concatenates them on the batch
    axis.

    ``static_modalities`` are those whose per-event features have no
    time axis (default: ``"text"``).  ``temporal_modalities`` have a
    trailing time axis (default: ``"audio"``, ``"video"``).
    """

    STATIC = ("text",)
    TEMPORAL = ("audio", "video")

    def __init__(
        self,
        events: list[FastEvent],
        segments: list[FastSegment],
        extractors: dict[str, _BaseFastExtractor],
        *,
        subject_to_id: dict[str, int] | None = None,
        segment_aggregation: str = "sum",
    ):
        self.events = events
        self.segments = segments
        self.extractors = extractors
        self.subject_to_id = subject_to_id or {}
        self.segment_aggregation = segment_aggregation
        self._event_lookup = {e.event_id: e for e in events}

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> FastBatch:
        seg = self.segments[idx]
        seg_events = [self._event_lookup[i] for i in seg.event_indices]

        data: dict[str, torch.Tensor] = {}

        for modality, extractor in self.extractors.items():
            paired = [
                (e, extractor.load_cached(e))
                for e in seg_events
                if _modality_accepts(modality, e.type) and extractor.has_cached(e)
            ]
            relevant = [p[0] for p in paired]
            feats = [p[1] for p in paired]
            if modality in self.STATIC:
                arr = slice_static_for_segment(
                    seg,
                    relevant,
                    feats,
                    frequency=extractor.config.frequency,
                    aggregation=self.segment_aggregation,
                )
            elif modality in self.TEMPORAL:
                arr = slice_temporal_for_segment(
                    seg,
                    relevant,
                    feats,
                    frequency=extractor.config.frequency,
                    aggregation=self.segment_aggregation,
                )
            else:
                raise ValueError(f"Unknown modality bucket for {modality!r}")
            if arr.size == 0:
                continue
            t = torch.from_numpy(arr)
            data[modality] = t.unsqueeze(0)  # add batch dim

        # Subject id: tiny long tensor, no extractor needed.
        sid = self.subject_to_id.get(seg.subject, 0)
        data["subject_id"] = torch.tensor([sid], dtype=torch.long)

        return FastBatch(data=data, segments=[seg])


def _modality_accepts(modality: str, ev_type: str) -> bool:
    if modality == "text":
        return ev_type == "Word"
    if modality == "audio":
        return ev_type in ("Audio", "Video")
    if modality == "video":
        return ev_type == "Video"
    return False


def fast_collate(batches: list[FastBatch]) -> FastBatch:
    """Concatenate per-item ``FastBatch`` objects on the batch axis."""
    if len(batches) == 1:
        return batches[0]
    out_data: dict[str, torch.Tensor] = {}
    keys: set[str] = set()
    for b in batches:
        keys.update(b.data.keys())
    for k in keys:
        tensors = []
        for b in batches:
            if k in b.data:
                tensors.append(b.data[k])
            elif k != "subject_id":
                ref = next((bb.data[k] for bb in batches if k in bb.data), None)
                if ref is not None:
                    zero = torch.zeros_like(ref)
                    tensors.append(zero)
        if not tensors:
            continue
        out_data[k] = _stack_with_pad(tensors)
    segs = [s for b in batches for s in b.segments]
    return FastBatch(data=out_data, segments=segs)


def _stack_with_pad(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Stack on dim 0 after padding the *last* dim to the max across the list."""
    max_T = max(t.shape[-1] for t in tensors)
    if all(t.shape[-1] == max_T for t in tensors):
        return torch.cat(tensors, dim=0)
    padded = []
    for t in tensors:
        if t.shape[-1] < max_T:
            pad = list(t.shape)
            pad[-1] = max_T - t.shape[-1]
            t = torch.cat([t, torch.zeros(*pad, dtype=t.dtype)], dim=-1)
        padded.append(t)
    return torch.cat(padded, dim=0)
