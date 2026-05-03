# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Segment construction + per-modality slicing.

This is a from-scratch replacement for ``neuralset.segments.list_segments``
+ the ``TimedArray`` time-alignment dance that
``BaseExtractor._tarrays_to_tensor`` performs inside
``SegmentDataset.__getitem__``.  The behaviour is matched but the
implementation is straight numpy (no shared event-store, no UUID
registry, no DataFrame round-trips per segment).

Per-modality output contract (matches what ``FmriEncoderModel`` expects):

* Static / per-event-static (text): ``(L, D, T_seg)`` where T_seg =
  ``int(round(seg.duration * extractor_freq))``.  Each event's static
  ``(L, D)`` feature is broadcast across the time samples covered by
  ``[event.start, event.start + event.duration]`` within the segment
  and the per-time contributions are summed (or mean-aggregated).
* Temporal (audio / video): ``(L, D, T_seg)``.  Each event's
  ``(L, D, T_event)`` feature is sliced along the time axis to the
  overlap with the segment and copied into the matching segment-time
  range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch


@dataclass
class FastEvent:
    """Lightweight event record - replaces neuralset.events.Event."""

    event_id: int
    type: str
    start: float
    duration: float
    timeline: str
    subject: str
    # Per-event payload depending on type:
    text: str | None = None
    context: str | None = None
    filepath: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def stop(self) -> float:
        return self.start + self.duration

    def uid(self) -> str:
        """Stable id used as the cache key for this event's per-event features.

        For text we key on ``(text, context)``.  For audio/video we key
        on ``(filepath, start, duration)`` so the same file segmented
        differently gets different cache entries.
        """
        if self.type == "Word":
            return f"text|{self.text}|{self.context or ''}"
        return f"{self.type}|{self.filepath}|{self.start:.6f}|{self.duration:.6f}"


@dataclass
class FastSegment:
    """Segment record - replaces neuralset.segments.Segment.

    Carries a back-reference to the parent ``events`` list so
    :meth:`copy` and :attr:`ns_events` can reconstruct the events
    overlapping a sub-window without re-querying any external store.
    The reference is excluded from comparisons / repr to keep the
    object lightweight.
    """

    start: float
    duration: float
    timeline: str
    subject: str
    event_indices: list[int]  # indices into the parent event list
    _events_ref: list["FastEvent"] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def stop(self) -> float:
        return self.start + self.duration

    @property
    def ns_events(self) -> list["FastEvent"]:
        """Events overlapping this segment (matches neuralset.Segment.ns_events).

        Used by ``InferenceRunner._compute_keep_mask`` (only its length
        matters) and by callers who want to inspect the events behind a
        returned prediction row.
        """
        if self._events_ref is None:
            return []
        return [self._events_ref[i] for i in self.event_indices]

    def copy(self, *, offset: float, duration: float) -> "FastSegment":
        """Return a sub-segment shifted by ``offset`` with the given duration.

        Matches the calling convention used by
        ``InferenceRunner._explode_segments`` to break each parent
        segment into per-TR sub-segments.  ``event_indices`` is filtered
        to events that overlap the new ``[start, start+duration)``
        window so ``len(sub.ns_events)`` correctly reports whether the
        sub-window contains any real events.
        """
        new_start = self.start + offset
        new_stop = new_start + duration
        if self._events_ref is not None:
            new_indices = [
                i
                for i in self.event_indices
                if self._events_ref[i].start < new_stop
                and self._events_ref[i].stop > new_start
            ]
        else:
            new_indices = list(self.event_indices)
        return FastSegment(
            start=new_start,
            duration=duration,
            timeline=self.timeline,
            subject=self.subject,
            event_indices=new_indices,
            _events_ref=self._events_ref,
        )


# --------------------------------------------------------------------------- #
# Event dataframe -> FastEvent list
# --------------------------------------------------------------------------- #


def events_from_dataframe(events: pd.DataFrame) -> list[FastEvent]:
    """Convert a standardised events DataFrame into ``FastEvent``s.

    The DataFrame must already have ``standardize_events`` applied (the
    pipeline does this).  Required columns: ``type``, ``start``,
    ``duration``, ``timeline``, ``subject``.  Optional: ``text``,
    ``context``, ``filepath``.
    """
    out: list[FastEvent] = []
    for i, row in enumerate(events.itertuples(index=False)):
        out.append(
            FastEvent(
                event_id=i,
                type=str(row.type),
                start=float(row.start),
                duration=float(row.duration),
                timeline=str(row.timeline),
                subject=str(row.subject),
                text=getattr(row, "text", None) if hasattr(row, "text") else None,
                context=(
                    getattr(row, "context", None) if hasattr(row, "context") else None
                ),
                filepath=(
                    getattr(row, "filepath", None)
                    if hasattr(row, "filepath")
                    else None
                ),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Segment construction
# --------------------------------------------------------------------------- #


def list_segments(
    events: list[FastEvent],
    *,
    duration: float,
    stride: float,
    stride_drop_incomplete: bool = False,
) -> list[FastSegment]:
    """Build segments by sliding a window over each timeline's span.

    Mirrors ``neuralset.segments.list_segments`` for the
    ``triggers = type == 'CategoricalEvent'`` case the inference
    pipeline uses.  One segment per ``(timeline, window_start)`` pair.

    For each timeline we look at its CategoricalEvent (the dummy
    sentinel injected by the pipeline) for the ``start`` / ``stop``
    bounds, then slide windows of length ``duration`` with step
    ``stride``.  Each segment's ``event_indices`` is the list of event
    positions in the input list that overlap the window's time range.
    """
    by_timeline: dict[str, list[FastEvent]] = {}
    for ev in events:
        by_timeline.setdefault(ev.timeline, []).append(ev)

    segments: list[FastSegment] = []
    for timeline, ev_list in by_timeline.items():
        triggers = [e for e in ev_list if e.type == "CategoricalEvent"]
        if not triggers:
            continue
        if len(triggers) > 1:
            raise ValueError(
                f"Timeline {timeline!r} has {len(triggers)} CategoricalEvent "
                "triggers; expected exactly one"
            )
        trig = triggers[0]
        starts = _strided_starts(
            trig.start, trig.stop, stride, duration, stride_drop_incomplete
        )
        for s in starts:
            stop = s + duration
            ev_idx = [
                ev.event_id
                for ev in ev_list
                if ev.type != "CategoricalEvent" and ev.start < stop and ev.stop > s
            ]
            segments.append(
                FastSegment(
                    start=s,
                    duration=duration,
                    timeline=timeline,
                    subject=trig.subject,
                    event_indices=ev_idx,
                    _events_ref=events,
                )
            )
    return segments


def _strided_starts(
    start: float, stop: float, stride: float, duration: float, drop_incomplete: bool
) -> list[float]:
    """Window start times.  Matches ``_prepare_strided_windows`` semantics."""
    span = stop - start
    if span <= 0:
        return []
    if drop_incomplete:
        n = int(math.floor((span - duration) / stride + 1e-9)) + 1
    else:
        n = int(math.ceil(span / stride - 1e-9))
    n = max(1, n)
    return [start + i * stride for i in range(n)]


# --------------------------------------------------------------------------- #
# Per-segment slicing for static and temporal features
# --------------------------------------------------------------------------- #


def slice_static_for_segment(
    seg: FastSegment,
    seg_events: Sequence[FastEvent],
    feats: Sequence[np.ndarray],
    frequency: float,
    aggregation: str = "sum",
) -> np.ndarray:
    """Aggregate static per-event features into a (L, D, T_seg) tensor.

    Mirrors the ``_tarrays_to_tensor(aggregation in {"sum","mean"}) + +=
    of frequency-0 TimedArrays into a frequency-``frequency`` segment
    container`` chain in neuralset.

    Each event's ``(L, D)`` (or ``(D,)``) feature is broadcast across
    the time samples covered by ``[event.start, event.stop]`` clipped
    to the segment, summed (``"sum"``) or running-mean-aggregated
    (``"mean"``).
    """
    if len(feats) == 0:
        return np.zeros((0,), dtype=np.float32)
    sample = feats[0]
    feat_shape = sample.shape  # (L, D) or (D,)
    T = max(1, int(round(seg.duration * frequency)))
    out_shape = feat_shape + (T,)
    out = np.zeros(out_shape, dtype=np.float32)
    counts: np.ndarray | None = None
    if aggregation == "mean":
        counts = np.zeros(T, dtype=np.int64)

    seg_stop = seg.start + seg.duration
    for ev, feat in zip(seg_events, feats):
        ov_start = max(ev.start, seg.start)
        ov_stop = min(ev.stop, seg_stop)
        if ov_stop <= ov_start:
            continue
        i0 = int(round((ov_start - seg.start) * frequency))
        i1 = int(round((ov_stop - seg.start) * frequency))
        if i1 <= i0:
            i1 = i0 + 1
        i0 = max(0, min(T, i0))
        i1 = max(0, min(T, i1))
        if i1 == i0:
            continue
        feat32 = feat.astype(np.float32, copy=False)
        if aggregation == "mean":
            assert counts is not None
            c = counts[i0:i1]
            upd = c / (1.0 + c)
            sl = out[..., i0:i1]
            sl *= upd
            sl += (1.0 - upd) * feat32[..., None]
            counts[i0:i1] = c + 1
        else:
            out[..., i0:i1] += feat32[..., None]
    return out


def slice_temporal_for_segment(
    seg: FastSegment,
    seg_events: Sequence[FastEvent],
    feats: Sequence[np.ndarray],
    frequency: float,
    aggregation: str = "sum",
) -> np.ndarray:
    """Aggregate temporal per-event features into ``(L, D, T_seg)``.

    Each event has features of shape ``(..., T_event)`` where
    ``T_event ~= round(event.duration * frequency)``.  We slice each
    event's feature tensor along the time axis to the overlap with the
    segment and copy into the matching destination range.
    """
    if len(feats) == 0:
        return np.zeros((0,), dtype=np.float32)
    sample = feats[0]
    feat_shape = sample.shape[:-1]  # everything except time
    T = max(1, int(round(seg.duration * frequency)))
    out_shape = feat_shape + (T,)
    out = np.zeros(out_shape, dtype=np.float32)
    counts: np.ndarray | None = None
    if aggregation == "mean":
        counts = np.zeros(T, dtype=np.int64)

    seg_stop = seg.start + seg.duration
    for ev, feat in zip(seg_events, feats):
        ev_T = feat.shape[-1]
        ov_start = max(ev.start, seg.start)
        ov_stop = min(ev.stop, seg_stop)
        if ov_stop <= ov_start:
            continue
        # Source slice (in event-local time samples)
        src_lo = int(round((ov_start - ev.start) * frequency))
        src_hi = int(round((ov_stop - ev.start) * frequency))
        src_lo = max(0, min(ev_T, src_lo))
        src_hi = max(src_lo, min(ev_T, src_hi))
        if src_hi == src_lo:
            continue
        # Destination slice (in segment-local time samples)
        dst_lo = int(round((ov_start - seg.start) * frequency))
        n = src_hi - src_lo
        dst_hi = dst_lo + n
        # Clip to segment bounds (truncate if event extends past T)
        if dst_hi > T:
            cut = dst_hi - T
            dst_hi = T
            src_hi -= cut
            n = src_hi - src_lo
            if n <= 0:
                continue
        if dst_lo < 0:
            cut = -dst_lo
            dst_lo = 0
            src_lo += cut
            n = src_hi - src_lo
            if n <= 0:
                continue
        feat32 = feat[..., src_lo:src_hi].astype(np.float32, copy=False)
        if aggregation == "mean":
            assert counts is not None
            c = counts[dst_lo:dst_hi]
            upd = c / (1.0 + c)
            sl = out[..., dst_lo:dst_hi]
            sl *= upd
            sl += (1.0 - upd) * feat32
            counts[dst_lo:dst_hi] = c + 1
        else:
            out[..., dst_lo:dst_hi] += feat32
    return out


# --------------------------------------------------------------------------- #
# Layer aggregation (matches BaseExtractor._aggregate_layers)
# --------------------------------------------------------------------------- #


def select_layer_indices(layers: Iterable[float], n_model_layers: int) -> list[int]:
    """``int(ratio * (n_model_layers - 1))`` for each ratio, deduplicated."""
    return np.unique(
        [int(r * (n_model_layers - 1)) for r in layers]
    ).tolist()


def aggregate_layers(
    latents: torch.Tensor, layer_indices: list[int], aggregation: str | None
) -> torch.Tensor:
    """Apply layer aggregation on the leading layer dim of ``latents``.

    Input shape: ``(n_model_layers, *)``.  Output shape depends on
    ``aggregation`` - matches ``BaseExtractor._aggregate_layers`` from
    neuralset.
    """
    if len(layer_indices) == 1:
        i = layer_indices[0]
        if aggregation is None:
            return latents[i : i + 1]
        return latents[i]
    if aggregation == "mean":
        return latents[layer_indices].mean(dim=0)
    if aggregation == "sum":
        return latents[layer_indices].sum(dim=0)
    if aggregation is None:
        return latents[layer_indices]
    if aggregation == "group_mean":
        idx = list(layer_indices)
        idx[-1] = idx[-1] + 1
        groups = [latents[a:b].mean(dim=0) for a, b in zip(idx[:-1], idx[1:])]
        return torch.stack(groups, dim=0)
    raise ValueError(f"Unknown layer aggregation: {aggregation}")


def aggregate_tokens(latents: torch.Tensor, mode: str | None) -> torch.Tensor:
    """Aggregate the second axis (tokens) of a ``(L, T, D)`` (or ``(L, T, ...)``) tensor.

    Matches ``BaseExtractor._aggregate_tokens`` in neuralset.
    """
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
