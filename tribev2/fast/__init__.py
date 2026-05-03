# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pure-PyTorch single-PC inference pipeline for TRIBE v2.

Replaces the ``neuralset``-mediated GPU work with hand-rolled
extractors that have no per-batch CPU/GPU sync points.  See
``pipeline.py`` for the user-facing entry point.
"""

from tribev2.fast.dataset import FastBatch, FastSegmentDataset, fast_collate
from tribev2.fast.extractors import (
    AudioExtractor,
    ExtractorConfig,
    TextExtractor,
    VideoExtractor,
)
from tribev2.fast.pipeline import FastTribePipeline
from tribev2.fast.segments import (
    FastEvent,
    FastSegment,
    aggregate_layers,
    aggregate_tokens,
    events_from_dataframe,
    list_segments,
    select_layer_indices,
    slice_static_for_segment,
    slice_temporal_for_segment,
)

__all__ = [
    "AudioExtractor",
    "ExtractorConfig",
    "FastBatch",
    "FastEvent",
    "FastSegment",
    "FastSegmentDataset",
    "FastTribePipeline",
    "TextExtractor",
    "VideoExtractor",
    "aggregate_layers",
    "aggregate_tokens",
    "events_from_dataframe",
    "fast_collate",
    "list_segments",
    "select_layer_indices",
    "slice_static_for_segment",
    "slice_temporal_for_segment",
]
