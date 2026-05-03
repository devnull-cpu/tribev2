# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Per-item disk cache for extracted features.

One ``.npy`` file per event keyed by a SHA-256 hash of
``(extractor_uid, event_uid)``.  Atomic writes via temp-file rename so a
crashed run never leaves half-written cache files.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class FeatureCache:
    """Tiny per-event npy cache keyed by ``(extractor_uid, event_uid)``."""

    def __init__(self, root: str | Path, extractor_uid: str):
        self.root = Path(root)
        self.extractor_uid = extractor_uid
        self.dir = self.root / extractor_uid
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, event_uid: str) -> Path:
        digest = hashlib.sha256(
            f"{self.extractor_uid}\0{event_uid}".encode("utf-8")
        ).hexdigest()
        # Two-level fan-out so a directory listing doesn't explode at 100k+ files.
        return self.dir / digest[:2] / f"{digest[2:]}.npy"

    def has(self, event_uid: str) -> bool:
        return self._path(event_uid).is_file()

    def load(self, event_uid: str, *, mmap: bool = True) -> np.ndarray:
        return np.load(
            self._path(event_uid),
            allow_pickle=False,
            mmap_mode="r" if mmap else None,
        )

    def save(self, event_uid: str, array: np.ndarray) -> None:
        path = self._path(event_uid)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp file in the same directory, then rename.
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".tmp.", suffix=".npy", delete=False
        ) as f:
            tmp = Path(f.name)
        try:
            np.save(tmp, array, allow_pickle=False)
            os.replace(tmp, path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise


def stable_uid(*parts: object) -> str:
    """Hash arbitrary stringifiable parts into a short stable id."""
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:32]
