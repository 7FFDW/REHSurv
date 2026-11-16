"""Exact per-slide spatial kNN without a quadratic patch-distance matrix."""
from __future__ import annotations

import numpy as np


def build_spatial_index(coords, slide_ids, k):
    coords = np.asarray(coords)
    slide_ids = np.asarray(slide_ids)
    if coords.shape != (len(slide_ids), 2) or not np.isfinite(coords).all():
        raise ValueError("Invalid spatial coordinates")
    if k < 0:
        raise ValueError("spatial_k must be nonnegative")
    neighbors = np.full((len(coords), k), -1, dtype=np.int64)
    if not k or not len(coords):
        return neighbors
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("Spatial kNN requires scipy; install requirements.txt") from exc
    for slide in np.unique(slide_ids):
        if slide < 0:
            continue
        rows = np.flatnonzero(slide_ids == slide)
        count = min(k, len(rows))
        # Include self, matching the original neighborhood definition. Exact query;
        # equal-distance ties may choose any tied patch, always within this slide.
        _, local = cKDTree(coords[rows]).query(coords[rows], k=count, eps=0, workers=1)
        neighbors[rows, :count] = rows[np.asarray(local).reshape(len(rows), count)]
    return neighbors
