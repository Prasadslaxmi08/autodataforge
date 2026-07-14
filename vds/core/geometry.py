"""Pure geometry/mask helpers (L0).

Shared by the verifier, the labeling engine, and the quality analyzer. Living in
`core` keeps the dependency graph pointing downward — services and agents import
these, not each other.
"""

from __future__ import annotations

import json

from vds.core.contracts import Box2D


def overlap_iou(a: Box2D, b: Box2D) -> float:
    """Intersection-over-union of two axis-aligned boxes."""
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def mask_is_empty(rle: str) -> bool:
    """A COCO RLE (JSON counts, leading zero-run) is empty if it has no
    foreground runs — i.e. at most one count."""
    try:
        counts = json.loads(rle)
    except (json.JSONDecodeError, TypeError):
        return True
    return len(counts) <= 1
