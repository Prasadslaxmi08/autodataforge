"""Builtin classical-vision adapter (Phase 1 default).

A real, dependency-light detector + segmenter that runs on CPU with no weight
downloads, so the full pipeline is runnable and testable everywhere (CI, an
8 GB laptop, air-gapped). It finds foreground blobs by intensity thresholding +
connected components, and segments each detected box by thresholding within it.

ponytail: this is the runnable-everywhere backend, not a foundation model. The
production path is a swap of one config line to a real adapter (e.g.
`vds.models.adapters.yolo:YoloAdapter`) — the plugin system already supports it.
Ceiling: union-find components is O(pixels); fine for MVP-scale images.
"""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
from PIL import Image

from vds.core.contracts import Box2D, Detection, Mask
from vds.models.protocols import Capability

_MIN_AREA_FRAC = 0.0015  # ignore blobs smaller than this fraction of the image
# ponytail: on photographic images the whole scene differs from the border-median
# background and fuses into one frame-spanning component whose box is (0,0,W,H) —
# a degenerate "detection", not an object. Mirror of the min-area guard above.
# Raise toward 1.0 if a dataset has legitimate near-full-frame subjects.
_MAX_AREA_FRAC = 0.95  # drop blobs whose BOX covers ~the whole frame


def _load_gray(image: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image)).convert("L")
    return np.asarray(img, dtype=np.uint8)


def _foreground(gray: np.ndarray) -> np.ndarray:
    """Binary mask of pixels that differ from the (border-estimated) background."""
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    bg = float(np.median(border))
    spread = float(gray.std()) or 1.0
    return np.abs(gray.astype(np.int16) - bg) > max(20.0, 0.5 * spread)


def _connected_components(mask: np.ndarray) -> np.ndarray:
    """Label 4-connected components with union-find. 0 = background."""
    h, w = mask.shape
    parent = np.arange(h * w, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    ys, xs = np.nonzero(mask)
    fg = set(zip(ys.tolist(), xs.tolist(), strict=True))
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        idx = y * w + x
        if (y - 1, x) in fg:
            union(idx, (y - 1) * w + x)
        if (y, x - 1) in fg:
            union(idx, y * w + (x - 1))

    labels = np.zeros((h, w), dtype=np.int32)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        labels[y, x] = find(y * w + x) + 1  # +1 so background stays 0
    return labels


def _rle_encode(binary: np.ndarray) -> str:
    """COCO-style uncompressed RLE (column-major), serialized as JSON counts."""
    flat = binary.flatten(order="F").astype(np.uint8)
    counts: list[int] = []
    prev, run = 0, 0  # COCO RLE starts counting zeros
    for v in flat:
        if v == prev:
            run += 1
        else:
            counts.append(run)
            prev, run = v, 1
    counts.append(run)
    return json.dumps(counts)


class BuiltinAdapter:
    name = "builtin"
    capabilities = frozenset({Capability.DETECTOR, Capability.SEGMENTER})
    vram_estimate_mb = 0

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    # --- Detector ---
    def detect(
        self, images: list[bytes], prompts: list[str], params: dict[str, Any]
    ) -> list[list[Detection]]:
        label = prompts[0] if prompts else "object"
        results: list[list[Detection]] = []
        for data in images:
            gray = _load_gray(data)
            h, w = gray.shape
            min_area = _MIN_AREA_FRAC * h * w
            labels = _connected_components(_foreground(gray))
            dets: list[Detection] = []
            for lbl in np.unique(labels):
                if lbl == 0:
                    continue
                ys, xs = np.nonzero(labels == lbl)
                area = len(xs)
                if area < min_area:
                    continue
                x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                box_area = (x1 - x0 + 1) * (y1 - y0 + 1)
                if box_area > _MAX_AREA_FRAC * h * w:
                    continue  # degenerate whole-frame blob (background merged in), not an object
                # Confidence: how solidly the blob fills its bounding box.
                fill = area / (box_area or 1)
                dets.append(
                    Detection(
                        box=Box2D(x=x0, y=y0, w=x1 - x0 + 1, h=y1 - y0 + 1),
                        label=label,
                        confidence=round(min(1.0, 0.5 + 0.5 * fill), 4),
                    )
                )
            dets.sort(key=lambda d: (d.box.y, d.box.x))
            results.append(dets)
        return results

    # --- Segmenter ---
    def segment(self, image: bytes, prompts: list[Box2D | tuple[float, float]]) -> Mask:
        gray = _load_gray(image)
        h, w = gray.shape
        fg = _foreground(gray)
        mask = np.zeros((h, w), dtype=np.uint8)
        for p in prompts:
            if isinstance(p, Box2D):
                x0, y0 = int(p.x), int(p.y)
                x1, y1 = int(p.x + p.w), int(p.y + p.h)
                mask[y0:y1, x0:x1] = fg[y0:y1, x0:x1]
        return Mask(rle=_rle_encode(mask), height=h, width=w)
