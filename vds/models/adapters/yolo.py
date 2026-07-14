"""Production Ultralytics YOLO adapter (Phase 18.5) — the real detector.

Replaces the classical BuiltinAdapter as the default. One forward pass produces
boxes (and, for `-seg` models, instance masks) with **confidence from the model** —
never fabricated. It implements BOTH the Detector and Segmenter capabilities so the
existing pipeline seam (labeler: `detect(...)` then per-box `segment(...)`) is unchanged
and never re-runs inference: `detect()` caches each image's result and `segment()`
reads that cache.

Supports YOLOv8 / YOLO11 detection & segmentation `.pt` weights (pose-compatible).
`.onnx` / TensorRT are accepted as a **future interface** — `YOLO()` will load an
`.onnx`, but they are not part of this phase's validated path.

Model, confidence, IoU, image size, and segmentation come from the GUI-settable
runtime config (`yolo_config`); the plugin *selection* stays config-driven (vds.toml).

Swap back to the classical detector for offline tests with:
    VDS_MODELS__DETECTOR=vds.models.adapters.builtin:BuiltinAdapter
"""

from __future__ import annotations

import hashlib
import io
from collections import OrderedDict
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from vds.core.contracts import Box2D, Detection, Mask
from vds.core.errors import ConfigError
from vds.models.adapters import yolo_config
from vds.models.adapters.builtin import _rle_encode  # shared RLE the editor decodes
from vds.models.protocols import Capability

_CACHE_MAX = 16  # last-N images' results, so segment() reuses detect()'s inference


def _iou_xyxy(a: tuple[float, float, float, float], b: Box2D) -> float:
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a[0], b.x), max(a[1], b.y)
    ix2, iy2 = min(a[2], bx2), min(a[3], by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


class YoloAdapter:
    name = "yolo"
    capabilities = frozenset({Capability.DETECTOR, Capability.SEGMENTER})
    vram_estimate_mb = 1024  # YOLO11n/s/m fit well inside the 8 GB budget

    def __init__(self) -> None:
        self._model = None
        self._weights: str | None = None
        self._device: Any = "cpu"
        self._cache: OrderedDict[str, Any] = OrderedDict()  # sha1 -> ultralytics Result

    # --- lifecycle ---
    def load(self) -> None:
        self._ensure_model(yolo_config.get_config().model)

    def unload(self) -> None:
        self._model = None
        self._weights = None
        self._cache.clear()

    def _ensure_model(self, weights: str) -> None:
        """(Re)load the model if the requested weights changed."""
        if self._model is not None and weights == self._weights:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ConfigError(
                "YoloAdapter requires 'ultralytics'. Install it with "
                "`pip install ultralytics`, or set VDS_MODELS__DETECTOR to the builtin "
                "adapter for the classical CPU backend."
            ) from exc
        try:
            import torch
            self._device = 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            self._device = "cpu"
        self._model = YOLO(weights)
        self._weights = weights
        self._cache.clear()

    # --- inference ---
    def _predict(self, images: list[bytes]) -> list[Any]:
        cfg = yolo_config.get_config()
        self._ensure_model(cfg.model)
        pil = [Image.open(io.BytesIO(d)).convert("RGB") for d in images]
        results = self._model.predict(
            pil, conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz,
            device=self._device, verbose=False,
        )
        for data, res in zip(images, results, strict=True):
            key = hashlib.sha1(data).hexdigest()
            self._cache[key] = res
            self._cache.move_to_end(key)
            while len(self._cache) > _CACHE_MAX:
                self._cache.popitem(last=False)
        return results

    def detect(
        self, images: list[bytes], prompts: list[str], params: dict[str, Any]
    ) -> list[list[Detection]]:
        results = self._predict(images)
        out: list[list[Detection]] = []
        for res in results:
            dets: list[Detection] = []
            boxes = getattr(res, "boxes", None)
            if boxes is not None:
                names = res.names
                for xyxy, conf, cls in zip(
                    boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(),
                    strict=True,
                ):
                    x0, y0, x1, y1 = xyxy
                    dets.append(Detection(
                        box=Box2D(x=x0, y=y0, w=x1 - x0, h=y1 - y0),
                        label=str(names[int(cls)]),
                        confidence=round(float(conf), 4),  # from the model, not fabricated
                    ))
            out.append(dets)
        return out

    def segment(self, image: bytes, prompts: list[Box2D | tuple[float, float]]) -> Mask | None:
        # Returns None for a detection-only model (or no matching instance): the
        # annotation then carries no mask and the verifier judges it by confidence.
        # A non-None *empty* mask would be rejected by RuleBasedVerifier's empty-mask
        # rule, silently dropping every detect-only annotation.
        res = self._cache.get(hashlib.sha1(image).hexdigest())
        masks = getattr(res, "masks", None) if res is not None else None
        target = next((p for p in prompts if isinstance(p, Box2D)), None)
        if masks is None or target is None or getattr(res, "boxes", None) is None:
            return None
        # Pick the instance whose detection box best matches the requested box.
        best_i, best_iou = -1, 0.0
        for i, xy in enumerate(res.boxes.xyxy.tolist()):
            iou = _iou_xyxy(tuple(xy), target)
            if iou > best_iou:
                best_i, best_iou = i, iou
        if best_i < 0 or best_i >= len(masks.xy):
            return None
        W, H = Image.open(io.BytesIO(image)).size
        canvas = Image.new("L", (W, H), 0)
        poly = masks.xy[best_i]
        if len(poly) >= 3:
            ImageDraw.Draw(canvas).polygon([(float(x), float(y)) for x, y in poly], fill=1)
        return Mask(rle=_rle_encode(np.asarray(canvas, dtype=np.uint8)), width=W, height=H)
