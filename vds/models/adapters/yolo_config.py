"""Runtime knobs for the YOLO adapter (Phase 18.5).

The adapter's *which model / how* choices (weights, confidence, IoU, image size,
segmentation) are per-run and chosen in the GUI — but the pipeline, planner, and
labeler are frozen and pass no such options. This tiny process-global is the seam:
the GUI writes it before a run, the adapter reads it per call.

ponytail: a module-level singleton, not threaded config plumbing — this is a
single-process desktop app and the detector is a shared registry instance. The
plugin selection itself stays config-driven (vds.toml); this only tunes the YOLO
adapter once it is the selected detector. Per-project persistence lives in the GUI
(GuiSettings); this holds only the currently-active run's values.
"""

from __future__ import annotations

from dataclasses import dataclass

# The three shipped presets + "custom" (a user-picked .pt path).
YOLO_PRESETS = ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt")


@dataclass
class YoloRuntimeConfig:
    model: str = "yolo11n.pt"   # a preset name or an absolute .pt path
    conf: float = 0.25          # detection confidence (model-driven, never fabricated)
    iou: float = 0.7            # NMS IoU
    imgsz: int = 640            # max inference image size
    segment: bool = False       # request instance masks (needs a -seg model)


# The single active config the adapter reads. Replaced wholesale by set_config().
_active = YoloRuntimeConfig()


def get_config() -> YoloRuntimeConfig:
    return _active


def set_config(cfg: YoloRuntimeConfig) -> None:
    global _active
    _active = cfg
