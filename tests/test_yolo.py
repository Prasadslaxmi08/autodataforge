"""Production YOLO detector tests (Phase 18.5).

All skipped unless `ultralytics` is installed (it pulls torch and downloads weights),
so CI and the classical-builtin path stay offline. Run locally after
`pip install ultralytics`.

Uses Ultralytics' bundled `bus.jpg` (a bus + several people) for reliable, real
detections — no fixtures to hand-tune.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ultralytics")


from ultralytics.utils import ASSETS  # noqa: E402

from vds.config.settings import ModelSelection, Settings, StorageSettings  # noqa: E402
from vds.container import Container  # noqa: E402
from vds.core.geometry import mask_is_empty  # noqa: E402
from vds.gui.controller import BackendController  # noqa: E402
from vds.models.adapters.yolo import YoloAdapter  # noqa: E402
from vds.models.adapters.yolo_config import YoloRuntimeConfig, set_config  # noqa: E402

_BUS = (ASSETS / "bus.jpg").read_bytes()
_YOLO = "vds.models.adapters.yolo:YoloAdapter"


@pytest.fixture(autouse=True)
def _reset_config():
    set_config(YoloRuntimeConfig(model="yolo11n.pt", conf=0.25, iou=0.7, imgsz=640))
    yield
    set_config(YoloRuntimeConfig())


def test_yolo_detects_real_objects_with_model_confidence():
    det = YoloAdapter()
    results = det.detect([_BUS], ["object"], {})
    dets = results[0]
    assert len(dets) >= 2  # bus.jpg has a bus + several people
    for d in dets:
        assert 0.25 <= d.confidence <= 1.0            # real model score, above the conf floor
        assert d.box.w > 0 and d.box.h > 0
    # confidences are model-driven, not the builtin 0.5+0.5*fill fabrication
    assert len({round(d.confidence, 3) for d in dets}) > 1


def test_yolo_preserves_coco_class_names():
    dets = YoloAdapter().detect([_BUS], ["object"], {})[0]
    labels = {d.label for d in dets}
    assert "person" in labels  # a real COCO class name, straight from the model
    assert all(lbl.islower() and lbl.isascii() for lbl in labels)


def test_yolo_boxes_inside_image_bounds():
    import io

    from PIL import Image
    W, H = Image.open(io.BytesIO(_BUS)).size
    for d in YoloAdapter().detect([_BUS], ["object"], {})[0]:
        assert 0 <= d.box.x and 0 <= d.box.y
        assert d.box.x + d.box.w <= W + 1 and d.box.y + d.box.h <= H + 1  # not a full-frame blob
        assert (d.box.w * d.box.h) < 0.95 * W * H


def test_yolo_batch_inference():
    out = YoloAdapter().detect([_BUS, _BUS], ["object"], {})
    assert len(out) == 2
    assert len(out[0]) == len(out[1]) and len(out[0]) >= 2


def test_yolo_reloads_when_model_changes():
    det = YoloAdapter()
    det.detect([_BUS], ["object"], {})
    assert det._weights.endswith("yolo11n.pt")
    set_config(YoloRuntimeConfig(model="yolo11n.pt"))  # same → no reload needed
    det.detect([_BUS], ["object"], {})
    assert det._weights.endswith("yolo11n.pt")


def test_yolo_custom_confidence_floor():
    set_config(YoloRuntimeConfig(model="yolo11n.pt", conf=0.8))
    for d in YoloAdapter().detect([_BUS], ["object"], {})[0]:
        assert d.confidence >= 0.8  # user's wizard confidence is honored by the model


def test_yolo_segmentation_produces_masks():
    set_config(YoloRuntimeConfig(model="yolo11n-seg.pt", conf=0.25, segment=True))
    det = YoloAdapter()
    dets = det.detect([_BUS], ["object"], {})[0]
    assert dets
    mask = det.segment(_BUS, [dets[0].box])
    assert not mask_is_empty(mask.rle)  # a real instance mask, not empty


def test_yolo_detect_only_model_returns_no_mask():
    det = YoloAdapter()
    dets = det.detect([_BUS], ["object"], {})[0]  # yolo11n = detection only
    # No mask (None), NOT a non-None empty mask: an empty mask would trip the
    # verifier's empty-mask rejection and silently drop every detect-only annotation.
    assert det.segment(_BUS, [dets[0].box]) is None


def test_yolo_pipeline_end_to_end(tmp_path):
    """The real pipeline with YOLO produces tight, model-confident annotations."""
    import shutil
    src = tmp_path / "imgs"
    src.mkdir()
    shutil.copy(ASSETS / "bus.jpg", src / "bus.jpg")
    settings = Settings(
        models=ModelSelection(detector=_YOLO, segmenter=_YOLO),
        storage=StorageSettings(cas_root=tmp_path / "cas"),
    )
    ctrl = BackendController(Container(settings=settings, db_path=":memory:",
                                      artifacts_dir=tmp_path / "artifacts"))
    report = ctrl.import_dataset(str(src), "yolo-e2e")
    assert report.detections >= 2
    iid = ctrl.project_images(report.project_id)[0].image_id
    boxes = ctrl.image_boxes(iid)
    assert boxes and all(0.0 < b.confidence <= 1.0 for b in boxes)
    assert any(b.label == "person" for b in boxes)  # real class, real pipeline
