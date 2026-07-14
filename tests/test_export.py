"""Unit tests for the COCO/YOLO exporters and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vds.container import Container
from vds.core.contracts import Annotation, Box2D, ImageRecord, Provenance
from vds.core.enums import AnnotationState
from vds.core.errors import ExportError


def _seed(container: Container) -> None:
    """One image with one exportable annotation."""
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (32, 32), (255, 255, 255)).save(buf, format="PNG")
    sha = container.cas.put(buf.getvalue())
    rec = ImageRecord(id="img1", project_id="p1", sha256=sha, width=32, height=32, state="labeled")
    container.images.add(rec)
    container.annotations.add(
        Annotation(
            id="ann1", image_id="img1", label="object",
            geometry=Box2D(x=4, y=4, w=10, h=10), confidence=0.9,
            state=AnnotationState.AUTO_ACCEPTED, provenance=Provenance(source="t"),
        )
    )


def test_export_coco(container: Container, tmp_path: Path):
    _seed(container)
    report = container.exporter.run("p1", "coco", str(tmp_path / "out"))
    assert report.validated and report.format == "coco"
    assert report.images == 1 and report.annotations == 1
    coco = json.loads((tmp_path / "out" / "annotations.json").read_text())
    assert len(coco["annotations"]) == 1
    assert coco["categories"][0]["name"] == "object"


def test_export_yolo(container: Container, tmp_path: Path):
    _seed(container)
    report = container.exporter.run("p1", "yolo", str(tmp_path / "out"))
    assert report.validated and report.format == "yolo"
    label = (tmp_path / "out" / "labels" / "img1.txt").read_text().strip()
    parts = label.split()
    assert len(parts) == 5 and all(0 <= float(v) <= 1 for v in parts[1:])


def test_export_rejects_unknown_format(container: Container, tmp_path: Path):
    _seed(container)
    with pytest.raises(ExportError):
        container.exporter.run("p1", "pascal", str(tmp_path / "out"))
