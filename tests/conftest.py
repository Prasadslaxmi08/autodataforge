"""Shared fixtures: synthetic image datasets the builtin detector can process."""

from __future__ import annotations

import os
from pathlib import Path

# Qt tests run headless in CI — pick the offscreen platform before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# The app now defaults to the real YOLO detector (vds.toml). The whole test suite must
# stay offline and deterministic, so pin the classical builtin backend for every
# Settings() the tests build — env outranks vds.toml, so this covers bare Settings()
# and the container fixture alike, with no per-test edits.
os.environ.setdefault("VDS_MODELS__DETECTOR", "vds.models.adapters.builtin:BuiltinAdapter")
os.environ.setdefault("VDS_MODELS__SEGMENTER", "vds.models.adapters.builtin:BuiltinAdapter")

import pytest
from PIL import Image, ImageDraw

from vds.config.settings import Settings, StorageSettings
from vds.container import Container


def _make_image(path: Path, boxes: list[tuple[int, int, int, int]], size=(64, 64)) -> None:
    img = Image.new("RGB", size, (255, 255, 255))  # white background
    draw = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in boxes:
        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))  # black object
    img.save(path)


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Three images, two well-separated black rectangles each (6 objects total)."""
    d = tmp_path / "imgs"
    d.mkdir()
    _make_image(d / "a.png", [(6, 6, 18, 18), (40, 40, 54, 54)])
    _make_image(d / "b.png", [(8, 8, 20, 20), (44, 30, 56, 50)])
    _make_image(d / "c.png", [(10, 10, 22, 22), (34, 44, 50, 58)])
    return d


@pytest.fixture
def container(tmp_path: Path) -> Container:
    """A container backed by an in-memory DB, tmp CAS, and tmp artifacts — fully
    isolated so tests never touch the repo's benchmarks/ directory."""
    settings = Settings(storage=StorageSettings(cas_root=tmp_path / "cas"))
    return Container(
        settings=settings,
        db_path=":memory:",
        artifacts_dir=tmp_path / "artifacts",
    )
