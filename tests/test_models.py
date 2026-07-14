"""Unit tests for the builtin detector and segmenter."""

from __future__ import annotations

import io
import json

from PIL import Image, ImageDraw

from vds.core.contracts import Box2D
from vds.models.adapters.builtin import BuiltinAdapter


def _img(boxes, size=(64, 64)) -> bytes:
    im = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(im)
    for b in boxes:
        d.rectangle(b, fill=(0, 0, 0))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def test_detector_finds_objects():
    adapter = BuiltinAdapter()
    data = _img([(6, 6, 18, 18), (40, 40, 54, 54)])
    dets = adapter.detect([data], ["object"], {})[0]
    assert len(dets) == 2
    for d in dets:
        assert 0.0 < d.confidence <= 1.0
        assert d.box.w > 0 and d.box.h > 0
        assert d.label == "object"


def test_detector_deterministic():
    adapter = BuiltinAdapter()
    data = _img([(10, 10, 24, 24)])
    a = adapter.detect([data], ["object"], {})[0]
    b = adapter.detect([data], ["object"], {})[0]
    assert [d.model_dump() for d in a] == [d.model_dump() for d in b]


def test_detector_empty_on_blank():
    adapter = BuiltinAdapter()
    blank = _img([])
    assert adapter.detect([blank], ["object"], {})[0] == []


def test_detector_drops_whole_frame_blob():
    # Regression: on photographic images the scene fused into one frame-spanning
    # component whose box was (0,0,W,H), rendered as a giant box over everything.
    # The max-area guard must drop any detection that covers ~the whole frame.
    adapter = BuiltinAdapter()
    data = _img([(1, 1, 98, 98)], size=(100, 100))  # blob covering ~96% of frame
    dets = adapter.detect([data], ["object"], {})[0]
    assert all((d.box.w * d.box.h) <= 0.95 * 100 * 100 for d in dets)


def test_segmenter_produces_nonempty_mask():
    adapter = BuiltinAdapter()
    data = _img([(10, 10, 30, 30)])
    mask = adapter.segment(data, [Box2D(x=10, y=10, w=20, h=20)])
    assert mask.height == 64 and mask.width == 64
    counts = json.loads(mask.rle)
    assert sum(counts) == 64 * 64  # RLE covers every pixel
    assert len(counts) > 1  # has foreground runs -> not empty
