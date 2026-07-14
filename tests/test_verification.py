"""Unit tests for the deterministic rule-based verifier."""

from __future__ import annotations

import json

from vds.agents.verifier import (
    APPROVED,
    NEEDS_REVIEW,
    REJECTED,
    RuleBasedVerifier,
    overlap_iou,
)
from vds.core.contracts import Annotation, Box2D, Mask, Provenance


def _ann(conf: float, box: Box2D, mask: Mask | None = None) -> Annotation:
    return Annotation(
        id="a", image_id="i", label="object", geometry=box, confidence=conf,
        state="labeled", provenance=Provenance(source="t"), mask=mask,
    )


def test_high_confidence_approved():
    v = RuleBasedVerifier()
    verdict = v.verify(b"", _ann(0.95, Box2D(x=0, y=0, w=10, h=10)))
    assert verdict.verdict == APPROVED


def test_half_confidence_approved():
    # policy: >= 0.50 approves, < 0.50 rejects (no needs-review band by default)
    v = RuleBasedVerifier()
    assert v.verify(b"", _ann(0.50, Box2D(x=0, y=0, w=10, h=10))).verdict == APPROVED
    assert v.verify(b"", _ann(0.49, Box2D(x=0, y=0, w=10, h=10))).verdict == REJECTED


def test_needs_review_band_with_custom_thresholds():
    v = RuleBasedVerifier(approve_threshold=0.75, reject_threshold=0.30)
    assert v.verify(b"", _ann(0.5, Box2D(x=0, y=0, w=10, h=10))).verdict == NEEDS_REVIEW


def test_low_confidence_rejected():
    v = RuleBasedVerifier()
    assert v.verify(b"", _ann(0.1, Box2D(x=0, y=0, w=10, h=10))).verdict == REJECTED


def test_invalid_geometry_rejected():
    v = RuleBasedVerifier()
    assert v.verify(b"", _ann(0.99, Box2D(x=0, y=0, w=0, h=10))).verdict == REJECTED


def test_empty_mask_rejected():
    v = RuleBasedVerifier()
    empty = Mask(rle=json.dumps([4096]), height=64, width=64)  # single background run
    assert v.verify(b"", _ann(0.99, Box2D(x=0, y=0, w=10, h=10), empty)).verdict == REJECTED


def test_verifier_deterministic():
    v = RuleBasedVerifier()
    ann = _ann(0.5, Box2D(x=1, y=2, w=10, h=10))
    assert v.verify(b"", ann).verdict == v.verify(b"", ann).verdict


def test_overlap_iou():
    a = Box2D(x=0, y=0, w=10, h=10)
    assert overlap_iou(a, a) == 1.0
    assert overlap_iou(a, Box2D(x=100, y=100, w=10, h=10)) == 0.0
