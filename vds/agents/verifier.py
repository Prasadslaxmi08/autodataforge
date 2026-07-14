"""Verifier Agent (System Design §2.9) — an independent LLM-judge.

Judges each candidate annotation before any human sees it. A deterministic
geometry pre-filter runs first (engine-side) so the VLM only burns compute on
plausible candidates; stratified sampling is the default. Must be a *different*
model family than the labeler — correlated errors are the main failure mode.

Phase 1 ships the deterministic RuleBasedVerifier below (the prompt requires
Phase-1 verification to be deterministic). The VLM-judge implementation of this
same interface lands in Phase 2.
"""

from __future__ import annotations

from typing import Protocol

from vds.core.contracts import Annotation, Box2D, Verdict
from vds.core.geometry import mask_is_empty, overlap_iou  # re-exported for callers

__all__ = [
    "APPROVED",
    "NEEDS_REVIEW",
    "REJECTED",
    "RuleBasedVerifier",
    "VerifierAgent",
    "mask_is_empty",
    "overlap_iou",
]


class VerifierAgent(Protocol):
    def verify(self, image: bytes, annotation: Annotation) -> Verdict: ...


# Verdict.verdict is a semantic label; the review-routing decision (approve /
# needs-review / reject) maps onto AnnotationState downstream. These constants
# name the three Phase-1 outcomes the prompt asks for.
APPROVED = "correct"
NEEDS_REVIEW = "bad_geometry"  # borderline -> a human should look
REJECTED = "hallucination"


class RuleBasedVerifier:
    """Deterministic Phase-1 verifier: confidence + geometry sanity checks.

    Same input given twice -> same verdict. No model, no randomness.
    """

    def __init__(
        self, approve_threshold: float = 0.50, reject_threshold: float = 0.50
    ) -> None:
        self._approve = approve_threshold
        self._reject = reject_threshold

    def verify(self, image: bytes, annotation: Annotation) -> Verdict:
        geom = annotation.geometry
        conf = annotation.confidence

        # 1. Invalid geometry -> reject (impossible box).
        if isinstance(geom, Box2D):
            if geom.w <= 0 or geom.h <= 0 or geom.x < 0 or geom.y < 0:
                return self._v(annotation, REJECTED, conf, "non-positive or off-image box")

        # 2. Empty mask -> reject.
        if annotation.mask is not None and mask_is_empty(annotation.mask.rle):
            return self._v(annotation, REJECTED, conf, "empty segmentation mask")

        # 3. Confidence bands -> approve / reject / needs-review.
        if conf < self._reject:
            return self._v(annotation, REJECTED, conf, "confidence below reject floor")
        if conf >= self._approve:
            return self._v(annotation, APPROVED, conf, "high confidence, valid geometry")
        return self._v(annotation, NEEDS_REVIEW, conf, "mid confidence, needs a human")

    @staticmethod
    def _v(ann: Annotation, verdict: str, conf: float, why: str) -> Verdict:
        return Verdict(
            annotation_id=ann.id, verdict=verdict, confidence=conf, rationale=why
        )
