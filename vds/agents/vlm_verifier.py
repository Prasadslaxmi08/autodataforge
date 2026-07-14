"""VLM Verifier (Phase 9) — multimodal semantic verification.

Replaces the heuristic RuleBasedVerifier with a Vision-Language-Model judge that
looks at the actual pixels. It is NOT an object detector and it does NOT create
annotations: it decides whether the annotations the pipeline already produced are
*semantically* correct, grounding every judgement in measurable evidence from the
detector + segmentation outputs (a deterministic evidence pack) combined with
multimodal scene understanding (cropped region + full scene).

Same safety contract as the Planner and Analyst (phase brief): the VLM never
fails the caller. On invalid output, provider failure, missing credentials, or a
text-only provider (Echo can't see images), it falls back to the deterministic
RuleBasedVerifier and logs why.

Anti-hallucination, same as the Analyst: a recommendation that cites no real
evidence [key] is dropped before it can reach a human.
"""

from __future__ import annotations

import base64
import io
import time
from typing import Literal

from pydantic import BaseModel, Field

from vds.agents.base import Agent
from vds.agents.cost import estimate_cost
from vds.agents.llm import LLMClient
from vds.agents.messages import ImageContent
from vds.agents.verifier import APPROVED, NEEDS_REVIEW, REJECTED, RuleBasedVerifier
from vds.core.contracts import Annotation, Box2D, Verdict
from vds.core.geometry import mask_is_empty, overlap_iou
from vds.logging import get_logger

log = get_logger(__name__)

# Decisions map onto the pipeline's existing state machine via these core
# VerdictLabels (see verifier.APPROVED/NEEDS_REVIEW/REJECTED). Reusing them keeps
# Phase1Pipeline._apply_verdict unchanged.
_DECISION_TO_VERDICT = {
    "approved": APPROVED,
    "needs_human_review": NEEDS_REVIEW,
    "rejected": REJECTED,
}

_DUP_IOU = 0.5  # boxes overlapping more than this are flagged as duplicate candidates
_SMALL_AREA_FRAC = 0.005  # <0.5% of the image = a "small object" (error-prone)


# --- structured VLM output -------------------------------------------------
class Correction(BaseModel):
    """A recommended correction. Every field the phase brief requires."""

    issue: str
    suggested_correction: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    expected_impact: str
    cited_evidence: list[str] = Field(default_factory=list)


class AnnotationVerdict(BaseModel):
    annotation_index: int
    decision: Literal["approved", "needs_human_review", "rejected"]
    object_identity_correct: bool
    label_correct: bool
    bbox_quality: Literal["good", "loose", "tight", "misaligned", "invalid"]
    segmentation_quality: Literal["good", "partial", "empty", "not_applicable"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    cited_evidence: list[str] = Field(default_factory=list)
    corrections: list[Correction] = Field(default_factory=list)


class SceneFinding(BaseModel):
    """A scene-level issue that isn't tied to one existing annotation:
    a missed object, or a duplicate of an existing one."""

    kind: Literal["missing_annotation", "duplicate_annotation"]
    description: str
    annotation_indices: list[int] = Field(default_factory=list)  # for duplicates
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_correction: str
    expected_impact: str
    cited_evidence: list[str] = Field(default_factory=list)


class VLMSceneVerdict(BaseModel):
    annotation_verdicts: list[AnnotationVerdict] = Field(default_factory=list)
    scene_findings: list[SceneFinding] = Field(default_factory=list)
    summary: str = ""


# --- result wrapper (verdicts + provenance + metrics) ----------------------
class SceneVerdictResult(BaseModel):
    verdicts: list[Verdict]  # one core Verdict per input annotation (pipeline-ready)
    detail: VLMSceneVerdict | None = None
    scene_findings: list[SceneFinding] = Field(default_factory=list)
    source: Literal["vlm", "deterministic"]
    evidence_coverage: float = 1.0  # fraction of recs whose citations are all valid
    dropped_recommendations: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None
    latency_ms: float = 0.0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float | None = None


VLM_SYSTEM_PROMPT = """\
You are a vision quality-assurance reviewer for an automated annotation pipeline.
You are NOT an object detector and you do NOT create annotations. You judge
whether the EXISTING annotations are semantically correct, using three inputs:
  1. the full scene image,
  2. a cropped region per annotation,
  3. an EVIDENCE PACK of measured facts, each tagged with a [key].

For every annotation decide one of: approved, needs_human_review, rejected. Also
assess object identity, label correctness, bounding-box quality, and segmentation
quality. Report scene-level findings for objects that are clearly present but have
NO annotation (missing_annotation) and for annotations that duplicate another
(duplicate_annotation).

Hard rules:
- Ground every judgement in what you see PLUS the evidence pack. Never invent a
  measurement. Each correction / finding's `cited_evidence` MUST list only [key]
  tokens that appear in the evidence pack.
- Every correction and finding needs: issue/description, suggested_correction,
  reasoning, confidence (0-1), and expected_impact.
- Be conservative: when the pixels are ambiguous, choose needs_human_review, not
  approved. Output ONLY a JSON object matching the required schema.
"""


# --- deterministic evidence pack -------------------------------------------
class _Fact(BaseModel):
    key: str
    statement: str


def build_evidence(
    annotations: list[Annotation], width: int, height: int
) -> list[_Fact]:
    """Every fact computed deterministically from detector + segmentation output.
    No model, no pixels — this is the measurable half the VLM must cite."""
    facts: list[_Fact] = []
    area_img = float(width * height) or 1.0

    def f(key: str, stmt: str) -> None:
        facts.append(_Fact(key=key, statement=stmt))

    f("scene_dims", f"Image is {width}x{height} px.")
    f("scene_count", f"{len(annotations)} existing annotations: "
      + ", ".join(f"#{i}={a.label!r}" for i, a in enumerate(annotations)))

    for i, a in enumerate(annotations):
        f(f"a{i}_label", f"#{i} proposed label is {a.label!r}.")
        f(f"a{i}_conf", f"#{i} detector confidence is {a.confidence:.2f}.")
        g = a.geometry
        if isinstance(g, Box2D):
            valid = g.w > 0 and g.h > 0 and g.x >= 0 and g.y >= 0
            frac = (g.w * g.h) / area_img
            aspect = round(g.w / g.h, 2) if g.h > 0 else 0.0
            f(f"a{i}_box", f"#{i} box x={g.x:.0f} y={g.y:.0f} w={g.w:.0f} h={g.h:.0f} "
              f"({frac:.2%} of image, aspect {aspect}).")
            if not valid:
                f(f"a{i}_boxvalid", f"#{i} box geometry is INVALID (non-positive or off-image).")
            if valid and frac < _SMALL_AREA_FRAC:
                f(f"a{i}_small", f"#{i} is a small object (<{_SMALL_AREA_FRAC:.1%} of image); error-prone.")
        if a.mask is not None:
            f(f"a{i}_mask", f"#{i} has a segmentation mask ("
              + ("EMPTY — no foreground" if mask_is_empty(a.mask.rle) else "non-empty") + ").")
        else:
            f(f"a{i}_mask", f"#{i} has no segmentation mask.")

    # Duplicate candidates (box IoU) — scene-level measurable evidence.
    for i in range(len(annotations)):
        gi = annotations[i].geometry
        if not isinstance(gi, Box2D):
            continue
        for j in range(i + 1, len(annotations)):
            gj = annotations[j].geometry
            if isinstance(gj, Box2D) and overlap_iou(gi, gj) > _DUP_IOU:
                f(f"dup_{i}_{j}", f"#{i} and #{j} overlap heavily "
                  f"(IoU {overlap_iou(gi, gj):.2f}); possible duplicate.")
    return facts


def render_evidence(facts: list[_Fact]) -> str:
    return "\n".join(f"[{f.key}] {f.statement}" for f in facts)


# --- image encoding (crops + full scene) -----------------------------------
def _png_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _thumb(img, max_dim: int):
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return img


def build_images(
    image: bytes, annotations: list[Annotation], *, scene_max=512, crop_max=256
) -> tuple[list[ImageContent], int, int]:
    """Full scene + one crop per boxed annotation, as base64 PNGs. Returns the
    images plus the true (width, height) needed by the evidence pack."""
    from PIL import Image

    img = Image.open(io.BytesIO(image)).convert("RGB")
    width, height = img.size
    out = [ImageContent(data_base64=_png_b64(_thumb(img, scene_max)))]
    for a in annotations:
        g = a.geometry
        if not isinstance(g, Box2D):
            continue
        x1, y1 = max(0, int(g.x)), max(0, int(g.y))
        x2, y2 = min(width, int(g.x + g.w)), min(height, int(g.y + g.h))
        if x2 > x1 and y2 > y1:
            out.append(ImageContent(data_base64=_png_b64(_thumb(img.crop((x1, y1, x2, y2)), crop_max))))
    return out, width, height


# --- the agent -------------------------------------------------------------
class LLMVerifier(Agent):
    """Multimodal semantic verifier. Scene-level so it can spot missing and
    duplicate annotations; also satisfies the single-annotation VerifierAgent
    protocol for drop-in use in the existing pipeline."""

    system_prompt = VLM_SYSTEM_PROMPT

    def __init__(self, client: LLMClient, fallback: RuleBasedVerifier | None = None) -> None:
        super().__init__(client)
        self._fallback = fallback or RuleBasedVerifier()

    # VerifierAgent protocol: judge one annotation (pipeline drop-in).
    def verify(self, image: bytes, annotation: Annotation) -> Verdict:
        return self.verify_scene(image, [annotation]).verdicts[0]

    def verify_scene(self, image: bytes, annotations: list[Annotation]) -> SceneVerdictResult:
        start = time.perf_counter()
        if not annotations:
            return SceneVerdictResult(verdicts=[], source="vlm",
                                      latency_ms=self._ms(start))
        try:
            images, width, height = build_images(image, annotations)
        except Exception as exc:  # bad/undecodable image bytes -> deterministic
            return self._fallback_result(image, annotations, f"image decode: {exc}", start)

        facts = build_evidence(annotations, width, height)
        keys = {f.key for f in facts}
        convo = self.new_conversation().user(
            "Review these existing annotations against the images and evidence. "
            "Cite only the [key] tokens below. Return one VLMSceneVerdict JSON "
            "object with a verdict for EVERY annotation index.\n\nEVIDENCE PACK:\n"
            + render_evidence(facts),
            images=images,
        )
        try:
            outcome = self._client.structured(convo, VLMSceneVerdict)
        except Exception as exc:  # provider/parse/validation failure -> deterministic
            return self._fallback_result(image, annotations, f"{type(exc).__name__}: {exc}", start)

        detail, coverage, dropped = self._enforce_evidence(outcome.value, keys)
        verdicts = self._to_verdicts(detail, annotations)
        usage = outcome.response.usage
        log.info("vlm_verifier.scene", annotations=len(annotations),
                 coverage=coverage, latency_ms=self._ms(start))
        return SceneVerdictResult(
            verdicts=verdicts, detail=detail, scene_findings=detail.scene_findings,
            source="vlm", evidence_coverage=coverage, dropped_recommendations=dropped,
            latency_ms=self._ms(start), retries=outcome.attempts - 1,
            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
            estimated_cost_usd=estimate_cost(outcome.response.model, usage),
        )

    # --- helpers ---
    def _to_verdicts(self, detail: VLMSceneVerdict, annotations: list[Annotation]) -> list[Verdict]:
        by_idx = {av.annotation_index: av for av in detail.annotation_verdicts}
        verdicts: list[Verdict] = []
        for i, ann in enumerate(annotations):
            av = by_idx.get(i)
            if av is None:  # VLM skipped one -> a human must look, never silently pass
                verdicts.append(Verdict(annotation_id=ann.id, verdict=NEEDS_REVIEW,
                                        confidence=0.0, rationale="VLM returned no verdict for this annotation"))
                continue
            verdicts.append(Verdict(
                annotation_id=ann.id,
                # Strict 3-way routing the pipeline understands: correct ->
                # auto-accept, hallucination -> auto-reject, bad_geometry ->
                # human review. The identity/label/geometry nuance lives in
                # `detail`, not the routing label.
                verdict=_DECISION_TO_VERDICT[av.decision],
                confidence=av.confidence,
                rationale=av.reasoning,
            ))
        return verdicts

    def _enforce_evidence(self, detail: VLMSceneVerdict, keys: set[str]):
        """Drop corrections/findings that cite no real evidence key (anti-hallucination).
        Returns (cleaned_detail, coverage, dropped_descriptions)."""
        dropped: list[str] = []
        total = 0

        def clean(items, cite_attr: str, label_attr: str):
            nonlocal total
            kept = []
            for it in items:
                total += 1
                valid = [k for k in getattr(it, cite_attr) if k in keys]
                if valid:
                    setattr(it, cite_attr, valid)
                    kept.append(it)
                else:
                    dropped.append(getattr(it, label_attr))
            return kept

        for av in detail.annotation_verdicts:
            av.corrections = clean(av.corrections, "cited_evidence", "issue")
        detail.scene_findings = clean(detail.scene_findings, "cited_evidence", "description")
        coverage = round((total - len(dropped)) / total, 4) if total else 1.0
        return detail, coverage, dropped

    def _fallback_result(self, image, annotations, reason: str, start: float) -> SceneVerdictResult:
        log.warning("vlm_verifier.fallback", reason=reason, annotations=len(annotations))
        verdicts = [self._fallback.verify(image, a) for a in annotations]
        return SceneVerdictResult(
            verdicts=verdicts, source="deterministic",
            fallback_reason=reason, latency_ms=self._ms(start),
        )

    @staticmethod
    def _ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 3)
