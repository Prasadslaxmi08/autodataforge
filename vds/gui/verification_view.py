"""Verification view-model (Phase 14) — plain data for the AI Verification Workspace.

No Qt here. The pipeline stores each annotation's *state* but not the verdict
object, so to explain WHY a decision was made this module reproduces the verdict by
calling the EXISTING `RuleBasedVerifier` (deterministic — same input, same result),
never a new verification implementation. Every evidence score is computed from a
measured backend output (confidence, geometry, mask, neighbour IoU, memory); where
the backend genuinely cannot supply a value (timestamps, original runtime) it is
reported as unavailable, never invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vds.agents.planner_agent import build_dataset_context
from vds.agents.verifier import APPROVED, NEEDS_REVIEW, REJECTED
from vds.container import Container
from vds.core.contracts import Box2D
from vds.core.enums import AnnotationState, is_legal_transition
from vds.core.geometry import mask_is_empty, overlap_iou
from vds.memory import DatasetFingerprint

# AI verification status shown in the results table.
STATUS_VERIFIED = "Verified"
STATUS_REVIEW = "Needs Review"
STATUS_REJECTED = "Rejected"
STATUS_UNCERTAIN = "Uncertain"

STATUS_COLOR = {
    STATUS_VERIFIED: "#4caf82", STATUS_REVIEW: "#e0a458",
    STATUS_REJECTED: "#e0605e", STATUS_UNCERTAIN: "#c98bff",
}

_VERDICT_TO_STATUS = {APPROVED: STATUS_VERIFIED, NEEDS_REVIEW: STATUS_REVIEW,
                      REJECTED: STATUS_REJECTED}

# Human-review actions -> target annotation state (existing state machine).
REVIEW_ACTIONS = {
    "approve": AnnotationState.ACCEPTED, "accept_detection": AnnotationState.ACCEPTED,
    "reject": AnnotationState.REJECTED, "reject_detection": AnnotationState.REJECTED,
    "mark_review": AnnotationState.NEEDS_REVIEW,
}


@dataclass
class ObjectVerdict:
    object_id: str
    image_id: str
    image_name: str
    label: str
    detection_confidence: float
    status: str
    verification_confidence: float
    suggested_action: str
    rationale: str
    state: str  # current stored annotation state (final decision)
    box: tuple[float, float, float, float] | None


@dataclass
class Star:
    label: str
    value: int | None  # 0..5, or None when unavailable
    detail: str


@dataclass
class Evidence:
    summary: str
    reason: str
    evidence: list[str]
    verification_confidence: float
    risk: str
    recommendation: str
    stars: list[Star]


@dataclass
class VerificationStats:
    verified: int
    rejected: int
    needs_review: int
    avg_verification_confidence: float
    avg_detection_confidence: float
    agreement_rate: float
    disagreement_rate: float
    review_percentage: float
    verification_runtime: str  # measured seconds or "unavailable"


@dataclass
class HistoricalComparison:
    influenced: bool
    note: str
    matches: list[dict] = field(default_factory=list)


@dataclass
class TimelineStep:
    stage: str
    status: str
    timestamp: str  # "unavailable" — not persisted by the backend


# --- verdict reproduction --------------------------------------------------
def _classify(verdict_label: str, confidence: float) -> str:
    status = _VERDICT_TO_STATUS.get(verdict_label, STATUS_UNCERTAIN)
    # Borderline confidence near a decision boundary reads as Uncertain.
    if status == STATUS_REVIEW and (abs(confidence - 0.30) < 0.08 or abs(confidence - 0.75) < 0.08):
        return STATUS_UNCERTAIN
    return status


def _suggested_action(status: str) -> str:
    return {
        STATUS_VERIFIED: "Accept", STATUS_REVIEW: "Send to human review",
        STATUS_REJECTED: "Reject", STATUS_UNCERTAIN: "Escalate to human review",
    }[status]


def object_verdicts(container: Container, project_id: str) -> list[ObjectVerdict]:
    verifier = container.verifier  # the EXISTING deterministic verifier
    out: list[ObjectVerdict] = []
    for img in container.images.by_project(project_id):
        data = container.cas.get(img.sha256)
        for ann in container.annotations.by_image(img.id):
            verdict = verifier.verify(data, ann)
            status = _classify(verdict.verdict, verdict.confidence)
            box = None
            if isinstance(ann.geometry, Box2D):
                g = ann.geometry
                box = (g.x, g.y, g.w, g.h)
            out.append(ObjectVerdict(
                object_id=ann.id, image_id=img.id, image_name=img.id[:12], label=ann.label,
                detection_confidence=round(ann.confidence, 4), status=status,
                verification_confidence=round(verdict.confidence, 4),
                suggested_action=_suggested_action(status), rationale=verdict.rationale,
                state=ann.state, box=box,
            ))
    return out


# --- evidence (all measured) ----------------------------------------------
def _stars(value: float) -> int:
    return max(0, min(5, round(value * 5)))


def _geometry_consistency(ann) -> tuple[float, str]:
    g = ann.geometry
    if not isinstance(g, Box2D):
        return 1.0, "non-box geometry"
    if g.w <= 0 or g.h <= 0 or g.x < 0 or g.y < 0:
        return 0.0, "invalid box (non-positive or off-image)"
    aspect = g.w / g.h if g.h else 0
    score = 1.0
    detail = f"box {g.w:.0f}×{g.h:.0f}, aspect {aspect:.2f}"
    if aspect > 6 or aspect < 1 / 6:
        score = 0.4
        detail += " (extreme aspect ratio)"
    return score, detail


def _context_consistency(container: Container, image_id: str, ann) -> tuple[float, str]:
    if not isinstance(ann.geometry, Box2D):
        return 1.0, "n/a"
    max_iou = 0.0
    for other in container.annotations.by_image(image_id):
        if other.id == ann.id or not isinstance(other.geometry, Box2D):
            continue
        max_iou = max(max_iou, overlap_iou(ann.geometry, other.geometry))
    score = 1.0 - min(1.0, max_iou)
    detail = (f"max neighbour IoU {max_iou:.2f}"
              + (" — overlaps another object" if max_iou > 0.5 else " — well isolated"))
    return score, detail


def evidence_for(container: Container, verdict: ObjectVerdict) -> Evidence:
    ann = container.annotations.get(verdict.object_id)
    if ann is None:
        return Evidence("Annotation unavailable", "unavailable", [], 0.0, "unknown",
                        "unavailable", [])
    geom_score, geom_detail = _geometry_consistency(ann)
    ctx_score, ctx_detail = _context_consistency(container, verdict.image_id, ann)
    mask_ok = ann.mask is not None and not mask_is_empty(ann.mask.rle)
    evidence_quality = round((geom_score + ctx_score + (1.0 if mask_ok else 0.4)) / 3, 3)

    hist = _historical_agreement(container, verdict.image_id)

    stars = [
        Star("Verification Confidence", _stars(verdict.verification_confidence),
             f"{verdict.verification_confidence:.0%} (deterministic verifier)"),
        Star("Detection Confidence", _stars(verdict.detection_confidence),
             f"{verdict.detection_confidence:.0%}"),
        Star("Evidence Quality", _stars(evidence_quality), f"composite {evidence_quality}"),
        Star("Geometry Consistency", _stars(geom_score), geom_detail),
        Star("Context Consistency", _stars(ctx_score), ctx_detail),
        Star("Historical Agreement", None if hist is None else _stars(hist),
             "no historical match" if hist is None else f"memory similarity {hist:.2f}"),
    ]
    evidence_lines = [
        geom_detail,
        ctx_detail,
        "segmentation mask present and non-empty" if mask_ok else "segmentation mask missing or empty",
        verdict.rationale,
    ]
    risk = {STATUS_VERIFIED: "Low", STATUS_REVIEW: "Medium",
            STATUS_UNCERTAIN: "Medium", STATUS_REJECTED: "High"}[verdict.status]
    return Evidence(
        summary=f"{verdict.status}: {verdict.rationale}.",
        reason=verdict.rationale,
        evidence=evidence_lines,
        verification_confidence=verdict.verification_confidence,
        risk=risk,
        recommendation=verdict.suggested_action,
        stars=stars,
    )


def _historical_agreement(container: Container, image_id: str) -> float | None:
    img = container.images.get(image_id)
    if img is None:
        return None
    project_id = img.project_id
    try:
        ctx = build_dataset_context(project_id, container.settings, container.images)
        fp = DatasetFingerprint(
            resolution_mp=ctx.resolution_summary.get("megapixels_max", -1.0),
            dataset_size=ctx.image_count)
        matches = container.memory.recall(fp).matches
    except Exception:
        return None
    return matches[0].score if matches else None


# --- statistics ------------------------------------------------------------
def verification_stats(container: Container, verdicts: list[ObjectVerdict]) -> VerificationStats:
    n = len(verdicts) or 1
    verified = sum(1 for v in verdicts if v.status == STATUS_VERIFIED)
    rejected = sum(1 for v in verdicts if v.status == STATUS_REJECTED)
    review = sum(1 for v in verdicts if v.status in (STATUS_REVIEW, STATUS_UNCERTAIN))
    avg_ver = round(sum(v.verification_confidence for v in verdicts) / n, 4)
    avg_det = round(sum(v.detection_confidence for v in verdicts) / n, 4)
    # Agreement: detector's confidence and the verifier's decision point the same way
    # (both "good" = high conf + verified, or both "not good").
    agree = sum(1 for v in verdicts
                if (v.detection_confidence >= 0.75) == (v.status == STATUS_VERIFIED))
    agreement = round(agree / n, 4)
    return VerificationStats(
        verified=verified, rejected=rejected, needs_review=review,
        avg_verification_confidence=avg_ver, avg_detection_confidence=avg_det,
        agreement_rate=agreement, disagreement_rate=round(1 - agreement, 4),
        review_percentage=round(review / n, 4),
        verification_runtime="unavailable (not persisted per dataset)",
    )


# --- historical comparison -------------------------------------------------
def historical_comparison(container: Container, project_id: str) -> HistoricalComparison:
    try:
        ctx = build_dataset_context(project_id, container.settings, container.images)
        fp = DatasetFingerprint(
            resolution_mp=ctx.resolution_summary.get("megapixels_max", -1.0),
            dataset_size=ctx.image_count)
        guidance = container.memory.recall(fp)
    except Exception as exc:
        return HistoricalComparison(False, f"Historical data unavailable ({type(exc).__name__}).")
    if not guidance.matches:
        return HistoricalComparison(False, "No similar objects found in Engineering Memory.")
    matches = []
    for m in guidance.matches:
        mem = m.memory
        corrections = mem.verification_outcomes.frequently_corrected_labels or \
            mem.verification_outcomes.common_semantic_failures
        matches.append({
            "dataset": mem.project_id or mem.id,
            "similarity": round(m.score, 3),
            "historical_verification": f"approval {mem.execution_metrics.approval_rate:.0%}, "
                                       f"review {mem.execution_metrics.review_rate:.0%}",
            "past_corrections": ", ".join(f"{k}:{v}" for k, v in list(corrections.items())[:3]) or "none",
            "analyst_notes": "; ".join(mem.analyst_conclusions.root_causes[:1]) or "none",
            "agreement_rate": f"{1 - mem.execution_metrics.review_rate:.0%}",
            "previous_review_decision": "review-heavy" if mem.execution_metrics.review_rate > 0.3
                                        else "mostly auto-accepted",
        })
    return HistoricalComparison(
        True, f"{len(matches)} similar dataset(s) informed this comparison "
              f"(closest similarity {guidance.matches[0].score}).", matches)


# --- timeline --------------------------------------------------------------
def timeline_for(verdict: ObjectVerdict) -> list[TimelineStep]:
    reviewed = verdict.state in ("accepted", "fixed", "rejected")
    human = "Completed" if reviewed else ("Pending" if verdict.state == "needs_review" else "Skipped")
    final = {
        "auto_accepted": "Auto-accepted", "needs_review": "Awaiting review",
        "rejected_auto": "Auto-rejected", "accepted": "Accepted (human)",
        "fixed": "Fixed (human)", "rejected": "Rejected (human)",
    }.get(verdict.state, verdict.state)
    return [
        TimelineStep("Detection", "Completed", "unavailable"),
        TimelineStep("Verification", "Completed", "unavailable"),
        TimelineStep("Recommendation", verdict.suggested_action, "unavailable"),
        TimelineStep("Human Review", human, "unavailable"),
        TimelineStep("Final Decision", final, "unavailable"),
    ]


# --- verification report (measured summary) --------------------------------
def verification_report(container: Container, project_id: str) -> str:
    verdicts = object_verdicts(container, project_id)
    stats = verification_stats(container, verdicts)
    proj = container.projects.get(project_id)
    lines = [
        f"# Verification Report — {proj.name if proj else project_id}",
        "",
        "## Statistics",
        f"- Verified: {stats.verified}",
        f"- Needs review: {stats.needs_review}",
        f"- Rejected: {stats.rejected}",
        f"- Avg verification confidence: {stats.avg_verification_confidence}",
        f"- Avg detection confidence: {stats.avg_detection_confidence}",
        f"- Agreement rate: {stats.agreement_rate:.0%}",
        f"- Disagreement rate: {stats.disagreement_rate:.0%}",
        f"- Review percentage: {stats.review_percentage:.0%}",
        f"- Verification runtime: {stats.verification_runtime}",
        "",
        "## Objects",
        "| Object | Class | Det. Conf | Status | Ver. Conf | Action | Reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in verdicts:
        lines.append(f"| {v.object_id[:8]} | {v.label} | {v.detection_confidence} | "
                     f"{v.status} | {v.verification_confidence} | {v.suggested_action} | {v.rationale} |")
    return "\n".join(lines) + "\n"


# --- human review (existing state machine) ---------------------------------
def apply_review(container: Container, annotation_id: str, action: str) -> tuple[bool, str]:
    """Integrate with the existing review workflow: validate against the annotation
    state machine, then persist via the repo. No new logic — just the legal move."""
    target = REVIEW_ACTIONS.get(action)
    if target is None:
        return False, f"Unknown action '{action}'."
    ann = container.annotations.get(annotation_id)
    if ann is None:
        return False, "Annotation not found."
    current = AnnotationState(ann.state)
    if current == target:
        return True, f"Already {target.value}."
    if not is_legal_transition(current, target):
        return False, f"'{action}' not permitted from state '{current.value}'."
    container.annotations.set_state(annotation_id, target.value)
    return True, f"{action} → {target.value}."
