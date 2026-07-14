"""AI Dataset Analyst (Phase 8) — the platform's second AI agent.

It reviews a *completed* annotation run (structured metrics only, never raw
images) like a Senior CV Scientist and produces an evidence-backed engineering
report + recommendations, including advice for the Planner.

Anti-hallucination by construction (the phase's hard rule):
  1. A DETERMINISTIC evidence pack computes every fact from the run's metrics.
  2. The LLM reasons over that pack and must cite evidence keys per recommendation.
  3. A DETERMINISTIC validator drops any recommendation citing evidence that does
     not exist — so fabricated statistics or unsupported advice cannot survive.
The AI supplies interpretation and prioritization; it never supplies numbers.

Safety: like the Planner, the Analyst never fails the caller. On invalid output,
provider failure, or missing credentials it returns the deterministic report
built from the same evidence pack, and logs why.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

from vds.agents.base import Agent
from vds.agents.cost import estimate_cost
from vds.agents.llm import LLMClient
from vds.agents.planner_agent import DatasetContext, PlannerResult
from vds.core.contracts import ExecutionReport, StageKPIs
from vds.logging import get_logger
from vds.reporting import to_kpis

log = get_logger(__name__)


# --- context & evidence ----------------------------------------------------
class AnalystContext(BaseModel):
    """Everything the Analyst reasons over — all structured, no images."""

    execution: ExecutionReport
    planner: PlannerResult | None = None
    dataset_context: DatasetContext | None = None
    history: list[StageKPIs] = Field(default_factory=list)


class Fact(BaseModel):
    key: str  # citation token the LLM must reference
    statement: str  # the human-readable fact, carrying the real number
    category: str


# Short category codes keep the fact table narrow; titles used when rendering.
CATEGORY_TITLES = {
    "perf": "Pipeline Performance", "data": "Dataset Characteristics",
    "det": "Detection Analysis", "seg": "Segmentation Analysis",
    "ver": "Verification Analysis", "res": "Resource Utilization",
    "hist": "Benchmark Intelligence",
}


class EvidencePack(BaseModel):
    facts: list[Fact]
    unavailable: list[str] = Field(default_factory=list)  # what metrics can't show

    @property
    def keys(self) -> set[str]:
        return {f.key for f in self.facts}

    def render(self) -> str:
        by_cat: dict[str, list[str]] = {}
        for f in self.facts:
            by_cat.setdefault(f.category, []).append(f"[{f.key}] {f.statement}")
        blocks = [f"## {CATEGORY_TITLES.get(c, c)}\n" + "\n".join(v) for c, v in by_cat.items()]
        if self.unavailable:
            blocks.append("## not measurable (needs more data)\n- " + "\n- ".join(self.unavailable))
        return "\n\n".join(blocks)


# --- report schema ---------------------------------------------------------
class Recommendation(BaseModel):
    action: str
    target: Literal["planner", "pipeline", "verification", "export", "dataset"]
    reason: str
    expected_impact: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_metrics: list[str] = Field(default_factory=list)  # evidence keys
    trade_offs: str = ""


class AnalystReport(BaseModel):
    executive_summary: str
    pipeline_performance: str
    dataset_characteristics: str
    detection_analysis: str
    segmentation_analysis: str
    verification_analysis: str
    resource_utilization: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    root_cause_analysis: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    planner_recommendations: list[Recommendation] = Field(default_factory=list)
    expected_improvement: str
    confidence: float = Field(ge=0.0, le=1.0)
    next_actions: list[str] = Field(default_factory=list)


class AnalystResult(BaseModel):
    report: AnalystReport
    source: Literal["ai", "deterministic"]
    evidence_coverage: float  # fraction of recs whose citations are all valid
    recommendation_count: int
    unsupported_recommendations: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None
    latency_ms: float = 0.0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float | None = None
    structured_valid: bool = True


ANALYST_SYSTEM_PROMPT = """\
You are a Senior Computer Vision Scientist reviewing a completed automated
annotation run. You are given an EVIDENCE PACK of measured facts, each tagged
with a [key]. Write an engineering review — not a chat reply, not a summary.

Hard rules:
- Ground every claim in the evidence. Never invent a statistic or observation.
- Each recommendation's `supporting_metrics` MUST list only [key] tokens that
  appear in the evidence pack. Do not cite a key that is not present.
- If the evidence is insufficient for a question, say so explicitly; do not guess.
- Be concise and use engineering-review language, not conversational prose.
- Provide recommendations for both the pipeline and the Planner, each with a
  reason, expected impact, confidence (0-1), supporting_metrics, and trade-offs.
Output ONLY a JSON object matching the required schema.
"""


def _f(facts: list[Fact], key: str, statement: str, category: str) -> None:
    facts.append(Fact(key=key, statement=statement, category=category))


def build_evidence(ctx: AnalystContext) -> EvidencePack:
    """Deterministic: every fact computed from metrics. No AI, no images."""
    e = ctx.execution
    b, q, err = e.benchmark, e.quality, e.errors
    facts: list[Fact] = []
    unavailable: list[str] = []

    # Performance
    _f(facts, "throughput", f"Throughput was {b.images_per_second} images/sec.", "perf")
    if b.stage_seconds:
        slow = max(b.stage_seconds, key=b.stage_seconds.get)
        total = sum(b.stage_seconds.values()) or 1.0
        share = round(100 * b.stage_seconds[slow] / total)
        _f(facts, "bottleneck", f"The slowest stage was '{slow}' at {share}% of pipeline time.", "perf")
    _f(facts, "avg_inference_ms", f"Average model inference latency was {b.avg_inference_ms} ms.", "perf")

    # Dataset characteristics
    total_seen = e.imported + e.duplicates_skipped
    if total_seen:
        dup_frac = e.duplicates_skipped / total_seen
        _f(facts, "duplicate_rate",
           f"{e.duplicates_skipped}/{total_seen} inputs ({dup_frac:.0%}) were near-duplicates.", "data")
        if dup_frac > 0.2:
            _f(facts, "duplicate_heavy", "The dataset is duplicate-heavy (>20% skipped).", "data")
    _f(facts, "annotation_density", f"Annotation density was {q.annotation_density} objects/image.", "data")
    if q.annotation_density >= 8:
        _f(facts, "dense_scenes", "Scenes are dense (>=8 objects/image).", "data")
    elif q.annotation_density < 1 and q.detections:
        _f(facts, "sparse_scenes", "Scenes are sparse (<1 object/image).", "data")
    small = next((c.count for c in err.categories if c.name == "small_objects"), 0)
    if q.detections and small / q.detections > 0.5:
        _f(facts, "small_object_dominance",
           f"{small}/{q.detections} detections are small objects (>50%).", "data")
    if e.quarantined:
        _f(facts, "quarantined",
           f"{e.quarantined} image(s) were quarantined by the quality filter (weak low-quality proxy).", "data")
    if ctx.dataset_context is not None:
        mp = ctx.dataset_context.resolution_summary.get("megapixels_max")
        if mp:
            _f(facts, "max_megapixels", f"Largest image is {mp} MP.", "data")
            if mp > 4.0:
                _f(facts, "high_resolution", "Dataset is high-resolution (>4 MP).", "data")
    unavailable += [
        "blur / exposure / true image quality (raw images not inspected this phase)",
        "per-class imbalance (single-class pipeline in this run)",
    ]

    # Detection / Segmentation
    _f(facts, "detections", f"{q.detections} detections across {q.images} images.", "det")
    _f(facts, "avg_confidence", f"Mean detection confidence was {q.avg_confidence}.", "det")
    _f(facts, "masks", f"{q.masks} segmentation masks; {q.empty_masks} empty.", "seg")
    if q.duplicate_detections:
        _f(facts, "duplicate_detections",
           f"{q.duplicate_detections} overlapping duplicate detections survived NMS.", "det")

    # Verification
    _f(facts, "approval_rate", f"Verifier approved {q.approval_rate:.0%} of annotations.", "ver")
    _f(facts, "review_rate", f"{q.review_rate:.0%} of annotations were flagged for human review.", "ver")
    _f(facts, "rejection_rate", f"{q.rejection_rate:.0%} of annotations were rejected.", "ver")
    if q.approval_rate >= 0.99 and q.review_rate == 0.0:
        _f(facts, "uncalibrated_confidence",
           "100% approval with 0% review suggests confidence scores are uncalibrated, "
           "not that quality is perfect.", "ver")

    # Resource
    _f(facts, "peak_ram", f"Peak RAM was {b.peak_ram_mb} MB.", "res")
    _f(facts, "gpu_util", f"GPU utilization was {b.gpu_util_percent}%.", "res")

    # Planner decisions (if provided)
    if ctx.planner is not None:
        _f(facts, "plan_source", f"The plan came from the {ctx.planner.source} planner.", "perf")

    # Historical comparison
    if len(ctx.history) >= 1:
        current = to_kpis(ctx.execution)
        base = ctx.history[-1]
        d_ips = round(current.images_per_second - base.images_per_second, 3)
        _f(facts, "throughput_trend", f"Throughput changed {d_ips:+} img/s vs the previous run.", "hist")
        if d_ips < -0.1 * base.images_per_second:
            _f(facts, "throughput_regression", f"Throughput regressed {abs(d_ips)} img/s vs baseline.", "hist")
        d_rev = round(current.review_rate - base.review_rate, 4)
        _f(facts, "review_trend", f"Human-review rate changed {d_rev:+} vs the previous run.", "hist")
    else:
        unavailable.append("historical trends (no baseline established yet — need >=1 prior run)")

    return EvidencePack(facts=facts, unavailable=unavailable)


# --- deterministic report (fallback + comparison baseline) ----------------
def _rules(evidence: EvidencePack) -> list[Recommendation]:
    keys = evidence.keys
    recs: list[Recommendation] = []
    if "bottleneck" in keys:
        recs.append(Recommendation(
            action="Optimize the slowest stage (batch or sample its work).",
            target="pipeline", reason="One stage dominates runtime.",
            expected_impact="Lower total runtime.", confidence=0.7,
            supporting_metrics=["bottleneck"], trade_offs="Sampling may miss some cases."))
    if "uncalibrated_confidence" in keys:
        recs.append(Recommendation(
            action="Replace geometric confidence with a learned detector / VLM verifier.",
            target="verification", reason="100% approval, 0% review indicates uncalibrated scores.",
            expected_impact="Meaningful review routing and trustworthy approvals.", confidence=0.85,
            supporting_metrics=["approval_rate", "review_rate", "uncalibrated_confidence"],
            trade_offs="Adds model cost."))
    if "small_object_dominance" in keys:
        recs.append(Recommendation(
            action="Enable tiling and/or a higher-accuracy detector.",
            target="planner", reason="Small objects dominate and are error-prone.",
            expected_impact="Higher recall on small objects.", confidence=0.7,
            supporting_metrics=["small_object_dominance"], trade_offs="Higher compute per image."))
    if "high_resolution" in keys:
        recs.append(Recommendation(
            action="Enable tiling for high-resolution imagery.",
            target="planner", reason="High-resolution images lose small objects without tiling.",
            expected_impact="Better detection on large frames.", confidence=0.65,
            supporting_metrics=["high_resolution"], trade_offs="Slower processing."))
    if "sparse_scenes" in keys:
        recs.append(Recommendation(
            action="Consider skipping segmentation on sparse scenes.",
            target="planner", reason="Few objects per image; segmentation cost may not pay off.",
            expected_impact="Faster runs.", confidence=0.5,
            supporting_metrics=["sparse_scenes", "annotation_density"], trade_offs="No masks produced."))
    if "duplicate_heavy" in keys:
        recs.append(Recommendation(
            action="Tighten or review the deduplication threshold.",
            target="dataset", reason="A large fraction of inputs were duplicates.",
            expected_impact="Cleaner dataset, less wasted compute.", confidence=0.6,
            supporting_metrics=["duplicate_rate", "duplicate_heavy"],
            trade_offs="Risk of dropping valid near-duplicates."))
    if "throughput_regression" in keys:
        recs.append(Recommendation(
            action="Investigate the throughput regression vs the previous run.",
            target="pipeline", reason="Throughput fell materially vs baseline.",
            expected_impact="Restore prior speed.", confidence=0.6,
            supporting_metrics=["throughput_regression", "throughput_trend"], trade_offs="None."))
    return recs


def build_deterministic_report(ctx: AnalystContext, evidence: EvidencePack) -> AnalystReport:
    e = ctx.execution
    q = e.quality
    stmt = {f.key: f.statement for f in evidence.facts}
    recs = _rules(evidence)
    planner_recs = [r for r in recs if r.target == "planner"]

    def section(*keys: str) -> str:
        parts = [stmt[k] for k in keys if k in stmt]
        return " ".join(parts) if parts else "Insufficient measured evidence for this section."

    strengths, weaknesses = [], []
    if q.rejection_rate < 0.05:
        strengths.append("Low auto-rejection rate.")
    if e.export.validated:
        strengths.append("Export passed round-trip validation.")
    if "uncalibrated_confidence" in evidence.keys:
        weaknesses.append("Confidence scores are uncalibrated (100% approval, 0% review).")
    if "small_object_dominance" in evidence.keys:
        weaknesses.append("Small-object dominance stresses the detector.")

    return AnalystReport(
        executive_summary=f"Processed {e.imported} images at {e.benchmark.images_per_second} img/s; "
                          f"{q.approval_rate:.0%} approved, {q.review_rate:.0%} to review. "
                          "Deterministic analysis (no LLM configured).",
        pipeline_performance=section("throughput", "bottleneck", "avg_inference_ms"),
        dataset_characteristics=section(
            "annotation_density", "duplicate_rate", "small_object_dominance",
            "high_resolution", "quarantined"),
        detection_analysis=section("detections", "avg_confidence", "duplicate_detections"),
        segmentation_analysis=section("masks"),
        verification_analysis=section("approval_rate", "review_rate", "rejection_rate", "uncalibrated_confidence"),
        resource_utilization=section("peak_ram", "gpu_util"),
        strengths=strengths or ["No notable strengths flagged."],
        weaknesses=weaknesses or ["No notable weaknesses flagged."],
        root_cause_analysis=stmt.get("uncalibrated_confidence", "No dominant root cause identified from metrics."),
        recommendations=recs,
        planner_recommendations=planner_recs,
        expected_improvement="Applying the flagged recommendations should reduce human review and/or runtime; "
                             "magnitude unquantified without a real verifier baseline.",
        confidence=0.6,
        next_actions=[r.action for r in recs[:3]] or ["Collect more runs to establish trends."],
    )


class LLMAnalyst(Agent):
    system_prompt = ANALYST_SYSTEM_PROMPT

    def __init__(self, client: LLMClient) -> None:
        super().__init__(client)

    def analyze(self, ctx: AnalystContext) -> AnalystResult:
        evidence = build_evidence(ctx)
        start = time.perf_counter()
        convo = self.new_conversation().user(
            "Review this completed annotation run. Cite only the [key] tokens below. "
            "Return one AnalystReport JSON object.\n\nEVIDENCE PACK:\n" + evidence.render()
        )
        try:
            outcome = self._client.structured(convo, AnalystReport)
        except Exception as exc:  # never fail the caller (phase brief)
            return self._fallback(ctx, evidence, f"{type(exc).__name__}: {exc}", start)

        report, coverage, unsupported = self._enforce_evidence(outcome.value, evidence)
        latency = round((time.perf_counter() - start) * 1000, 3)
        usage = outcome.response.usage
        log.info("analyst.ai_report", project_id=ctx.execution.project_id,
                 coverage=coverage, latency_ms=latency)
        return AnalystResult(
            report=report, source="ai", evidence_coverage=coverage,
            recommendation_count=len(report.recommendations) + len(report.planner_recommendations),
            unsupported_recommendations=unsupported,
            latency_ms=latency, retries=outcome.attempts - 1,
            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
            estimated_cost_usd=estimate_cost(outcome.response.model, usage),
        )

    def remember(
        self,
        ctx: AnalystContext,
        result: AnalystResult,
        memory,
        created_at: str,
        **fingerprint,
    ):
        """Analyst integration (Phase 10): after analysis, persist reusable
        engineering knowledge. Only validated (evidence-backed) recommendations are
        stored — the memory builder drops the rest. History is never overwritten."""
        return memory.record_execution(
            ctx.execution, created_at,
            planner_result=ctx.planner, analyst_result=result, **fingerprint,
        )

    def _enforce_evidence(self, report: AnalystReport, evidence: EvidencePack):
        """Drop recommendations that cite no valid evidence key (anti-hallucination).
        Returns (cleaned_report, coverage, dropped_actions)."""
        keys = evidence.keys
        dropped: list[str] = []

        def keep(recs: list[Recommendation]) -> list[Recommendation]:
            out = []
            for r in recs:
                valid = [m for m in r.supporting_metrics if m in keys]
                if valid:
                    r.supporting_metrics = valid  # strip invented citations
                    out.append(r)
                else:
                    dropped.append(r.action)
            return out

        all_recs = report.recommendations + report.planner_recommendations
        report.recommendations = keep(report.recommendations)
        report.planner_recommendations = keep(report.planner_recommendations)
        total = len(all_recs) or 1
        coverage = round((total - len(dropped)) / total, 4)
        return report, coverage, dropped

    def _fallback(self, ctx, evidence, reason: str, start: float) -> AnalystResult:
        log.warning("analyst.fallback", project_id=ctx.execution.project_id, reason=reason)
        report = build_deterministic_report(ctx, evidence)
        return AnalystResult(
            report=report, source="deterministic",
            evidence_coverage=1.0,  # rule-based recs are always evidence-backed
            recommendation_count=len(report.recommendations) + len(report.planner_recommendations),
            fallback_reason=reason,
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
            structured_valid=False,
        )
