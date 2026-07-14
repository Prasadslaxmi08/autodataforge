"""Dataset Intelligence view-model (Phase 15) — plain data for the Intelligence Workspace.

No Qt here. It SUMMARIZES the existing AI Dataset Analyst: it feeds the real cached
ExecutionReport to `LLMAnalyst` exactly as implemented, reuses the Analyst's own
`build_evidence` for measured facts, and reads Engineering-Memory trends — it never
computes new intelligence. Every value comes from a measured metric or a validated
Analyst recommendation; anything the backend can't supply is marked unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vds.agents.analyst_agent import (
    AnalystContext,
    AnalystResult,
    LLMAnalyst,
    build_evidence,
)
from vds.agents.planner_agent import build_dataset_context
from vds.container import Container
from vds.core.contracts import ExecutionReport
from vds.memory import DatasetFingerprint
from vds.memory.trends import TrendAnalyzer

# Problem evidence keys the Analyst's evidence pack may emit -> a readable title.
_ISSUE_TITLES = {
    "duplicate_heavy": "High Duplicate Rate",
    "small_object_dominance": "Small Objects Dominate",
    "dense_scenes": "Dense Scenes",
    "sparse_scenes": "Sparse Scenes",
    "uncalibrated_confidence": "Low Verification Agreement",
    "high_resolution": "High-Resolution Imagery",
    "throughput_regression": "Throughput Regression",
    "quarantined": "Quarantined Images",
}


@dataclass
class ExecutiveSummary:
    dataset: str
    version: int
    size_mb: float
    image_count: int
    overall_health: int  # 0..100
    annotation_quality: float
    verification_confidence: float
    production_readiness: str
    overall_recommendation: str
    historical_improvement: str
    analyst_summary: str
    analyst_confidence: float
    source: str  # ai | deterministic


@dataclass
class HealthKPI:
    name: str
    value: str
    score: int | None  # 0..100 for the gauge/bar, or None when unavailable


@dataclass
class RootCauseIssue:
    title: str
    description: str
    evidence: list[str]
    impact: str
    recommendation: str
    expected_improvement: str
    confidence: float


@dataclass
class RankedRecommendation:
    priority: str  # HIGH | MEDIUM | LOW
    problem: str
    recommendation: str
    expected_gain: str
    estimated_effort: str
    expected_review_reduction: str
    expected_runtime_impact: str
    rationale: str
    confidence: float
    target: str


@dataclass
class TrendSeries:
    metric: str
    series: list[float]
    first: float
    last: float
    delta: float
    improved: bool


@dataclass
class HistoricalIntelligence:
    available: bool
    note: str
    trends: list[TrendSeries] = field(default_factory=list)
    matches: list[dict] = field(default_factory=list)
    runs: int = 0


@dataclass
class ReadinessCriterion:
    name: str
    met: bool
    reasoning: str


@dataclass
class DatasetIntelligence:
    summary: ExecutiveSummary
    kpis: list[HealthKPI]
    issues: list[RootCauseIssue]
    recommendations: list[RankedRecommendation]
    historical: HistoricalIntelligence
    readiness: list[ReadinessCriterion]
    analyst_report_markdown: str


# --- helpers ---------------------------------------------------------------
def _score(value: float) -> int:
    return max(0, min(100, round(value * 100)))


def _priority(confidence: float) -> str:
    return "HIGH" if confidence >= 0.8 else ("MEDIUM" if confidence >= 0.6 else "LOW")


def _effort(target: str) -> str:
    return {"pipeline": "Medium", "verification": "High", "planner": "Medium",
            "dataset": "Low", "export": "Low"}.get(target, "Medium")


# --- sections --------------------------------------------------------------
def _storage_mb(container: Container, project_id: str) -> float:
    total = 0
    for img in container.images.by_project(project_id):
        p = container.cas.path(img.sha256)
        if p.exists():
            total += p.stat().st_size
    return round(total / (1024 * 1024), 2)


def _health_kpis(container: Container, report: ExecutionReport,
                 memory_influence: str) -> tuple[list[HealthKPI], int]:
    q = report.quality
    ann_quality = max(0.0, q.approval_rate - q.rejection_rate)
    ver_agreement = max(0.0, 1.0 - q.review_rate)
    dup_rate = q.duplicate_detections / q.detections if q.detections else 0.0
    seg_quality = (q.masks - q.empty_masks) / q.masks if q.masks else 0.0
    export_rate = 1.0 if report.export.validated else 0.0

    kpis = [
        HealthKPI("Annotation Quality", f"{ann_quality:.0%}", _score(ann_quality)),
        HealthKPI("Verification Agreement", f"{ver_agreement:.0%}", _score(ver_agreement)),
        HealthKPI("Duplicate Rate", f"{dup_rate:.0%}", _score(1 - dup_rate)),
        HealthKPI("Review Rate", f"{q.review_rate:.0%}", _score(1 - q.review_rate)),
        HealthKPI("Detection Quality", f"{q.avg_confidence:.0%}", _score(q.avg_confidence)),
        HealthKPI("Segmentation Quality", f"{seg_quality:.0%}", _score(seg_quality)),
        HealthKPI("Export Success Rate", f"{export_rate:.0%}", _score(export_rate)),
        HealthKPI("Planner Confidence", "unavailable (run the Planner)", None),
        HealthKPI("Engineering Memory Influence", memory_influence, None),
    ]
    scored = [k.score for k in kpis if k.score is not None]
    overall = round(sum(scored) / len(scored)) if scored else 0
    return kpis, overall


def _root_causes(evidence, result: AnalystResult) -> list[RootCauseIssue]:
    facts = {f.key: f.statement for f in evidence.facts}
    recs = result.report.recommendations + result.report.planner_recommendations
    issues: list[RootCauseIssue] = []
    for key, title in _ISSUE_TITLES.items():
        if key not in facts:
            continue
        # Find a validated Analyst recommendation that cites this evidence key.
        rec = next((r for r in recs if key in r.supporting_metrics), None)
        support = [facts[k] for k in facts if k.startswith(key.split("_")[0])][:3] or [facts[key]]
        issues.append(RootCauseIssue(
            title=title, description=facts[key], evidence=support,
            impact=(rec.expected_impact if rec else "See recommendations."),
            recommendation=(rec.action if rec else "Review this dataset characteristic."),
            expected_improvement=(rec.expected_impact if rec else "—"),
            confidence=(rec.confidence if rec else 0.5),
        ))
    return issues


def _rank_recommendations(result: AnalystResult) -> list[RankedRecommendation]:
    recs = result.report.recommendations + result.report.planner_recommendations
    ranked = [
        RankedRecommendation(
            priority=_priority(r.confidence), problem=r.reason, recommendation=r.action,
            expected_gain=r.expected_impact, estimated_effort=_effort(r.target),
            expected_review_reduction="unavailable (qualitative)",
            expected_runtime_impact="unavailable (qualitative)",
            rationale=r.reason + (f" Trade-offs: {r.trade_offs}" if r.trade_offs else ""),
            confidence=r.confidence, target=r.target,
        )
        for r in recs
    ]
    ranked.sort(key=lambda x: x.confidence, reverse=True)
    return ranked


def _historical(container: Container, report: ExecutionReport) -> HistoricalIntelligence:
    memories = container.memory.all()
    if not memories:
        return HistoricalIntelligence(False, "No historical datasets in Engineering Memory yet.")
    ev = TrendAnalyzer().evolution(memories)
    trends = [TrendSeries(t.metric, t.series, t.first, t.last, t.delta, t.improved)
              for t in ev.values()]
    matches = []
    try:
        ctx = build_dataset_context(report.project_id, container.settings, container.images)
        fp = DatasetFingerprint(
            resolution_mp=ctx.resolution_summary.get("megapixels_max", -1.0),
            dataset_size=ctx.image_count)
        for m in container.memory.recall(fp).matches:
            matches.append({
                "dataset": m.memory.project_id or m.memory.id,
                "similarity": round(m.score, 3),
                "review_rate": f"{m.memory.execution_metrics.review_rate:.0%}",
                "quality": m.memory.benchmark_summary.quality_score,
            })
    except Exception:
        pass
    return HistoricalIntelligence(True,
                                  f"{len(memories)} historical run(s) available.",
                                  trends, matches, len(memories))


def _readiness(report: ExecutionReport, evidence) -> list[ReadinessCriterion]:
    q = report.quality
    keys = evidence.keys
    health_ok = (q.approval_rate - q.rejection_rate) >= 0.7
    return [
        ReadinessCriterion(
            "Ready for Training",
            health_ok and q.review_rate < 0.2 and q.rejection_rate < 0.1,
            f"annotation quality {q.approval_rate - q.rejection_rate:.0%}, "
            f"review {q.review_rate:.0%}, rejection {q.rejection_rate:.0%}"),
        ReadinessCriterion(
            "Requires Human Review", q.review_rate >= 0.2,
            f"{q.review_rate:.0%} of annotations flagged for review"),
        ReadinessCriterion(
            "Requires Re-Annotation", q.rejection_rate >= 0.2 or q.invalid_annotations > 0,
            f"rejection {q.rejection_rate:.0%}, {q.invalid_annotations} invalid geometry"),
        ReadinessCriterion(
            "Requires Verification",
            q.avg_confidence < 0.6 or "uncalibrated_confidence" in keys,
            f"avg confidence {q.avg_confidence:.0%}"
            + ("; confidence appears uncalibrated" if "uncalibrated_confidence" in keys else "")),
        ReadinessCriterion(
            "Requires Additional Data", q.images_with_no_detection > 0 or report.imported < 20,
            f"{q.images_with_no_detection} image(s) with no detection, "
            f"{report.imported} images total"),
    ]


def _overall_recommendation(readiness: list[ReadinessCriterion]) -> tuple[str, str]:
    by = {c.name: c for c in readiness}
    if by["Ready for Training"].met:
        return "Ready for Training", "Ready"
    if by["Requires Re-Annotation"].met:
        return "Needs Improvement", "Not Ready"
    if by["Requires Human Review"].met or by["Requires Verification"].met:
        return "Human Review Recommended", "Conditional"
    return "Needs Improvement", "Conditional"


def _historical_improvement(historical: HistoricalIntelligence) -> str:
    if not historical.available:
        return "No historical baseline yet."
    q = next((t for t in historical.trends if t.metric == "quality_score"), None)
    if q is None:
        return "Historical data available."
    return f"Quality {'improved' if q.improved else 'regressed'} {q.delta:+} vs first run."


# --- analyst report -> markdown (presentation of validated output) ---------
def analyst_report_markdown(result: AnalystResult) -> str:
    r = result.report

    def recs(rs):
        return "\n".join(
            f"- **{x.action}** ({x.target}, confidence {x.confidence})\n"
            f"  - reason: {x.reason}\n  - impact: {x.expected_impact}\n"
            f"  - evidence: {', '.join(x.supporting_metrics) or 'none'}"
            for x in rs) or "- none"

    return "\n".join([
        f"# Engineering Report ({result.source})",
        f"_evidence coverage: {result.evidence_coverage} · confidence: {r.confidence}_",
        "",
        f"## Executive Summary\n{r.executive_summary}",
        f"## Pipeline Performance\n{r.pipeline_performance}",
        f"## Dataset Characteristics\n{r.dataset_characteristics}",
        f"## Detection Analysis\n{r.detection_analysis}",
        f"## Segmentation Analysis\n{r.segmentation_analysis}",
        f"## Verification Analysis\n{r.verification_analysis}",
        f"## Resource Utilization\n{r.resource_utilization}",
        "## Strengths\n- " + "\n- ".join(r.strengths),
        "## Weaknesses\n- " + "\n- ".join(r.weaknesses),
        f"## Root Cause Analysis\n{r.root_cause_analysis}",
        "## Engineering Recommendations\n" + recs(r.recommendations),
        "## Planner Recommendations\n" + recs(r.planner_recommendations),
        f"## Expected Improvement\n{r.expected_improvement}",
    ]) + "\n"


# --- assembly --------------------------------------------------------------
def build_intelligence(container: Container, project_id: str, created_at: str) -> DatasetIntelligence | None:
    report = None
    from vds.gui.controller import BackendController  # local import avoids a cycle

    report = BackendController(container).cached_report(project_id)
    if report is None:
        return None

    analyst = LLMAnalyst(container.llm_client)
    ctx = AnalystContext(execution=report)
    result = analyst.analyze(ctx)  # the EXISTING Analyst, unchanged
    evidence = build_evidence(ctx)  # the Analyst's own measured evidence

    # Analyst integration (Phase 10): record validated knowledge; dedup-safe.
    try:
        cctx = build_dataset_context(project_id, container.settings, container.images)
        analyst.remember(ctx, result, container.memory, created_at,
                         resolution_mp=cctx.resolution_summary.get("megapixels_max", -1.0))
    except Exception:
        pass

    proj = container.projects.get(project_id)
    historical = _historical(container, report)
    mem_influence = (f"{historical.matches[0]['dataset']} "
                     f"(similarity {historical.matches[0]['similarity']})"
                     if historical.matches else "No historical match")
    kpis, overall = _health_kpis(container, report, mem_influence)
    readiness = _readiness(report, evidence)
    overall_rec, production = _overall_recommendation(readiness)
    q = report.quality

    summary = ExecutiveSummary(
        dataset=proj.name if proj else project_id,
        version=historical.runs or 1,
        size_mb=_storage_mb(container, project_id),
        image_count=report.imported,
        overall_health=overall,
        annotation_quality=round(max(0.0, q.approval_rate - q.rejection_rate), 4),
        verification_confidence=q.avg_confidence,
        production_readiness=production,
        overall_recommendation=overall_rec,
        historical_improvement=_historical_improvement(historical),
        analyst_summary=result.report.executive_summary,
        analyst_confidence=result.report.confidence,
        source=result.source,
    )
    return DatasetIntelligence(
        summary=summary, kpis=kpis, issues=_root_causes(evidence, result),
        recommendations=_rank_recommendations(result), historical=historical,
        readiness=readiness, analyst_report_markdown=analyst_report_markdown(result),
    )
