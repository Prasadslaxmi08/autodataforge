"""Build an EngineeringMemory from measured pipeline outputs (Phase 10).

This is the anti-hallucination gate: every field is copied from a measured metric
(ExecutionReport / BenchmarkResult / quality + error analysis) or from an Analyst
recommendation that already passed evidence validation. Nothing is invented here —
if a fact wasn't measured, its field stays at its neutral default.
"""

from __future__ import annotations

import hashlib

from vds.core.contracts import ExecutionReport
from vds.memory.schema import (
    AnalystConclusions,
    BenchmarkSummary,
    DatasetFingerprint,
    EngineeringMemory,
    ExecutionMetrics,
    MemoryRecommendation,
    PlannerDecisions,
    VerificationOutcomes,
)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, round(x, 4)))


def _fingerprint(
    execution: ExecutionReport, resolution_mp: float, scene_type: str, environment: dict
) -> DatasetFingerprint:
    q, e = execution.quality, execution.errors
    dets = q.detections or 1
    small = next((c.count for c in e.categories if c.name == "small_objects"), 0)
    seen = execution.imported + execution.duplicates_skipped or 1
    return DatasetFingerprint(
        resolution_mp=resolution_mp,
        dataset_size=execution.imported,
        class_distribution={c: q.detections for c in _classes(execution)},
        scene_type=scene_type,
        environment=environment,
        scene_density=q.annotation_density,
        object_density=round(q.detections / resolution_mp, 4) if resolution_mp > 0 else -1.0,
        duplicate_ratio=round(execution.duplicates_skipped / seen, 4),
        small_object_ratio=round(small / dets, 4),
        image_quality=round(1.0 - execution.quarantined / (execution.imported or 1), 4),
        avg_confidence=q.avg_confidence,
    )


def _classes(execution: ExecutionReport) -> list[str]:
    # Single-class pipeline in this phase; the label lives on the plan when present.
    return ["object"]


def _planner_decisions(execution: ExecutionReport, planner_result) -> PlannerDecisions:
    plan = getattr(planner_result, "processing_plan", None) or planner_result
    ai = getattr(planner_result, "ai_plan", None)
    return PlannerDecisions(
        detector=getattr(plan, "detector", "unknown"),
        segmentation_enabled=bool(getattr(ai, "run_segmentation", True)) if ai else True,
        confidence_threshold=getattr(plan, "confidence_threshold", 0.0),
        batch_size=getattr(plan, "batch_size", 0),
        worker_count=getattr(ai, "worker_count", 1) if ai else 1,
        tiling=getattr(ai, "tiling_required", False) if ai else False,
        export_strategy=getattr(plan, "export_format", execution.export.format),
    )


def _execution_metrics(execution: ExecutionReport) -> ExecutionMetrics:
    b, q = execution.benchmark, execution.quality
    return ExecutionMetrics(
        throughput_ips=b.images_per_second,
        runtime_seconds=b.total_seconds,
        gpu_util_percent=b.gpu_util_percent,
        cpu_percent=b.cpu_percent,
        peak_ram_mb=b.peak_ram_mb,
        review_rate=q.review_rate,
        approval_rate=q.approval_rate,
        rejection_rate=q.rejection_rate,
        invalid_annotations=q.invalid_annotations,
        empty_masks=q.empty_masks,
        export_format=execution.export.format,
        export_validated=execution.export.validated,
    )


def _verification_outcomes(execution: ExecutionReport, overrides: VerificationOutcomes | None) -> VerificationOutcomes:
    if overrides is not None:
        return overrides
    q, e = execution.quality, execution.errors
    failures = {c.name: c.count for c in e.categories if c.count > 0}
    return VerificationOutcomes(
        common_semantic_failures=failures,
        bbox_issues=q.invalid_annotations,
        segmentation_issues=q.empty_masks,
        false_positives=execution.rejected,
        false_negatives=q.images_with_no_detection,
    )


def _analyst_knowledge(analyst_result) -> tuple[AnalystConclusions, list[MemoryRecommendation], str, float]:
    """Only *validated* recommendations become reusable knowledge. The Analyst has
    already dropped recommendations that cite non-existent evidence, so what remains
    is evidence-backed; we keep only those that still carry supporting_metrics."""
    if analyst_result is None:
        return AnalystConclusions(), [], "provisional", 0.5
    report = analyst_result.report
    recs: list[MemoryRecommendation] = []
    for r in report.recommendations + report.planner_recommendations:
        if not r.supporting_metrics:  # unvalidated -> never remembered
            continue
        recs.append(MemoryRecommendation(
            action=r.action, target=r.target, reason=r.reason,
            expected_impact=r.expected_impact, confidence=r.confidence,
            supporting_metrics=r.supporting_metrics,
        ))
    conclusions = AnalystConclusions(
        root_causes=[report.root_cause_analysis] if report.root_cause_analysis else [],
        bottlenecks=list(report.weaknesses),
        improvement_opportunities=list(report.next_actions),
        confidence=report.confidence,
    )
    # Validated only when the AI produced it AND its citations mostly survived.
    validated = analyst_result.source == "ai" and analyst_result.evidence_coverage >= 0.75
    return conclusions, recs, ("validated" if validated else "provisional"), report.confidence


def build_memory(
    execution: ExecutionReport,
    created_at: str,
    *,
    planner_result=None,
    analyst_result=None,
    resolution_mp: float = -1.0,
    scene_type: str = "unknown",
    environment: dict | None = None,
    source: str = "pipeline",
    verification: VerificationOutcomes | None = None,
) -> EngineeringMemory:
    fp = _fingerprint(execution, resolution_mp, scene_type, environment or {})
    conclusions, recs, status, ana_conf = _analyst_knowledge(analyst_result)
    q = execution.quality
    ident = hashlib.sha256(f"{fp.hash()}|{execution.project_id}|{created_at}".encode()).hexdigest()[:12]
    return EngineeringMemory(
        id=f"mem_{ident}",
        created_at=created_at,
        project_id=execution.project_id,
        source=source,
        dataset_fingerprint=fp,
        planner_decisions=_planner_decisions(execution, planner_result) if planner_result
        else PlannerDecisions(detector="deterministic", segmentation_enabled=True,
                              confidence_threshold=0.3, batch_size=execution.benchmark.num_batches or 1,
                              export_strategy=execution.export.format),
        execution_metrics=_execution_metrics(execution),
        analyst_conclusions=conclusions,
        verification_outcomes=_verification_outcomes(execution, verification),
        benchmark_summary=BenchmarkSummary(
            throughput_ips=execution.benchmark.images_per_second,
            review_rate=q.review_rate, approval_rate=q.approval_rate,
            avg_confidence=q.avg_confidence,
            quality_score=_clamp01(q.approval_rate - q.rejection_rate),
        ),
        engineering_recommendations=recs,
        validation_status=status,
        confidence=ana_conf,
    )
