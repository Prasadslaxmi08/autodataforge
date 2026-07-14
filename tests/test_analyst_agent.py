"""AI Dataset Analyst tests: evidence, AI path, enforcement, and every fallback."""

from __future__ import annotations

import json
from pathlib import Path

from vds.agents.analyst_agent import (
    AnalystContext,
    LLMAnalyst,
    build_evidence,
)
from vds.agents.llm import LLMClient
from vds.agents.providers.echo import EchoProvider
from vds.config.settings import LLMSettings
from vds.container import Container
from vds.core.contracts import (
    BenchmarkResult,
    DatasetQualityReport,
    ErrorAnalysis,
    ErrorCategory,
    ExecutionReport,
    ExportReport,
    StageKPIs,
)

FAST = LLMSettings(model="m", retry_backoff_seconds=0.0, max_retries=2)


def make_execution(
    *, imported=50, detections=100, density=2.0, approval=1.0, review=0.0,
    reject=0.0, quarantined=0, duplicates=0, small=0, ips=25.0,
    stage_seconds=None,
) -> ExecutionReport:
    stage_seconds = stage_seconds or {"ingest": 0.3, "detection": 0.2,
                                       "segmentation": 0.1, "verification": 0.5, "export": 0.05}
    return ExecutionReport(
        project_id="proj", source="test", imported=imported,
        duplicates_skipped=duplicates, quarantined=quarantined, detections=detections,
        verified_approved=int(detections * approval), needs_review=int(detections * review),
        rejected=int(detections * reject),
        export=ExportReport(format="coco", images=imported, annotations=detections,
                            categories=["object"], output_path="/x", validated=True),
        benchmark=BenchmarkResult(
            project_id="proj", images_processed=imported, total_seconds=imported / ips,
            images_per_second=ips, avg_inference_ms=1.5, stage_seconds=stage_seconds,
            peak_ram_mb=55.0, cpu_percent=50.0, gpu_util_percent=0.0, created_at="2026-01-01T00:00:00Z",
        ),
        quality=DatasetQualityReport(
            project_id="proj", images=imported, detections=detections, masks=detections,
            approval_rate=approval, review_rate=review, rejection_rate=reject,
            invalid_annotations=0, duplicate_detections=0, empty_masks=0,
            annotation_density=density, avg_confidence=0.97,
            images_with_no_detection=0,
        ),
        errors=ErrorAnalysis(
            project_id="proj", total_annotations=detections,
            categories=[ErrorCategory(name="small_objects", count=small, description="d"),
                        ErrorCategory(name="crowded_scenes", count=0, description="d")],
            unmeasurable=["occlusion"],
        ),
    )


def _analyst(provider) -> LLMAnalyst:
    return LLMAnalyst(LLMClient(provider, FAST))


def _ctx(**over) -> AnalystContext:
    return AnalystContext(execution=make_execution(**over))


AI_REPORT = {
    "executive_summary": "Run reviewed.", "pipeline_performance": "ok",
    "dataset_characteristics": "ok", "detection_analysis": "ok",
    "segmentation_analysis": "ok", "verification_analysis": "ok",
    "resource_utilization": "ok", "strengths": ["export validated"],
    "weaknesses": ["uncalibrated confidence"], "root_cause_analysis": "scores uncalibrated",
    "recommendations": [{
        "action": "Add a learned verifier.", "target": "verification",
        "reason": "0% review.", "expected_impact": "meaningful review", "confidence": 0.8,
        "supporting_metrics": ["review_rate", "approval_rate"], "trade_offs": "cost",
    }],
    "planner_recommendations": [{
        "action": "Enable tiling.", "target": "planner", "reason": "small objects",
        "expected_impact": "recall", "confidence": 0.7,
        "supporting_metrics": ["annotation_density"], "trade_offs": "slower",
    }],
    "expected_improvement": "lower review", "confidence": 0.75,
    "next_actions": ["wire a verifier"],
}


# --- evidence (deterministic) ---
def test_evidence_facts_and_keys():
    ev = build_evidence(_ctx())
    keys = ev.keys
    assert {"throughput", "bottleneck", "approval_rate", "review_rate"} <= keys
    assert any("historical" in u for u in ev.unavailable)  # no baseline yet


def test_evidence_flags_uncalibrated_confidence():
    ev = build_evidence(_ctx(approval=1.0, review=0.0))
    assert "uncalibrated_confidence" in ev.keys


def test_evidence_dense_and_sparse():
    assert "dense_scenes" in build_evidence(_ctx(density=12)).keys
    assert "sparse_scenes" in build_evidence(_ctx(density=0.5, detections=10)).keys


def test_evidence_small_object_dominance():
    ev = build_evidence(_ctx(detections=100, small=80))
    assert "small_object_dominance" in ev.keys


# --- AI path + enforcement ---
def test_ai_report_adopted():
    result = _analyst(EchoProvider(reply=json.dumps(AI_REPORT))).analyze(_ctx())
    assert result.source == "ai"
    assert result.evidence_coverage == 1.0
    assert result.recommendation_count == 2


def test_evidence_enforcement_drops_uncited_recs():
    bad = json.loads(json.dumps(AI_REPORT))
    bad["recommendations"][0]["supporting_metrics"] = ["totally_made_up_key"]
    result = _analyst(EchoProvider(reply=json.dumps(bad))).analyze(_ctx())
    assert result.source == "ai"
    assert "Add a learned verifier." in result.unsupported_recommendations
    assert result.evidence_coverage < 1.0  # the hallucinated rec was dropped


# --- fallbacks ---
def test_fallback_on_invalid_json():
    result = _analyst(EchoProvider(reply="garbage")).analyze(_ctx())
    assert result.source == "deterministic"
    assert result.fallback_reason is not None
    assert result.recommendation_count >= 0


def test_fallback_on_schema_violation():
    bad = dict(AI_REPORT, confidence=5.0)  # > 1.0
    result = _analyst(EchoProvider(reply=json.dumps(bad))).analyze(_ctx())
    assert result.source == "deterministic"


def test_fallback_on_provider_failure():
    assert _analyst(EchoProvider(fail_times=99)).analyze(_ctx()).source == "deterministic"


def test_deterministic_report_is_evidence_backed():
    result = _analyst(EchoProvider(reply="garbage")).analyze(_ctx(approval=1.0, review=0.0))
    ev_keys = build_evidence(_ctx(approval=1.0, review=0.0)).keys
    for rec in result.report.recommendations:
        assert all(m in ev_keys for m in rec.supporting_metrics)


# --- scenarios ---
def test_empty_dataset_does_not_crash():
    result = _analyst(EchoProvider(reply="garbage")).analyze(
        AnalystContext(execution=make_execution(imported=0, detections=0, density=0.0))
    )
    assert result.source == "deterministic"
    assert result.report.executive_summary


def test_regression_detection():
    history = [StageKPIs(
        stage="deterministic", label="prev", created_at="2026-01-01T00:00:00Z",
        images_per_second=100.0, approval_rate=1.0, review_rate=0.0, rejection_rate=0.0,
        avg_confidence=0.97, annotation_density=2.0, peak_ram_mb=55.0,
        invalid_annotations=0, empty_masks=0,
    )]
    ctx = AnalystContext(execution=make_execution(ips=50.0), history=history)
    ev = build_evidence(ctx)
    assert "throughput_regression" in ev.keys


def test_no_history_states_no_baseline():
    ev = build_evidence(_ctx())
    assert any("baseline" in u for u in ev.unavailable)


# --- integration: real run through container (echo -> deterministic) ---
def test_container_analyst_on_real_run(container: Container, dataset_dir: Path, tmp_path: Path):
    execution = container.pipeline.run(str(dataset_dir), dest=str(tmp_path / "e"))
    result = container.analyst_agent.analyze(AnalystContext(execution=execution))
    assert result.source == "deterministic"  # echo default -> fallback
    assert result.report.verification_analysis
    assert result.recommendation_count >= 1  # uncalibrated-confidence rule fires
