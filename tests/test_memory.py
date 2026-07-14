"""Tests for Engineering Memory (Phase 10): creation, updates, versioning,
similarity, planner retrieval, analyst updates, historical comparison, regression
safety, corrupted memory, duplicate memory."""

from __future__ import annotations

import json

from vds.core.contracts import (
    BenchmarkResult,
    DatasetQualityReport,
    ErrorAnalysis,
    ErrorCategory,
    ExecutionReport,
    ExportReport,
)
from vds.memory import (
    DatasetFingerprint,
    EngineeringMemoryService,
    MemoryStore,
    build_memory,
)


# --- fixtures --------------------------------------------------------------
def _execution(project="p1", *, imported=30, review=0.2, approval=0.7, reject=0.1,
               ips=12.0, dets=60, small=5, dup_skipped=2, conf=0.8) -> ExecutionReport:
    quality = DatasetQualityReport(
        project_id=project, images=imported, detections=dets, masks=dets,
        approval_rate=approval, review_rate=review, rejection_rate=reject,
        invalid_annotations=1, duplicate_detections=3, empty_masks=2,
        annotation_density=round(dets / imported, 4), avg_confidence=conf,
        images_with_no_detection=1,
    )
    errors = ErrorAnalysis(
        project_id=project, total_annotations=dets,
        categories=[ErrorCategory(name="small_objects", count=small, description="small"),
                    ErrorCategory(name="low_confidence", count=4, description="lowconf")],
    )
    benchmark = BenchmarkResult(
        project_id=project, images_processed=imported, total_seconds=round(imported / ips, 3),
        images_per_second=ips, avg_inference_ms=5.0, stage_seconds={"detector": 1.0},
        num_batches=2, peak_ram_mb=500.0, cpu_percent=40.0, gpu_util_percent=None,
        created_at="2026-07-10T10:00:00",
    )
    export = ExportReport(format="coco", images=imported, annotations=dets,
                          categories=["object"], output_path="x", validated=True)
    return ExecutionReport(
        project_id=project, source="s", imported=imported, duplicates_skipped=dup_skipped,
        quarantined=0, detections=dets, verified_approved=int(dets * approval),
        needs_review=int(dets * review), rejected=int(dets * reject),
        export=export, benchmark=benchmark, quality=quality, errors=errors,
    )


def _service(tmp_path) -> EngineeringMemoryService:
    return EngineeringMemoryService(tmp_path / "mem.json", min_similarity=0.4)


# --- creation --------------------------------------------------------------
def test_memory_creation_from_measured_output(tmp_path):
    svc = _service(tmp_path)
    m = svc.record_execution(_execution(), "2026-07-10T10:00:00", resolution_mp=2.0, scene_type="aerial")
    assert m.id.startswith("mem_")
    assert m.dataset_fingerprint.dataset_size == 30
    assert m.execution_metrics.throughput_ips == 12.0
    assert m.version == 1
    assert svc.all()[0].id == m.id


def test_only_validated_recommendations_are_stored(tmp_path):
    """Analyst recs without supporting evidence are never remembered."""
    from vds.agents.analyst_agent import AnalystReport, AnalystResult, Recommendation

    report = AnalystReport(
        executive_summary="", pipeline_performance="", dataset_characteristics="",
        detection_analysis="", segmentation_analysis="", verification_analysis="",
        resource_utilization="", root_cause_analysis="geometric heuristic",
        expected_improvement="", confidence=0.8,
        recommendations=[
            Recommendation(action="keep me", target="pipeline", reason="r", expected_impact="i",
                           confidence=0.9, supporting_metrics=["bottleneck"]),
            Recommendation(action="drop me", target="pipeline", reason="r", expected_impact="i",
                           confidence=0.5, supporting_metrics=[]),  # unvalidated
        ],
    )
    result = AnalystResult(report=report, source="ai", evidence_coverage=1.0,
                           recommendation_count=2)
    m = build_memory(_execution(), "2026-07-10T10:00:00", analyst_result=result)
    actions = [r.action for r in m.engineering_recommendations]
    assert "keep me" in actions and "drop me" not in actions
    assert m.validation_status == "validated"


# --- versioning + no-overwrite ---------------------------------------------
def test_versioning_keeps_full_history(tmp_path):
    svc = _service(tmp_path)
    svc.record_execution(_execution(review=0.3), "2026-07-10T10:00:00", resolution_mp=2.0, scene_type="aerial")
    # Same dataset family (same fingerprint), a later run -> version 2, both retained.
    svc.record_execution(_execution(review=0.1), "2026-07-11T10:00:00", resolution_mp=2.0, scene_type="aerial")
    fam = MemoryStore(tmp_path / "mem.json").all()
    assert len(fam) == 2
    assert sorted(m.version for m in fam) == [1, 2]


# --- duplicate memory ------------------------------------------------------
def test_duplicate_memory_is_suppressed(tmp_path):
    svc = _service(tmp_path)
    a = svc.record_execution(_execution(), "2026-07-10T10:00:00", resolution_mp=2.0, scene_type="aerial")
    b = svc.record_execution(_execution(), "2026-07-10T10:00:00", resolution_mp=2.0, scene_type="aerial")
    assert a.id == b.id  # identical content -> not stored twice
    assert len(svc.all()) == 1


# --- similarity + explanation ----------------------------------------------
def test_similarity_search_and_explanation(tmp_path):
    svc = _service(tmp_path)
    svc.record_execution(_execution("aerial1"), "2026-07-10T10:00:00", resolution_mp=2.0, scene_type="aerial")
    svc.record_execution(_execution("street1", imported=500, ips=3.0),
                         "2026-07-10T11:00:00", resolution_mp=12.0, scene_type="street")
    query = DatasetFingerprint(resolution_mp=2.1, dataset_size=32, scene_type="aerial")
    matches = svc.similar(query, top_k=2)
    assert matches, "expected at least one match"
    assert matches[0].memory.project_id == "aerial1"  # closer than the street dataset
    assert "resolution_mp" in matches[0].explain() or matches[0].score > 0
    assert matches[0].score > matches[-1].score or len(matches) == 1


def test_query_uses_only_known_features(tmp_path):
    # A pre-run query with post-run features unknown still matches on resolution/scale.
    svc = _service(tmp_path)
    svc.record_execution(_execution(), "2026-07-10T10:00:00", resolution_mp=2.0, scene_type="aerial")
    query = DatasetFingerprint(resolution_mp=2.0, dataset_size=30, scene_type="aerial")
    matches = svc.similar(query)
    assert matches and matches[0].score >= 0.9


# --- planner retrieval -----------------------------------------------------
def test_planner_recall_reports_experience_or_absence(tmp_path):
    svc = _service(tmp_path)
    empty = svc.recall(DatasetFingerprint(resolution_mp=2.0, dataset_size=30, scene_type="aerial"))
    assert not empty.has_experience
    assert "No similar" in empty.note

    svc.record_execution(_execution(approval=0.9, reject=0.02), "2026-07-10T10:00:00",
                         resolution_mp=2.0, scene_type="aerial")
    guidance = svc.recall(DatasetFingerprint(resolution_mp=2.0, dataset_size=30, scene_type="aerial"))
    assert guidance.has_experience
    assert "similar past dataset" in guidance.note
    assert guidance.render() != guidance.note  # includes match detail


def test_planner_agent_uses_memory(tmp_path):
    from vds.agents.llm import LLMClient
    from vds.agents.planner import ExecutionPlanner
    from vds.agents.planner_agent import DatasetContext, LLMPlanner
    from vds.agents.providers.echo import EchoProvider
    from vds.config.settings import LLMSettings, Settings

    svc = _service(tmp_path)
    svc.record_execution(_execution(), "2026-07-10T10:00:00", resolution_mp=2.0, scene_type="aerial")
    planner = LLMPlanner(LLMClient(EchoProvider(LLMSettings()), LLMSettings()),
                         ExecutionPlanner(Settings()), svc)
    ctx = DatasetContext(
        project_id="new", image_count=30, resolution_summary={"megapixels_max": 2.0},
        file_types=["png"], classes=["object"], available_detectors=["builtin"],
        available_segmenters=["builtin"], gpu_device="cpu", vram_budget_mb=8192,
        export_format="coco", review_budget_hours=8.0, user_preferences={"scene_type": "aerial"},
    )
    result = planner.plan(ctx)  # Echo -> deterministic fallback, but memory still consulted
    assert result.memory_used is True
    assert result.memory_matches
    assert "similar past dataset" in result.memory_note


# --- analyst updates -------------------------------------------------------
def test_analyst_remember_stores_validated_knowledge(tmp_path):
    from vds.agents.analyst_agent import (
        AnalystContext,
        AnalystReport,
        AnalystResult,
        LLMAnalyst,
        Recommendation,
    )
    from vds.agents.llm import LLMClient
    from vds.agents.providers.echo import EchoProvider
    from vds.config.settings import LLMSettings

    svc = _service(tmp_path)
    execution = _execution()
    report = AnalystReport(
        executive_summary="", pipeline_performance="", dataset_characteristics="",
        detection_analysis="", segmentation_analysis="", verification_analysis="",
        resource_utilization="", root_cause_analysis="rc", expected_improvement="",
        confidence=0.8,
        recommendations=[Recommendation(action="tile it", target="planner", reason="r",
                                        expected_impact="recall", confidence=0.8,
                                        supporting_metrics=["small_object_dominance"])],
    )
    result = AnalystResult(report=report, source="ai", evidence_coverage=1.0, recommendation_count=1)
    analyst = LLMAnalyst(LLMClient(EchoProvider(LLMSettings()), LLMSettings()))
    m = analyst.remember(AnalystContext(execution=execution), result, svc,
                         "2026-07-10T10:00:00", resolution_mp=2.0, scene_type="aerial")
    assert m.engineering_recommendations[0].action == "tile it"
    assert svc.all()[0].id == m.id


# --- historical comparison / trends ----------------------------------------
def test_historical_trend_and_engineering_reports(tmp_path):
    svc = _service(tmp_path)
    svc.record_execution(_execution("r1", review=0.4, approval=0.5, reject=0.2, ips=8.0),
                         "2026-07-10T10:00:00", resolution_mp=2.0, scene_type="aerial")
    svc.record_execution(_execution("r2", review=0.1, approval=0.85, reject=0.05, ips=15.0),
                         "2026-07-12T10:00:00", resolution_mp=3.0, scene_type="street")
    ev = __import__("vds.memory.trends", fromlist=["TrendAnalyzer"]).TrendAnalyzer().evolution(svc.all())
    assert ev["review_rate"].improved is True  # 0.4 -> 0.1 is an improvement (lower better)
    assert ev["throughput"].improved is True
    report = svc.engineering_report()
    assert "Most Successful Planner Strategies" in report
    assert svc.trend_report().strip()


# --- corrupted memory ------------------------------------------------------
def test_corrupted_memory_file_is_quarantined(tmp_path):
    path = tmp_path / "mem.json"
    path.write_text("{ this is not valid json", encoding="utf-8")
    store = MemoryStore(path)
    assert store.all() == []  # graceful, not a crash
    assert path.with_suffix(".corrupt").exists()
    # store still usable afterwards
    svc = EngineeringMemoryService(path)
    m = svc.record_execution(_execution(), "2026-07-10T10:00:00", resolution_mp=2.0)
    assert svc.all()[0].id == m.id


def test_invalid_row_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "mem.json"
    good = build_memory(_execution(), "2026-07-10T10:00:00", resolution_mp=2.0)
    path.write_text(json.dumps([good.model_dump(), {"garbage": True}]), encoding="utf-8")
    assert len(MemoryStore(path).all()) == 1


# --- regression safety -----------------------------------------------------
def test_regression_safety_missing_file_is_empty(tmp_path):
    assert MemoryStore(tmp_path / "does_not_exist.json").all() == []
    svc = _service(tmp_path)
    assert svc.recall(DatasetFingerprint(resolution_mp=1.0)).note.startswith("No similar")
    assert svc.trend_report().endswith("\n")


def test_no_raw_images_ever_stored(tmp_path):
    svc = _service(tmp_path)
    m = svc.record_execution(_execution(), "2026-07-10T10:00:00", resolution_mp=2.0)
    blob = json.dumps(m.model_dump())
    for banned in ("image_bytes", "png", "jpeg", "rle", "base64"):
        assert banned not in blob.lower()
