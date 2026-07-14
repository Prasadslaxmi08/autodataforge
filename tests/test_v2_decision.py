"""V2-23 DecisionAgent — decision generation, plan enrichment, confidence/tradeoffs,
overrides, serialization, GUI surface. Optimization only; never executes.
"""

from __future__ import annotations

from vds.v2 import (
    DatasetMetadata,
    DecisionAgent,
    DecisionArea,
    DecisionReport,
    FrameStrategy,
    PlannerAgent,
    ReviewLevel,
    decision_view,
    new_goal,
)


def _plan(text="Create a vehicle detection dataset from images", **params):
    return PlannerAgent().create_plan(new_goal(text, source="imgs/", **params))


# --- decision generation ----------------------------------------------
def test_decisions_cover_all_core_areas():
    enriched, report = DecisionAgent().decide(_plan(), DatasetMetadata(image_count=800))
    areas = {d.area for d in report.decisions}
    for a in (DecisionArea.DETECTION_CONFIDENCE, DecisionArea.IOU_THRESHOLD, DecisionArea.SEGMENTATION,
              DecisionArea.EXPORT_FORMAT, DecisionArea.REVIEW_LEVEL, DecisionArea.BATCH_SIZE,
              DecisionArea.COMPUTE, DecisionArea.DUPLICATE_REMOVAL, DecisionArea.EXPECTED_RUNTIME,
              DecisionArea.ANNOTATION_COUNT):
        assert a in areas
    # every decision is fully explained
    for d in report.decisions:
        assert d.reason and 0.0 <= d.confidence <= 1.0 and d.value


def test_every_recommendation_has_confidence_and_tradeoffs():
    _, report = DecisionAgent().decide(_plan(), DatasetMetadata(image_count=100))
    conf = report.get(DecisionArea.DETECTION_CONFIDENCE)
    assert conf.confidence > 0 and conf.alternative and conf.impact and conf.tradeoffs
    assert 0.0 < report.overall_confidence <= 1.0


# --- metadata-driven refinement (not planner duplication) -------------
def test_thermal_metadata_lowers_confidence():
    _, report = DecisionAgent().decide(_plan(), DatasetMetadata(image_count=100, file_types=["png", "thermal"]))
    d = report.get(DecisionArea.DETECTION_CONFIDENCE)
    assert d.value == "0.20" and "thermal" in d.reason.lower() and d.confidence > 0.9


def test_high_false_positive_history_raises_confidence():
    _, report = DecisionAgent().decide(
        _plan(), DatasetMetadata(image_count=100, historical_stats={"false_positive_rate": 0.5}))
    assert report.get(DecisionArea.DETECTION_CONFIDENCE).value == "0.45"


def test_large_count_recommends_gpu_and_dedup():
    _, report = DecisionAgent().decide(_plan(), DatasetMetadata(image_count=6000))
    assert report.get(DecisionArea.COMPUTE).value == "GPU"
    assert report.get(DecisionArea.DUPLICATE_REMOVAL).value == "true"
    assert any("large dataset" in w.lower() for w in report.warnings)


def test_video_frame_sampling_and_count_estimate():
    plan = PlannerAgent().create_plan(
        new_goal("detect cars from drone video", source="d.mp4"))
    enriched, report = DecisionAgent().decide(
        plan, DatasetMetadata(video_duration_seconds=20, fps=30))  # short => dense sampling
    assert report.get(DecisionArea.FRAME_SAMPLING).value == FrameStrategy.EVERY_2.value
    assert enriched.estimated_dataset_size == int(20 * 30 / 2)
    assert report.expected_annotation_count == enriched.estimated_dataset_size * 3


def test_previous_exports_drive_export_format():
    _, report = DecisionAgent().decide(
        _plan(), DatasetMetadata(image_count=100, previous_exports=["COCO", "YOLO"]))
    assert report.get(DecisionArea.EXPORT_FORMAT).value == "coco,yolo"


# --- plan enrichment (deep copy; planner output untouched) -------------
def test_enrichment_writes_params_into_copy_without_touching_original():
    plan = _plan()
    original_conf = plan.recommended_confidence
    enriched, _ = DecisionAgent().decide(plan, DatasetMetadata(image_count=100, file_types=["thermal"]))
    assert enriched.recommended_confidence == 0.20
    assert plan.recommended_confidence == original_conf  # planner plan untouched
    det = enriched.get("run_detection")
    assert det.arguments["confidence"] == 0.20 and "device" in det.arguments and "batch_size" in det.arguments


def test_review_level_high_for_thermal():
    _, report = DecisionAgent().decide(_plan(), DatasetMetadata(image_count=100, file_types=["thermal"]))
    assert report.recommended_review == ReviewLevel.HIGH.value


# --- overrides (accept / reject / override) ---------------------------
def test_apply_overrides():
    agent = DecisionAgent()
    enriched, report = agent.decide(_plan(), DatasetMetadata(image_count=100))
    p2, r2 = agent.apply_overrides(enriched, report, {
        DecisionArea.DETECTION_CONFIDENCE: 0.5, DecisionArea.REVIEW_LEVEL: "low"})
    assert p2.recommended_confidence == 0.5
    assert p2.get("run_detection").arguments["confidence"] == 0.5
    assert p2.estimated_review == ReviewLevel.LOW
    d = r2.get(DecisionArea.DETECTION_CONFIDENCE)
    assert d.value == "0.5" and d.reason == "User override." and d.confidence == 1.0


# --- serialization -----------------------------------------------------
def test_report_roundtrip():
    _, report = DecisionAgent().decide(_plan(), DatasetMetadata(image_count=100))
    data = report.model_dump_json()
    restored = DecisionReport.model_validate_json(data)
    assert restored.plan_id == report.plan_id
    assert len(restored.decisions) == len(report.decisions)
    assert restored.decisions[0].tradeoffs == report.decisions[0].tradeoffs


# --- GUI surface + integration ----------------------------------------
def test_decision_view_surface():
    _, report = DecisionAgent().decide(_plan(), DatasetMetadata(image_count=100))
    v = decision_view(report)
    assert v["recommendations"] and v["reasoning"]
    assert "estimated_runtime_seconds" in v and "expected_annotation_count" in v
    assert v["review_level"] and 0 < v["overall_confidence"] <= 1


def test_dataset_engineer_optimize_plan_then_execute():
    class _FakeController:
        def __getattr__(self, name):
            return lambda *a, **k: None

    from vds.v2 import DatasetEngineerAgent, PlanStatus
    from vds.v2.state import SessionStatus

    eng = DatasetEngineerAgent(_FakeController())  # type: ignore[arg-type]
    plan = eng.generate_plan(new_goal("detect cars from images", source="imgs/"))
    enriched, report = eng.optimize_plan(plan, {"image_count": 6000, "file_types": ["thermal"]})
    assert enriched.recommended_confidence == 0.20 and report.overall_confidence > 0
    # the enriched plan still executes through the ExecutionAgent unchanged
    enriched.status = PlanStatus.APPROVED
    ctx = eng.execute_plan(enriched, {"source": "imgs/", "name": "cars", "project_id": "cars"})
    assert ctx.status in (SessionStatus.AWAITING_APPROVAL, SessionStatus.COMPLETED)
