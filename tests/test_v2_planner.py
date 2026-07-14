"""V2-21 PlannerAgent — goal parsing, plan generation, recommendations, validation,
session support, serialization. Deterministic, no LLM, nothing executes.
"""

from __future__ import annotations

import pytest

from vds.v2 import (
    ExecutionPlan,
    FrameStrategy,
    GoalParser,
    PlanContext,
    PlannerAgent,
    PlanSessionStore,
    PlanStatus,
    ReviewLevel,
    TaskType,
    new_goal,
    plan_view,
)
from vds.v2.planner import PlanStep


# --- goal parsing ------------------------------------------------------
def test_goal_parser_classifies():
    p = GoalParser()
    assert p.parse(new_goal("Create a vehicle detection dataset from this highway video",
                            source="road.mp4")).task_type == TaskType.DETECTION
    seg = p.parse(new_goal("Prepare a segmentation dataset"))
    assert seg.task_type == TaskType.SEGMENTATION
    thermal = p.parse(new_goal("Create a thermal person dataset"))
    assert thermal.thermal and "person" in thermal.target_classes
    assert p.parse(new_goal("Export this dataset to COCO")).task_type == TaskType.EXPORT
    assert p.parse(new_goal("Improve this existing dataset")).modality == "existing"


def test_video_modality_from_source_extension():
    g = new_goal("build a dataset", source="clip.mov")
    assert GoalParser().parse(g).modality == "video"


# --- plan generation ---------------------------------------------------
def test_plan_generation_video_detection():
    plan = PlannerAgent().create_plan(
        new_goal("Create a vehicle detection dataset from this highway video", source="hwy.mp4"),
        context=PlanContext(video_duration_seconds=120, fps=30, expected_density="high"),
    )
    ids = [s.id for s in plan.steps]
    assert ids[0] == "analyse_inputs"
    # import_video runs the self-contained pipeline (extract -> detect -> segment ->
    # verify -> export), so a video plan has no standalone extract/detect/segment steps.
    assert "import_video" in ids
    assert "extract_frames" not in ids and "run_detection" not in ids
    assert "run_segmentation" not in ids  # detection-only
    assert plan.frame_strategy == FrameStrategy.EVERY_2  # dense scene
    assert plan.estimated_dataset_size == int(120 * 30 / 2)
    assert plan.approvals_required == ["Manual Review"]
    assert plan.reasoning and plan.recommendations and plan.alternatives


def test_segmentation_adds_step_and_recommends_seg_model():
    plan = PlannerAgent().create_plan(new_goal("Prepare a segmentation dataset", source="imgs/"))
    assert "run_segmentation" in [s.id for s in plan.steps]
    assert plan.recommended_segmentation and "seg" in plan.recommended_model.lower()


def test_export_only_plan_has_no_detection():
    plan = PlannerAgent().create_plan(new_goal("Export this dataset to yolo", dataset="d1"))
    ids = [s.id for s in plan.steps]
    assert "run_detection" not in ids and "export_dataset" in ids
    assert plan.steps[[s.id for s in plan.steps].index("export_dataset")].arguments["format"] == "yolo"


def test_thermal_lowers_confidence_and_warns():
    plan = PlannerAgent().create_plan(new_goal("Create a thermal person dataset", source="ir/"))
    assert plan.recommended_confidence == 0.20
    assert any("thermal" in w.lower() for w in plan.warnings)


def test_missing_source_warns():
    plan = PlannerAgent().create_plan(new_goal("Create a vehicle detection dataset from a video"))
    assert any("no input source" in w.lower() for w in plan.warnings)


# --- recommendations ---------------------------------------------------
def test_large_dataset_recommends_dedup():
    plan = PlannerAgent().create_plan(
        new_goal("detect cars in images", source="imgs/"),
        context=PlanContext(image_count=5000),
    )
    assert any(r.topic == "dedup" and r.value == "enabled" for r in plan.recommendations)


def test_high_res_small_objects_recommend_larger_model():
    plan = PlannerAgent().create_plan(
        new_goal("detect cars", source="imgs/"),
        context=PlanContext(resolution="high", small_objects=True),
    )
    assert plan.recommended_model == "YOLO11m"
    assert any(a.topic == "model" and a.alternative == "YOLO11s" for a in plan.alternatives)


def test_review_estimation_high_for_segmentation():
    plan = PlannerAgent().create_plan(new_goal("Prepare a segmentation dataset", source="imgs/"))
    assert plan.estimated_review in (ReviewLevel.MEDIUM, ReviewLevel.HIGH)


# --- validation --------------------------------------------------------
def test_validate_rejects_export_before_import():
    agent = PlannerAgent()
    plan = ExecutionPlan(
        goal_id="g",
        frame_strategy=FrameStrategy.NONE,
        steps=[
            PlanStep(id="export_dataset", name="Export", agent="ExportAgent", task="export_dataset"),
            PlanStep(id="import_images", name="Import", agent="ImportAgent", task="import_images",
                     depends_on=["export_dataset"]),
        ],
    )
    errors = agent.validate(plan)
    assert any("Export step precedes Import" in e for e in errors)


def test_validate_rejects_frame_strategy_on_images():
    plan = ExecutionPlan(
        goal_id="g",
        frame_strategy=FrameStrategy.EVERY_5,  # but no extract step
        steps=[PlanStep(id="import_images", name="Import", agent="ImportAgent", task="import_images")],
    )
    assert any("non-video" in e for e in PlannerAgent().validate(plan))


def test_modify_toggle_segmentation_and_revalidate():
    agent = PlannerAgent()
    # An image-detection plan has a standalone run_detection step; toggling segmentation
    # inserts run_segmentation right after it. (Video plans are self-contained via
    # import_video, so segmentation there is a pipeline parameter, not a separate step.)
    plan = agent.create_plan(new_goal("detect cars in these images", source="imgs/"),
                             context=PlanContext())
    assert "run_segmentation" not in [s.id for s in plan.steps]
    edited = agent.modify(plan, segmentation=True, confidence=0.4)
    assert "run_segmentation" in [s.id for s in edited.steps]
    assert edited.recommended_confidence == 0.4
    assert edited.status == PlanStatus.DRAFT
    # linear deps stay valid after the insert
    assert agent.validate(edited) == []


def test_modify_bad_export_format_raises():
    agent = PlannerAgent()
    plan = agent.create_plan(new_goal("detect cars", source="imgs/"))
    with pytest.raises(ValueError):  # FrameStrategy("bogus") inside modify
        agent.modify(plan, frame_strategy="bogus")


# --- session + serialization ------------------------------------------
def test_session_lifecycle_and_roundtrip():
    store = PlanSessionStore()
    plan = PlannerAgent().create_plan(new_goal("detect cars from video", source="v.mp4"))
    store.create(plan)
    assert plan.id in store.list()
    assert store.approve(plan.id).status == PlanStatus.APPROVED
    assert store.reject(plan.id).status == PlanStatus.REJECTED

    data = store.export(plan.id)
    restored = PlanSessionStore.restore(data)
    assert isinstance(restored, ExecutionPlan)
    assert restored.id == plan.id
    assert [s.id for s in restored.steps] == [s.id for s in plan.steps]
    assert restored.recommendations[0].reason  # nested models survive the round-trip


# --- GUI surface -------------------------------------------------------
def test_plan_view_surface():
    plan = PlannerAgent().create_plan(new_goal("Create a vehicle detection dataset from video",
                                               source="v.mp4"))
    v = plan_view(plan)
    assert v["task_type"] == "detection"
    assert v["timeline"] and v["recommended"]["model"]
    assert "approvals_required" in v and "warnings" in v


def test_dataset_engineer_generate_plan_is_plan_only():
    # The master agent's GUI entry returns a plan without running anything.
    class _FakeController:
        def __getattr__(self, name):
            return lambda *a, **k: None

    from vds.v2 import DatasetEngineerAgent

    eng = DatasetEngineerAgent(_FakeController())  # type: ignore[arg-type]
    plan = eng.generate_plan(new_goal("detect cars from video", source="v.mp4"),
                             context={"video_duration_seconds": 60, "fps": 30})
    assert isinstance(plan, ExecutionPlan) and plan.steps
