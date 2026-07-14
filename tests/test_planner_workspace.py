"""Planner Workspace tests (Phase 12): Planner integration, Engineering Memory
integration, interactive updates, plan comparison, backend communication, thread
safety, and Planner execution through the page. Headless (offscreen, set in conftest)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from vds.agents.planner_agent import build_dataset_context  # noqa: E402
from vds.core.contracts import (  # noqa: E402
    BenchmarkResult,
    DatasetQualityReport,
    ErrorAnalysis,
    ErrorCategory,
    ExecutionReport,
    ExportReport,
)
from vds.gui.controller import BackendController  # noqa: E402
from vds.gui.notifications import NotificationSystem  # noqa: E402
from vds.gui.pages.planner import PlannerPage  # noqa: E402
from vds.gui.planner_view import PlanControls, diff_plans  # noqa: E402
from vds.gui.threads import ThreadManager  # noqa: E402


def _imported(ctrl: BackendController, dataset_dir) -> str:
    ctrl.import_dataset(str(dataset_dir), "planme")
    return ctrl.list_datasets()[0].project_id


def _seed_matching_memory(container, project_id: str) -> None:
    """Record a memory whose fingerprint matches the Planner's pre-run query for
    this dataset (same resolution + size), so recall returns it."""
    ctx = build_dataset_context(project_id, container.settings, container.images)
    mp = ctx.resolution_summary.get("megapixels_max", 0.0)
    n = ctx.image_count
    quality = DatasetQualityReport(
        project_id="prior", images=n, detections=n * 2, masks=n * 2,
        approval_rate=0.8, review_rate=0.1, rejection_rate=0.1, invalid_annotations=0,
        duplicate_detections=0, empty_masks=0, annotation_density=2.0, avg_confidence=0.8,
        images_with_no_detection=0,
    )
    errors = ErrorAnalysis(project_id="prior", total_annotations=n * 2, categories=[
        ErrorCategory(name="small_objects", count=1, description="s")])
    bench = BenchmarkResult(project_id="prior", images_processed=n, total_seconds=1.0,
                            images_per_second=10.0, avg_inference_ms=5.0, stage_seconds={"d": 1.0},
                            num_batches=1, peak_ram_mb=100.0, created_at="2026-07-01T00:00:00")
    export = ExportReport(format="coco", images=n, annotations=n * 2, categories=["object"],
                          output_path="x", validated=True)
    execution = ExecutionReport(
        project_id="prior", source="s", imported=n, duplicates_skipped=0, quarantined=0,
        detections=n * 2, verified_approved=n, needs_review=0, rejected=0,
        export=export, benchmark=bench, quality=quality, errors=errors)
    container.memory.record_execution(execution, "2026-07-01T00:00:00", resolution_mp=mp)


# --- Planner integration ---------------------------------------------------
def test_planner_integration(container, dataset_dir):
    ctrl = BackendController(container)
    pid = _imported(ctrl, dataset_dir)
    view = ctrl.plan_dataset(pid)
    assert view.source in ("ai", "deterministic")
    assert len(view.decisions) == 11
    assert view.profile.image_count == 3
    assert view.effective_controls.detector  # a detector was chosen
    # every decision carries the six required fields
    d = view.decisions[0]
    assert d.reason and d.expected_impact and d.validation


# --- Engineering Memory integration ----------------------------------------
def test_memory_integration_absent(container, dataset_dir):
    ctrl = BackendController(container)
    pid = _imported(ctrl, dataset_dir)
    view = ctrl.plan_dataset(pid)
    assert view.memory_used is False
    assert view.memory_matches == []
    assert "No similar" in view.memory_note


def test_memory_integration_present(container, dataset_dir):
    ctrl = BackendController(container)
    pid = _imported(ctrl, dataset_dir)
    _seed_matching_memory(container, pid)
    view = ctrl.plan_dataset(pid)
    assert view.memory_matches, "expected a recalled similar dataset"
    assert view.memory_used is True
    assert view.memory_matches[0].similarity > 0.5


# --- interactive updates + comparison --------------------------------------
def test_interactive_update_changes_plan(container, dataset_dir):
    ctrl = BackendController(container)
    pid = _imported(ctrl, dataset_dir)
    modified = ctrl.plan_dataset(pid, PlanControls(
        detector="yolo", confidence_threshold=0.6, export_format="yolo"))
    assert modified.effective_controls.detector == "yolo"
    assert modified.effective_controls.confidence_threshold == 0.6
    assert modified.effective_controls.export_format == "yolo"


def test_plan_comparison_highlights_differences(container, dataset_dir):
    ctrl = BackendController(container)
    pid = _imported(ctrl, dataset_dir)
    original = ctrl.plan_dataset(pid)
    modified = ctrl.plan_dataset(pid, PlanControls(detector="yolo", confidence_threshold=0.6))
    rows = diff_plans(original, modified)
    fields = {r.field for r in rows}
    assert "Detector" in fields and "Confidence Threshold" in fields
    # identical plans -> no differences
    assert diff_plans(original, original) == []


# --- backend communication -------------------------------------------------
def test_backend_communication_options(container):
    ctrl = BackendController(container)
    assert ctrl.detector_options()
    assert "coco" in ctrl.export_options()


# --- thread safety ---------------------------------------------------------
def test_planner_runs_off_ui_thread(qtbot, container, dataset_dir):
    ctrl = BackendController(container)
    pid = _imported(ctrl, dataset_dir)
    tm = ThreadManager()
    out = {}
    worker = tm.submit(ctrl.plan_dataset, pid, None,
                       on_finished=lambda v: out.setdefault("view", v))
    with qtbot.waitSignal(worker.signals.finished, timeout=30000):
        pass
    assert len(out["view"].decisions) == 11
    assert tm.active == 0


# --- Planner execution through the page ------------------------------------
def test_planner_page_generate_populates(qtbot, container, dataset_dir):
    ctrl = BackendController(container)
    _imported(ctrl, dataset_dir)
    page = PlannerPage(ctrl, ThreadManager(), NotificationSystem())
    qtbot.addWidget(page)
    page.on_show()  # loads the dataset combo
    assert page._dataset.count() == 1
    page._generate()  # threaded plan
    qtbot.waitUntil(lambda: page._decisions.rowCount() == 11, timeout=30000)
    assert page._summary.rowCount() == 11
    assert page._current is not None and page._original is not None
