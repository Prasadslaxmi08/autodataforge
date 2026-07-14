"""Pipeline Workspace tests (Phase 13): pipeline execution, thread safety, pause,
resume, cancel, restart, live preview, metrics refresh, charts, processing log,
pipeline summary, backend communication, error recovery. Headless (offscreen)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from vds.gui.controller import BackendController  # noqa: E402
from vds.gui.notifications import NotificationSystem  # noqa: E402
from vds.gui.pages.pipeline import PipelinePage  # noqa: E402
from vds.gui.threads import ThreadManager  # noqa: E402


def _page(container, dataset_dir):
    ctrl = BackendController(container)
    page = PipelinePage(ctrl, ThreadManager(), NotificationSystem())
    return ctrl, page, str(dataset_dir)


# --- backend communication + execution -------------------------------------
def test_pipeline_execution_and_derivations(container, dataset_dir):
    ctrl = BackendController(container)
    report = ctrl.run_pipeline(str(dataset_dir), "run1")
    assert report.imported == 3
    stages = ctrl.stage_timeline(report)
    assert [s.name for s in stages][:2] == ["Dataset Import", "Validation"]
    assert len(stages) == 8
    assert stages[-1].name == "Export"
    activity = ctrl.model_activity(report)
    assert activity.detection["objects"] == report.detections
    events = ctrl.console_events(report)
    assert any("Dataset loaded" in m for _lvl, m in events)
    summary = ctrl.pipeline_summary(report)
    assert summary.total_images == report.imported + report.duplicates_skipped
    assert summary.analyst_summary  # real Analyst executive summary


def test_engineering_memory_stage_is_honestly_skipped(container, dataset_dir):
    ctrl = BackendController(container)
    report = ctrl.run_pipeline(str(dataset_dir), "run2")
    mem_stage = next(s for s in ctrl.stage_timeline(report) if s.name == "Engineering Memory Update")
    assert mem_stage.status == "Skipped"


# --- thread safety ---------------------------------------------------------
def test_pipeline_runs_off_ui_thread(qtbot, container, dataset_dir):
    ctrl = BackendController(container)
    tm = ThreadManager()
    out = {}
    worker = tm.submit(ctrl.run_pipeline, str(dataset_dir), "threaded",
                       on_finished=lambda r: out.setdefault("report", r))
    with qtbot.waitSignal(worker.signals.finished, timeout=30000):
        pass
    assert out["report"].imported == 3
    assert tm.active == 0


# --- page: execution populates all sections --------------------------------
def test_page_run_populates_sections(qtbot, container, dataset_dir):
    ctrl, page, src = _page(container, dataset_dir)
    qtbot.addWidget(page)
    page._source, page._name = src, "pagerun"
    page._launch()  # threaded
    qtbot.waitUntil(lambda: page._state == "completed", timeout=30000)
    assert page._timeline.rowCount() == 8
    assert not page._summary.isHidden()  # summary revealed (page not shown in headless test)
    assert len(page._preview_items) > 0
    assert page._report is not None


# --- pause / resume --------------------------------------------------------
def test_pause_and_resume(container, dataset_dir):
    _ctrl, page, _src = _page(container, dataset_dir)
    page._state = "running"
    page._timer.start()
    page._pause()
    assert page._state == "paused" and not page._timer.isActive()
    page._resume()
    assert page._state == "running" and page._timer.isActive()
    page._timer.stop()


# --- cancel rolls back ------------------------------------------------------
def test_cancel_rolls_back_dataset(container, dataset_dir):
    ctrl, page, src = _page(container, dataset_dir)
    report = ctrl.run_pipeline(src, "tocancel")
    assert ctrl.list_datasets()  # dataset exists
    page._cancelled = True
    page._on_done(report)  # finish arrives after cancel -> rollback
    assert page._state == "cancelled"
    assert all(d.project_id != report.project_id for d in ctrl.list_datasets())


# --- restart ---------------------------------------------------------------
def test_restart_enabled_after_source_set(container, dataset_dir):
    ctrl, page, src = _page(container, dataset_dir)
    report = ctrl.run_pipeline(src, "r")
    page._source, page._name = src, "r"
    page._on_done(report)
    page._refresh_controls()
    assert page._buttons["restart"].isEnabled()


# --- live preview ----------------------------------------------------------
def test_live_preview_navigation(container, dataset_dir):
    ctrl, page, src = _page(container, dataset_dir)
    report = ctrl.run_pipeline(src, "prev")
    page._on_done(report)
    assert len(page._preview_items) > 0
    first = page._preview_idx
    page._next_image()
    assert page._preview_idx != first or len(page._preview_items) == 1


# --- metrics refresh + charts ----------------------------------------------
def test_metrics_tick_feeds_charts(container, dataset_dir):
    import time as _t

    _ctrl, page, _src = _page(container, dataset_dir)
    page._start_ts = _t.monotonic()
    page._tick()
    page._tick()
    assert len(page._cpu_spark._values) == 2
    assert len(page._ram_spark._values) == 2


# --- processing console filtering ------------------------------------------
def test_console_severity_filter(container, dataset_dir):
    _ctrl, page, _src = _page(container, dataset_dir)
    page._add_log("info", "routine message")
    page._add_log("error", "boom failure")
    page._severity.setCurrentText("error")
    page._render_console()
    text = page._console.toPlainText()
    assert "boom failure" in text and "routine message" not in text


# --- error recovery --------------------------------------------------------
def test_error_recovery(container, dataset_dir):
    _ctrl, page, _src = _page(container, dataset_dir)
    page._on_error("simulated failure")
    assert page._state == "failed"
    assert page._timeline.item(0, 1).text() == "Failed"
    assert any(lvl == "error" for _ts, lvl, _msg in page._log)
    # the page recovers: a new run can start
    assert page._buttons["start"].isEnabled()
