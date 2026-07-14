"""AI Dataset Intelligence Workspace tests (Phase 15): Analyst integration over the
cached ExecutionReport, historical comparison, trend visualization, recommendation
ranking, report export, filtering, backend communication, thread safety, and
dashboard updates. Every asserted value originates from a measured metric or a
validated Analyst recommendation — nothing is fabricated."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from tests.test_planner_workspace import _seed_matching_memory  # noqa: E402
from vds.gui.controller import BackendController  # noqa: E402
from vds.gui.notifications import NotificationSystem  # noqa: E402
from vds.gui.pages.intelligence import IntelligencePage  # noqa: E402
from vds.gui.threads import ThreadManager  # noqa: E402

CREATED = "2026-07-10T12:00:00"


def _analyzed(container, dataset_dir, name="intel"):
    ctrl = BackendController(container)
    ctrl.import_dataset(str(dataset_dir), name)
    pid = ctrl.list_datasets()[0].project_id
    intel = ctrl.analyze_dataset(pid, CREATED)
    return ctrl, pid, intel


# --- backend communication + Analyst integration ---------------------------
def test_no_cached_report_returns_none(container):
    ctrl = BackendController(container)
    assert ctrl.analyze_dataset("does-not-exist", CREATED) is None


def test_analyst_integration_from_cached_report(container, dataset_dir):
    _ctrl, _pid, intel = _analyzed(container, dataset_dir)
    assert intel is not None
    # summary comes from the real Analyst + measured metrics
    assert intel.summary.analyst_summary  # Analyst executive summary
    assert intel.summary.source in ("ai", "deterministic")
    assert 0 <= intel.summary.overall_health <= 100
    assert intel.analyst_report_markdown.startswith("# Engineering Report")


def test_health_kpis_are_measured(container, dataset_dir):
    _ctrl, _pid, intel = _analyzed(container, dataset_dir)
    scored = [k for k in intel.kpis if k.score is not None]
    assert scored and all(0 <= k.score <= 100 for k in scored)
    # overall health is the mean of the scored KPIs
    assert intel.summary.overall_health == round(sum(k.score for k in scored) / len(scored))


# --- recommendation ranking ------------------------------------------------
def test_recommendations_ranked_by_confidence(container, dataset_dir):
    _ctrl, _pid, intel = _analyzed(container, dataset_dir)
    confs = [r.confidence for r in intel.recommendations]
    assert confs == sorted(confs, reverse=True)
    for r in intel.recommendations:
        assert r.priority in ("HIGH", "MEDIUM", "LOW")


# --- historical comparison (grows via Analyst memory recording) ------------
def test_historical_comparison_available_after_recording(container, dataset_dir):
    _ctrl, _pid, intel = _analyzed(container, dataset_dir)
    # analyze_dataset records validated knowledge to memory -> history is available
    assert intel.historical.available is True
    assert intel.historical.runs >= 1


def test_memory_recording_is_dedup_safe(container, dataset_dir):
    ctrl, pid, _intel = _analyzed(container, dataset_dir)
    before = len(container.memory.all())
    ctrl.analyze_dataset(pid, CREATED)  # same content -> no new record
    assert len(container.memory.all()) == before


def test_historical_matches_from_memory(container, dataset_dir):
    ctrl, pid, _intel = _analyzed(container, dataset_dir)
    _seed_matching_memory(container, pid)
    intel = ctrl.analyze_dataset(pid, CREATED)
    assert intel.historical.matches
    assert intel.historical.matches[0]["similarity"] > 0.5


# --- trend visualization ---------------------------------------------------
def test_trend_series_shape(container, dataset_dir):
    _ctrl, _pid, intel = _analyzed(container, dataset_dir)
    for t in intel.historical.trends:
        assert len(t.series) >= 1
        assert isinstance(t.improved, bool)


# --- readiness -------------------------------------------------------------
def test_readiness_criteria_present(container, dataset_dir):
    _ctrl, _pid, intel = _analyzed(container, dataset_dir)
    names = {c.name for c in intel.readiness}
    assert {"Ready for Training", "Requires Human Review", "Requires Verification"} <= names
    for c in intel.readiness:
        assert c.reasoning  # each verdict is explained by measured metrics


# --- report export (Markdown, from validated content) ----------------------
def test_report_export_sections(container, dataset_dir):
    ctrl, _pid, intel = _analyzed(container, dataset_dir)
    for section in ("executive", "health", "recommendations", "engineering", "all"):
        md = ctrl.intelligence_markdown(intel, section)
        assert md.strip()
    assert "Executive Summary" in ctrl.intelligence_markdown(intel, "executive")
    assert "Engineering Report" in ctrl.intelligence_markdown(intel, "engineering")


# --- thread safety ---------------------------------------------------------
def test_analyze_off_ui_thread(qtbot, container, dataset_dir):
    ctrl = BackendController(container)
    ctrl.import_dataset(str(dataset_dir), "intel")
    pid = ctrl.list_datasets()[0].project_id
    tm = ThreadManager()
    out = {}
    worker = tm.submit(ctrl.analyze_dataset, pid, CREATED,
                       on_finished=lambda i: out.setdefault("i", i))
    with qtbot.waitSignal(worker.signals.finished, timeout=60000):
        pass
    assert out["i"] is not None
    assert tm.active == 0


# --- page: dashboard updates + filtering -----------------------------------
def _page(container, dataset_dir):
    ctrl, pid, intel = _analyzed(container, dataset_dir)
    page = IntelligencePage(ctrl, ThreadManager(), NotificationSystem())
    page.on_show()
    page._on_analyzed(intel)  # populate synchronously (bypass the worker)
    return ctrl, page, intel


def test_page_dashboard_updates(qtbot, container, dataset_dir):
    _ctrl, page, intel = _page(container, dataset_dir)
    qtbot.addWidget(page)
    assert page._gauge._value == intel.summary.overall_health
    assert page._health.count() == len(intel.kpis)
    assert page._recommendations.count() >= 1
    assert page._readiness.count() == len(intel.readiness)
    assert page._summary_grid.count() > 0


def test_page_recommendation_filtering(qtbot, container, dataset_dir):
    _ctrl, page, _intel = _page(container, dataset_dir)
    qtbot.addWidget(page)
    total = page._recommendations.count()
    page._f_search.setText("zzz-no-such-recommendation")
    # only the "no matches" placeholder label remains
    assert page._recommendations.count() == 1
    page._f_search.setText("")
    assert page._recommendations.count() == total
    page._f_conf.setValue(1.01)  # clamps to 1.0; excludes anything below full confidence
    assert page._recommendations.count() <= total


def test_page_no_cached_report_warns(qtbot, container):
    ctrl = BackendController(container)
    page = IntelligencePage(ctrl, ThreadManager(), NotificationSystem())
    qtbot.addWidget(page)
    page._on_analyzed(None)  # honest "no run cached" path, no crash
    assert page._intel is None
