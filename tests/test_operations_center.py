"""Operations & Performance Center tests (Phase 17): benchmark loading, comparison,
historical trends, filtering, report export, backend communication, thread safety,
dashboard refresh, and performance calculations. Every asserted value comes from
measured execution data; unmeasured metrics are reported as 'Unavailable'."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from tests.test_planner_workspace import _seed_matching_memory  # noqa: E402
from vds.gui.controller import BackendController  # noqa: E402
from vds.gui.notifications import NotificationSystem  # noqa: E402
from vds.gui.operations_view import NA  # noqa: E402
from vds.gui.pages.operations import OperationsPage  # noqa: E402
from vds.gui.threads import ThreadManager  # noqa: E402

CREATED = "2026-07-10T12:00:00"


def _with_runs(container, dataset_dir):
    ctrl = BackendController(container)
    ctrl.import_dataset(str(dataset_dir), "alpha")
    pid = ctrl.list_datasets()[0].project_id
    ctrl.analyze_dataset(pid, CREATED)  # records one benchmark run
    _seed_matching_memory(container, pid)  # a second run
    return ctrl, pid


# --- backend communication + performance calculations ----------------------
def test_empty_platform_is_honest(container):
    ctrl = BackendController(container)
    live = ctrl.ops_snapshot(0)
    kpis = {k.label: k.value for k in ctrl.ops_overview(live)}
    assert kpis["Average Throughput"] == NA  # no runs -> not fabricated
    assert kpis["Completed Jobs"] == "0"
    assert ctrl.ops_benchmarks() == []
    assert ctrl.ops_trends() == []


def test_kpis_from_measured_data(container, dataset_dir):
    ctrl, _pid = _with_runs(container, dataset_dir)
    live = ctrl.ops_snapshot(0)
    kpis = {k.label: k.value for k in ctrl.ops_overview(live)}
    assert kpis["Completed Jobs"] == "2"
    assert kpis["Failed Jobs"] == NA  # no failure log -> honest
    assert "img/s" in kpis["Average Throughput"]
    assert kpis["Platform Status"] in ("Healthy", "Warning", "Critical", "Unknown")


# --- system performance (live or latest measured, GPU unavailable) ---------
def test_system_performance_gpu_unavailable(container, dataset_dir):
    ctrl, _pid = _with_runs(container, dataset_dir)
    stats = {s.name: s for s in ctrl.ops_system(ctrl.ops_snapshot(0))}
    assert NA in stats["GPU Usage"].value  # never fabricated
    assert stats["GPU Usage"].status == "na"
    assert stats["Backend Status"].value == "Online"


# --- benchmark explorer ----------------------------------------------------
def test_benchmark_loading(container, dataset_dir):
    ctrl, _pid = _with_runs(container, dataset_dir)
    runs = ctrl.ops_benchmarks()
    assert len(runs) == 2
    r = runs[0]
    assert r.peak_gpu == NA  # GPU memory not measured
    assert "s" in r.runtime and "%" in r.review_rate


# --- performance comparison ------------------------------------------------
def test_comparison_highlights_trend(container, dataset_dir):
    ctrl, _pid = _with_runs(container, dataset_dir)
    ids = [r.run_id for r in ctrl.ops_benchmarks()]
    cmp = ctrl.ops_compare(ids)
    assert len(cmp.runs) == 2
    metrics = {row.metric for row in cmp.rows}
    assert {"Runtime (s)", "Throughput (img/s)", "Review Rate"} <= metrics
    for row in cmp.rows:
        assert row.trend in ("improved", "regressed", "")
    # unmeasured comparison metrics are Unavailable, not invented
    gpu = next(r for r in cmp.rows if r.metric == "GPU")
    assert all(v == NA for v in gpu.values)


def test_comparison_empty(container, dataset_dir):
    ctrl, _pid = _with_runs(container, dataset_dir)
    cmp = ctrl.ops_compare([])
    assert cmp.runs == [] and cmp.rows == []


# --- historical trends -----------------------------------------------------
def test_historical_trends(container, dataset_dir):
    ctrl, _pid = _with_runs(container, dataset_dir)
    trends = {t.metric: t for t in ctrl.ops_trends()}
    assert {"Review Reduction", "Throughput (img/s)", "Dataset Growth"} <= set(trends)
    assert trends["Dataset Growth"].series == [1.0, 2.0]  # cumulative, measured count


# --- platform health -------------------------------------------------------
def test_platform_health_indicators(container, dataset_dir):
    ctrl, _pid = _with_runs(container, dataset_dir)
    health = ctrl.ops_health(ctrl.ops_snapshot(0))
    names = {i.name for i in health.indicators}
    assert {"Export Failures", "Memory Pressure", "GPU Availability"} <= names
    assert health.status in ("Healthy", "Warning", "Critical", "Unknown")


# --- report export ---------------------------------------------------------
def test_report_export_sections(container, dataset_dir):
    ctrl, _pid = _with_runs(container, dataset_dir)
    live = ctrl.ops_snapshot(0)
    for section in ("operations", "benchmark", "trends", "performance_summary", "full"):
        md = ctrl.ops_markdown(section, live)
        assert md.strip()
    assert "Executive Operations Overview" in ctrl.ops_markdown("operations", live)


# --- thread safety ---------------------------------------------------------
def test_refresh_off_ui_thread(qtbot, container, dataset_dir):
    ctrl, _pid = _with_runs(container, dataset_dir)
    tm = ThreadManager()
    out = {}
    worker = tm.submit(ctrl.ops_benchmarks, on_finished=lambda r: out.setdefault("r", r))
    with qtbot.waitSignal(worker.signals.finished, timeout=30000):
        pass
    assert len(out["r"]) == 2
    assert tm.active == 0


# --- page: refresh, filtering, compare -------------------------------------
def _page(container, dataset_dir):
    ctrl, _pid = _with_runs(container, dataset_dir)
    page = OperationsPage(ctrl, ThreadManager(), NotificationSystem())
    page._refresh()  # loads synchronously on the UI thread
    return ctrl, page


def test_page_dashboard_refresh(qtbot, container, dataset_dir):
    _ctrl, page = _page(container, dataset_dir)
    qtbot.addWidget(page)
    assert page._overview.count() == 12  # all KPI tiles
    assert page._system.count() == 11
    assert page._benchmark.rowCount() == 2
    assert page._trends.count() >= 1


def test_page_filtering(qtbot, container, dataset_dir):
    _ctrl, page = _page(container, dataset_dir)
    qtbot.addWidget(page)
    total = page._benchmark.rowCount()
    page._f_search.setText("zzz-no-such-run")
    assert page._benchmark.rowCount() == 0
    page._f_search.setText("")
    assert page._benchmark.rowCount() == total
    page._f_date.setText("1999-01-01")
    assert page._benchmark.rowCount() == 0


def test_page_compare_selected(qtbot, container, dataset_dir):
    _ctrl, page = _page(container, dataset_dir)
    qtbot.addWidget(page)
    page._benchmark.selectAll()
    page._compare()
    assert page._comparison.rowCount() >= 1
    assert page._comparison.columnCount() == 3  # Metric + 2 runs
