"""Knowledge Center tests (Phase 16): search, filtering, historical comparison,
knowledge cards, timeline, lessons, export, backend communication, thread safety.
Every asserted value comes from a stored Engineering-Memory record — nothing is
fabricated, and empty memory is reported honestly."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from tests.test_planner_workspace import _seed_matching_memory  # noqa: E402
from vds.gui.controller import BackendController  # noqa: E402
from vds.gui.notifications import NotificationSystem  # noqa: E402
from vds.gui.pages.knowledge import KnowledgePage  # noqa: E402
from vds.gui.threads import ThreadManager  # noqa: E402

CREATED = "2026-07-10T12:00:00"


def _with_memory(container, dataset_dir):
    """Import + analyze (records one memory) + seed a second matching record."""
    ctrl = BackendController(container)
    ctrl.import_dataset(str(dataset_dir), "alpha")
    pid = ctrl.list_datasets()[0].project_id
    ctrl.analyze_dataset(pid, CREATED)  # records validated knowledge
    _seed_matching_memory(container, pid)  # a second, higher-quality prior run
    return ctrl, pid


# --- backend communication + dataset history -------------------------------
def test_empty_memory_is_honest(container):
    ctrl = BackendController(container)
    assert ctrl.knowledge_records() == []
    assert ctrl.knowledge_cards() == []
    assert ctrl.lessons_learned() == []
    assert ctrl.knowledge_timeline() == []


def test_dataset_history_from_records(container, dataset_dir):
    ctrl, _pid = _with_memory(container, dataset_dir)
    records = ctrl.knowledge_records()
    assert len(records) == 2
    r = records[0]
    assert 0 <= r.health <= 100
    assert r.status in ("validated", "provisional", "rejected")
    assert r.planner_strategy  # measured planner decisions


# --- search ----------------------------------------------------------------
def test_search_by_field(container, dataset_dir):
    ctrl, _pid = _with_memory(container, dataset_dir)
    all_recs = ctrl.knowledge_records()
    # keyword search with empty query returns everything
    assert len(ctrl.search_knowledge("", "Keyword")) == len(all_recs)
    # a detector present in the records is found; a bogus one is not
    det = all_recs[0].detector
    assert ctrl.search_knowledge(det, "Detector")
    assert ctrl.search_knowledge("zzz-nonexistent-detector", "Detector") == []


# --- knowledge cards -------------------------------------------------------
def test_knowledge_cards_are_measured(container, dataset_dir):
    ctrl, _pid = _with_memory(container, dataset_dir)
    cards = ctrl.knowledge_cards()
    for c in cards:
        assert c.occurrences >= 1
        assert 0.0 <= c.success_rate <= 1.0
        assert c.supporting_datasets  # never a card without its datasets


# --- timeline --------------------------------------------------------------
def test_timeline_records_processing(container, dataset_dir):
    ctrl, _pid = _with_memory(container, dataset_dir)
    events = ctrl.knowledge_timeline()
    kinds = {e.kind for e in events}
    assert "Dataset Processed" in kinds
    # two records -> at least one comparative event appears
    assert len(events) >= 2


# --- historical comparison -------------------------------------------------
def test_compare_records(container, dataset_dir):
    ctrl, _pid = _with_memory(container, dataset_dir)
    ids = [r.id for r in ctrl.knowledge_records()]
    cmp = ctrl.compare_knowledge(ids)
    assert len(cmp.datasets) == 2
    metrics = {row.metric for row in cmp.rows}
    assert {"Review Rate", "Dataset Health", "Runtime (s)"} <= metrics
    # trend direction is one of the allowed labels
    for row in cmp.rows:
        assert row.trend in ("improved", "regressed", "")


def test_compare_empty_selection(container, dataset_dir):
    ctrl, _pid = _with_memory(container, dataset_dir)
    cmp = ctrl.compare_knowledge([])
    assert cmp.datasets == [] and cmp.rows == []


# --- lessons learned -------------------------------------------------------
def test_lessons_learned_from_validated_recs(container, dataset_dir):
    ctrl, _pid = _with_memory(container, dataset_dir)
    lessons = ctrl.lessons_learned()
    for lsn in lessons:
        assert lsn.solution and lsn.problem
        assert lsn.occurrences >= 1
        assert 0.0 <= lsn.confidence <= 1.0
        assert lsn.reference_datasets


# --- export ----------------------------------------------------------------
def test_export_sections(container, dataset_dir):
    ctrl, _pid = _with_memory(container, dataset_dir)
    ids = [r.id for r in ctrl.knowledge_records()]
    for section, extra in [("knowledge_report", None), ("engineering_summary", None),
                           ("lessons", None), ("comparison", ids), ("full", None)]:
        md = ctrl.knowledge_markdown(section, extra)
        assert md.strip()
    assert "| Metric |" in ctrl.knowledge_markdown("comparison", ids)
    assert "Lessons Learned" in ctrl.knowledge_markdown("lessons")


# --- thread safety ---------------------------------------------------------
def test_load_off_ui_thread(qtbot, container, dataset_dir):
    ctrl, _pid = _with_memory(container, dataset_dir)
    tm = ThreadManager()
    out = {}
    worker = tm.submit(ctrl.knowledge_records, on_finished=lambda r: out.setdefault("r", r))
    with qtbot.waitSignal(worker.signals.finished, timeout=30000):
        pass
    assert len(out["r"]) == 2
    assert tm.active == 0


# --- page: load, filter, compare -------------------------------------------
def _page(container, dataset_dir):
    ctrl, _pid = _with_memory(container, dataset_dir)
    page = KnowledgePage(ctrl, ThreadManager(), NotificationSystem())
    page._on_loaded(page._load())  # populate synchronously (bypass the worker)
    return ctrl, page


def test_page_populates_all_sections(qtbot, container, dataset_dir):
    _ctrl, page = _page(container, dataset_dir)
    qtbot.addWidget(page)
    assert page._history.rowCount() == 2
    assert page._cards.count() >= 1
    assert page._timeline.count() >= 2
    assert page._lessons.count() >= 1


def test_page_filtering(qtbot, container, dataset_dir):
    _ctrl, page = _page(container, dataset_dir)
    qtbot.addWidget(page)
    total = page._history.rowCount()
    page._f_priority.setCurrentText("High (≥75)")  # excludes lower-health records
    assert page._history.rowCount() <= total
    page._f_priority.setCurrentText("All")
    assert page._history.rowCount() == total
    page._f_date.setText("1999-01-01")  # no record has this date
    assert page._history.rowCount() == 0


def test_page_compare_selected(qtbot, container, dataset_dir):
    _ctrl, page = _page(container, dataset_dir)
    qtbot.addWidget(page)
    page._history.selectAll()
    page._compare()
    assert page._comparison.rowCount() >= 1
    assert page._comparison.columnCount() == 3  # Metric + 2 datasets
