"""AI Verification Workspace tests (Phase 14): image loading, overlay rendering,
object navigation, evidence display, Engineering Memory integration, filtering,
search, human-review actions, statistics, backend communication, thread safety."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from tests.test_planner_workspace import _seed_matching_memory  # noqa: E402
from vds.gui.controller import BackendController  # noqa: E402
from vds.gui.notifications import NotificationSystem  # noqa: E402
from vds.gui.pages.verification import VerificationPage  # noqa: E402
from vds.gui.threads import ThreadManager  # noqa: E402


def _loaded(container, dataset_dir):
    ctrl = BackendController(container)
    ctrl.import_dataset(str(dataset_dir), "verifyme")
    pid = ctrl.list_datasets()[0].project_id
    return ctrl, pid


# --- backend communication + verdict reproduction --------------------------
def test_object_verdicts_reproduced(container, dataset_dir):
    ctrl, pid = _loaded(container, dataset_dir)
    verdicts = ctrl.object_verdicts(pid)
    assert len(verdicts) == 6  # 3 images × 2 objects
    v = verdicts[0]
    assert v.status in ("Verified", "Needs Review", "Rejected", "Uncertain")
    assert v.rationale  # real verifier rationale
    assert v.box is not None


# --- evidence (measured, never fabricated) ---------------------------------
def test_evidence_scores_are_measured(container, dataset_dir):
    ctrl, pid = _loaded(container, dataset_dir)
    v = ctrl.object_verdicts(pid)[0]
    ev = ctrl.object_evidence(v)
    names = {s.label for s in ev.stars}
    assert {"Detection Confidence", "Geometry Consistency", "Context Consistency"} <= names
    # No memory yet -> historical agreement is unavailable, not invented.
    hist = next(s for s in ev.stars if s.label == "Historical Agreement")
    assert hist.value is None
    assert ev.risk in ("Low", "Medium", "High")


# --- statistics ------------------------------------------------------------
def test_verification_statistics(container, dataset_dir):
    ctrl, pid = _loaded(container, dataset_dir)
    verdicts = ctrl.object_verdicts(pid)
    s = ctrl.verification_stats(verdicts)
    assert s.verified + s.rejected + s.needs_review == len(verdicts)
    assert "unavailable" in s.verification_runtime  # honestly reported


# --- Engineering Memory integration ----------------------------------------
def test_history_absent_then_present(container, dataset_dir):
    ctrl, pid = _loaded(container, dataset_dir)
    absent = ctrl.verification_history(pid)
    assert absent.influenced is False
    _seed_matching_memory(container, pid)
    present = ctrl.verification_history(pid)
    assert present.influenced is True
    assert present.matches and present.matches[0]["similarity"] > 0.5


# --- human review integrates with the state machine ------------------------
def test_review_action_respects_state_machine(container, dataset_dir):
    ctrl, pid = _loaded(container, dataset_dir)
    verdicts = ctrl.object_verdicts(pid)
    auto = next(v for v in verdicts if v.state == "auto_accepted")
    ok, _msg = ctrl.apply_review(auto.object_id, "mark_review")  # auto_accepted -> needs_review (legal)
    assert ok
    ok2, _ = ctrl.apply_review(auto.object_id, "approve")  # needs_review -> accepted (legal)
    assert ok2
    ok3, msg3 = ctrl.apply_review(auto.object_id, "mark_review")  # accepted -> needs_review (illegal)
    assert not ok3 and "not permitted" in msg3


# --- thread safety ---------------------------------------------------------
def test_verdicts_load_off_ui_thread(qtbot, container, dataset_dir):
    ctrl, pid = _loaded(container, dataset_dir)
    tm = ThreadManager()
    out = {}
    worker = tm.submit(ctrl.object_verdicts, pid, on_finished=lambda v: out.setdefault("v", v))
    with qtbot.waitSignal(worker.signals.finished, timeout=30000):
        pass
    assert len(out["v"]) == 6
    assert tm.active == 0


# --- page: load, navigation, evidence, filtering, search -------------------
def _page(container, dataset_dir):
    ctrl, pid = _loaded(container, dataset_dir)
    page = VerificationPage(ctrl, ThreadManager(), NotificationSystem())
    page.on_show()
    page._verdicts = ctrl.object_verdicts(pid)
    page._project_id = pid
    page._on_loaded(page._verdicts)
    return ctrl, page


def test_page_load_and_navigation(qtbot, container, dataset_dir):
    _ctrl, page = _page(container, dataset_dir)
    qtbot.addWidget(page)
    assert page._table.rowCount() == 6
    assert page._stats.count() > 0
    page._table.selectRow(0)
    assert page._selected is not None
    assert page._evidence.count() > 1  # evidence populated
    assert page._timeline.count() > 1  # timeline populated
    assert page._review_buttons[0].isEnabled()


def test_page_filtering_and_search(qtbot, container, dataset_dir):
    _ctrl, page = _page(container, dataset_dir)
    qtbot.addWidget(page)
    total = page._table.rowCount()
    page._f_status.setCurrentText("Rejected")  # synthetic objects are all high-confidence
    assert page._table.rowCount() == 0
    page._f_status.setCurrentText("All")
    assert page._table.rowCount() == total
    page._f_label.setText("no-such-label")  # search by label
    assert page._table.rowCount() == 0
    page._f_label.setText("object")
    assert page._table.rowCount() == total


def test_page_overlay_toggle(qtbot, container, dataset_dir):
    _ctrl, page = _page(container, dataset_dir)
    qtbot.addWidget(page)
    page._table.selectRow(0)
    page._t_boxes.setChecked(False)  # hide boxes -> should not raise, re-renders
    page._t_boxes.setChecked(True)
    assert page._selected is not None
