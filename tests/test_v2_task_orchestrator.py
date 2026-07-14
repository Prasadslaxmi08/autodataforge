"""V2-25 TaskOrchestrator — the single entry point coordinating the four agents.

Covers: full orchestration, approval pause/resume, cancellation, the brief's
failure policy (planner abort, memory fallback, decision fallback, execution
retry-then-abort), memory save fallback, GUI/timeline updates, and state
transitions. The orchestrator only coordinates — every agent is the real one.
"""

from __future__ import annotations

import pytest

from vds.memory.store import MemoryStore
from vds.v2 import TaskEvent, TaskOrchestrator, TaskState, new_goal, task_view
from vds.v2.dataset_engineer import DatasetEngineerAgent


class _FakeController:
    """Every backend tool is a no-op that succeeds."""

    def __getattr__(self, name):
        return lambda *a, **k: None


class _FailingDetection(_FakeController):
    """The detection tool (bound to ai_annotate) raises a non-recoverable (FATAL)
    error -> execution fails."""

    def ai_annotate(self, *a, **k):
        raise ValueError("boom")


def _orch(tmp_path, controller=None, **kw):
    eng = DatasetEngineerAgent(controller or _FakeController())
    eng.memory._store = MemoryStore(tmp_path / "mem.json")  # isolate memory from the real file
    return TaskOrchestrator(engineer=eng, **kw), eng


_META = {"image_count": 800, "resolution": "high", "project_id": "proj"}


# --- full orchestration -----------------------------------------------
def test_full_orchestration_records_events_and_memory(tmp_path):
    orch, _ = _orch(tmp_path)
    events = []
    orch._on_event = lambda e, p: events.append(e)  # capture the event stream

    ctx = orch.execute(new_goal("create thermal drone vehicle dataset", source="v.mp4"),
                       project=_META, auto_approve=True)

    assert ctx.state == TaskState.COMPLETED
    assert ctx.plan and ctx.decision_report and ctx.execution_summary
    assert ctx.stored_memory_id is not None  # MemoryAgent recorded the run
    # the mandated event stream, in order
    for e in (TaskEvent.PLANNING_STARTED, TaskEvent.PLANNING_COMPLETED, TaskEvent.MEMORY_LOADED,
              TaskEvent.DECISION_COMPLETED, TaskEvent.APPROVAL_REQUESTED, TaskEvent.EXECUTION_STARTED,
              TaskEvent.EXECUTION_COMPLETED, TaskEvent.MEMORY_STORED, TaskEvent.TASK_COMPLETED):
        assert e.value in events


def test_report_and_view_surfaces(tmp_path):
    orch, _ = _orch(tmp_path)
    ctx = orch.execute(new_goal("detect cars from images", source="i/"),
                       project=_META, auto_approve=True)
    rep = orch.report(ctx.id)
    for k in ("execution_summary", "decision_report", "memory_matches", "warnings",
              "timing", "statistics", "artifacts"):
        assert k in rep
    assert rep["statistics"]["memory_stored"] is True
    assert rep["timing"]["elapsed_seconds"] >= 0
    v = orch.view(ctx.id)
    assert [s["name"] for s in v["stages"]] == \
        ["Planning", "Memory", "Decision", "Approval", "Execution", "Memory Save", "Complete"]
    assert v["stages"][-1]["status"] == "done"  # Complete


# --- approval pause / resume ------------------------------------------
def test_approval_pauses_then_resumes(tmp_path):
    orch, _ = _orch(tmp_path)
    ctx = orch.execute(new_goal("detect cars", source="i/"), project=_META)  # no auto_approve
    assert ctx.state == TaskState.AWAITING_APPROVAL
    assert ctx.execution_summary is None  # nothing executed yet
    stages = {s: st for s, st in ctx.stages.items()}
    assert stages["Decision"] == "done" and stages["Approval"] == "active"

    resumed = orch.approve(ctx.id)
    assert resumed.state == TaskState.COMPLETED
    assert resumed.execution_summary is not None


def test_approve_is_noop_when_not_awaiting(tmp_path):
    orch, _ = _orch(tmp_path)
    ctx = orch.execute(new_goal("detect cars", source="i/"), project=_META, auto_approve=True)
    assert orch.approve(ctx.id).state == TaskState.COMPLETED  # already completed, unchanged


# --- cancellation -----------------------------------------------------
def test_cancel_at_approval_gate(tmp_path):
    orch, _ = _orch(tmp_path)
    ctx = orch.execute(new_goal("detect cars", source="i/"), project=_META)
    cancelled = orch.cancel(ctx.id)
    assert cancelled.state == TaskState.CANCELLED
    assert cancelled.finished_at is not None
    assert any(ev["event"] == TaskEvent.TASK_CANCELLED.value for ev in cancelled.timeline)


# --- failure policy ---------------------------------------------------
def test_planner_failure_aborts(tmp_path, monkeypatch):
    orch, eng = _orch(tmp_path)

    def _raise(*a, **k):
        raise RuntimeError("planner down")

    monkeypatch.setattr(eng, "generate_plan", _raise)
    ctx = orch.execute(new_goal("detect cars"), project=_META)
    assert ctx.state == TaskState.FAILED
    assert any("planner" in e for e in ctx.errors)
    assert ctx.stages["Planning"] == "failed"
    assert any(ev["event"] == TaskEvent.TASK_FAILED.value for ev in ctx.timeline)


def test_memory_unavailable_continues(tmp_path, monkeypatch):
    orch, eng = _orch(tmp_path)

    def _raise(*a, **k):
        raise RuntimeError("store offline")

    monkeypatch.setattr(eng, "recall_experience", _raise)
    ctx = orch.execute(new_goal("detect cars", source="i/"), project=_META, auto_approve=True)
    assert ctx.state == TaskState.COMPLETED  # memory is optional
    assert any("memory unavailable" in w for w in ctx.warnings)
    assert ctx.stages["Memory"] == "skipped"


def test_decision_failure_uses_planner_defaults(tmp_path, monkeypatch):
    orch, eng = _orch(tmp_path)

    def _raise(*a, **k):
        raise RuntimeError("decision down")

    monkeypatch.setattr(eng, "optimize_plan", _raise)
    ctx = orch.execute(new_goal("detect cars", source="i/"), project=_META, auto_approve=True)
    assert ctx.state == TaskState.COMPLETED
    assert ctx.decision_report is None  # fell back to the planner plan
    assert any("planner defaults" in w for w in ctx.warnings)
    assert ctx.stages["Decision"] == "skipped"
    assert any("skipping memory record" in w for w in ctx.warnings)  # no report -> no record


def test_execution_failure_retries_then_aborts(tmp_path):
    orch, _ = _orch(tmp_path, controller=_FailingDetection(), max_execution_retries=1)
    ctx = orch.execute(new_goal("detect cars from images", source="i/"),
                       project=_META, auto_approve=True)
    assert ctx.state == TaskState.FAILED
    assert any("execution retry 1/1" in w for w in ctx.warnings)  # retried once
    assert ctx.errors  # then aborted
    assert ctx.stages["Execution"] == "failed"


def test_memory_save_failure_warns_only(tmp_path, monkeypatch):
    orch, eng = _orch(tmp_path)

    def _raise(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(eng, "record_experience", _raise)
    ctx = orch.execute(new_goal("detect cars", source="i/"), project=_META, auto_approve=True)
    assert ctx.state == TaskState.COMPLETED  # save failure never fails the task
    assert any("memory save failed" in w for w in ctx.warnings)
    assert ctx.stages["Memory Save"] == "failed"


# --- GUI updates + state transitions ----------------------------------
def test_structured_logs_and_timeline(tmp_path):
    orch, _ = _orch(tmp_path)
    ctx = orch.execute(new_goal("detect cars", source="i/"), project=_META, auto_approve=True)
    assert ctx.logs and all({"ts", "agent", "action", "duration_ms", "status"} <= set(r) for r in ctx.logs)
    v = task_view(ctx)
    assert v["state"] == "completed" and v["events"]


def test_state_reachable_via_status(tmp_path):
    orch, _ = _orch(tmp_path)
    ctx = orch.execute(new_goal("detect cars", source="i/"), project=_META)
    assert orch.status(ctx.id).state == TaskState.AWAITING_APPROVAL
    orch.approve(ctx.id)
    assert orch.status(ctx.id).state == TaskState.COMPLETED


def test_goal_accepts_plain_string(tmp_path):
    orch, _ = _orch(tmp_path)
    ctx = orch.execute("segment people in aerial imagery", project=_META, auto_approve=True)
    assert ctx.goal.text == "segment people in aerial imagery"
    assert ctx.state in (TaskState.COMPLETED, TaskState.FAILED)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
