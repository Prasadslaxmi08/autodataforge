"""V2-22 ExecutionAgent — step execution, gates, pause/resume, retry, failure,
cancellation, progress, summary, GUI surface. Tools are fakes (no real backend).
"""

from __future__ import annotations

import pytest

from vds.v2 import (
    ExecutionAgent,
    ExecutionError,
    GateReason,
    PlannerAgent,
    PlanStatus,
    Tool,
    ToolRegistry,
    new_goal,
)
from vds.v2.state import SessionStatus

_TOOL_NAMES = ["import_images", "import_video", "extract_frames", "run_detection",
               "run_segmentation", "review_dataset", "export_dataset", "generate_report",
               "open_project"]


class _Recorder:
    """A fake tool: records calls, optionally fails the first N calls."""

    def __init__(self, fail_times: int = 0, exc: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._fail = fail_times
        self._exc = exc or FileNotFoundError("folder unavailable")

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self._fail:
            raise self._exc
        return None


def _tools(**overrides) -> ToolRegistry:
    reg = ToolRegistry()
    for name in _TOOL_NAMES:
        reg.register(Tool(name=name, description="", run=overrides.get(name, _Recorder())))
    return reg


def _approved_plan(text="detect cars from images", **params):
    plan = PlannerAgent().create_plan(new_goal(text, source="imgs/", **params))
    plan.status = PlanStatus.APPROVED
    return plan


def _inputs():
    return {"source": "imgs/", "name": "cars", "project_id": "cars"}


# --- step execution + progress + summary ------------------------------
def test_runs_to_gate_then_completes():
    plan = _approved_plan()
    agent = ExecutionAgent(_tools())
    ctx = agent.execute(plan, _inputs())
    assert ctx.status == SessionStatus.AWAITING_APPROVAL
    assert ctx.gate_reason == GateReason.HUMAN_REVIEW.value
    assert ctx.step_status["import_images"] == "done"
    assert ctx.step_status["analyse_inputs"] == "skipped"  # Agent never plans
    assert 0.0 < ctx.progress < 1.0

    ctx = agent.approve(plan.id)
    assert ctx.status == SessionStatus.COMPLETED
    assert ctx.progress == 1.0
    s = agent.summary(plan.id)
    assert s.completed == 6 and s.skipped == 1 and s.failed == 0
    assert agent.report(plan.id).startswith("# Execution Summary")


def test_tools_invoked_with_resolved_args():
    export = _Recorder()
    plan = _approved_plan()
    agent = ExecutionAgent(_tools(export_dataset=export))
    agent.execute(plan, _inputs())
    agent.approve(plan.id)
    assert export.calls == [{"project_id": "cars", "fmt": "coco"}]


# --- approval gates ----------------------------------------------------
def test_export_confirmation_gate_when_enabled():
    plan = _approved_plan()
    agent = ExecutionAgent(_tools(), gates={GateReason.HUMAN_REVIEW, GateReason.EXPORT_CONFIRMATION})
    agent.execute(plan, _inputs())
    ctx = agent.approve(plan.id)  # clear the manual-review gate
    assert ctx.status == SessionStatus.AWAITING_APPROVAL  # now stopped at export gate
    assert ctx.gate_reason == GateReason.EXPORT_CONFIRMATION.value
    ctx = agent.approve(plan.id, "export_dataset")
    assert ctx.status == SessionStatus.COMPLETED


# --- pause / resume ----------------------------------------------------
def test_pause_then_resume():
    plan = _approved_plan()
    agent = ExecutionAgent(_tools())
    agent.execute(plan, _inputs())  # parks at gate
    agent.pause(plan.id)
    assert agent.context(plan.id).status == SessionStatus.PAUSED
    ctx = agent.resume(plan.id)
    assert ctx.status == SessionStatus.AWAITING_APPROVAL  # resumes, re-parks at the gate


# --- retry / recovery --------------------------------------------------
def test_recoverable_failure_retries_and_succeeds():
    flaky = _Recorder(fail_times=2)  # FileNotFoundError twice, then ok
    plan = _approved_plan()
    agent = ExecutionAgent(_tools(import_images=flaky), max_retries=2)
    ctx = agent.execute(plan, _inputs())
    assert ctx.step_status["import_images"] == "done"
    assert ctx.attempts["import_images"] == 2 and len(flaky.calls) == 3
    assert any("retry" in w for w in ctx.warnings)


def test_recoverable_failure_exhausts_retries():
    always = _Recorder(fail_times=99)
    plan = _approved_plan()
    agent = ExecutionAgent(_tools(import_images=always), max_retries=1)
    ctx = agent.execute(plan, _inputs())
    assert ctx.status == SessionStatus.FAILED
    assert ctx.step_status["import_images"] == "failed"
    # then a manual retry recovers if the tool is (conceptually) fixed — here it still fails,
    # so it just re-fails; verify retry re-drives the step:
    ctx = agent.retry(plan.id, "import_images")
    assert ctx.status == SessionStatus.FAILED
    assert len(always.calls) > 2


def test_fatal_failure_stops_without_retry():
    boom = _Recorder(fail_times=99, exc=ValueError("totally broken"))
    plan = _approved_plan()
    agent = ExecutionAgent(_tools(import_images=boom), max_retries=3)
    ctx = agent.execute(plan, _inputs())
    assert ctx.status == SessionStatus.FAILED
    assert len(boom.calls) == 1  # fatal -> no retries


def test_retry_after_fix_completes():
    flaky = _Recorder(fail_times=1)
    plan = _approved_plan()
    agent = ExecutionAgent(_tools(import_images=flaky), max_retries=0)  # no auto-retry
    ctx = agent.execute(plan, _inputs())
    assert ctx.status == SessionStatus.FAILED  # failed on first attempt
    ctx = agent.retry(plan.id, "import_images")  # now the flaky tool succeeds
    assert ctx.status == SessionStatus.AWAITING_APPROVAL  # recovered, ran on to the gate


# --- cancellation ------------------------------------------------------
def test_cancel_marks_remaining_cancelled():
    plan = _approved_plan()
    agent = ExecutionAgent(_tools())
    agent.execute(plan, _inputs())  # at gate
    ctx = agent.cancel(plan.id)
    assert ctx.status == SessionStatus.CANCELLED
    assert ctx.step_status["export_dataset"] == "cancelled"
    assert ctx.step_status["manual_review"] == "cancelled"


# --- prerequisites -----------------------------------------------------
def test_unapproved_plan_is_rejected():
    plan = PlannerAgent().create_plan(new_goal("detect cars", source="imgs/"))  # DRAFT
    agent = ExecutionAgent(_tools())
    with pytest.raises(ExecutionError):
        agent.execute(plan, _inputs())
    ctx = agent.execute(plan, _inputs(), require_approval=False)  # override allowed
    assert ctx.status in (SessionStatus.AWAITING_APPROVAL, SessionStatus.COMPLETED)


# --- GUI surface + progress updates -----------------------------------
def test_view_surface_and_live_log():
    events: list[str] = []
    plan = _approved_plan()
    agent = ExecutionAgent(_tools(), on_event=lambda k, p: events.append(k))
    agent.execute(plan, _inputs())
    v = agent.view(plan.id)
    assert v["status"] == "awaiting_approval"
    assert v["current_step"] == "Manual Review"
    assert "Export Dataset" in v["upcoming"]
    assert v["log"] and 0.0 < v["progress"] < 1.0
    assert "step_done" in events and "awaiting_approval" in events


def test_dataset_engineer_execute_plan_integration():
    class _FakeController:
        def __getattr__(self, name):
            return lambda *a, **k: None

    from vds.v2 import DatasetEngineerAgent

    eng = DatasetEngineerAgent(_FakeController())  # type: ignore[arg-type]
    plan = eng.generate_plan(new_goal("detect cars from images", source="imgs/"))
    plan.status = PlanStatus.APPROVED
    ctx = eng.execute_plan(plan, _inputs())
    assert ctx.status == SessionStatus.AWAITING_APPROVAL
    view = eng.execution_view(plan.id)
    assert view["total"] == len(plan.steps)
    eng.executor.approve(plan.id)
    assert eng.execution_view(plan.id)["status"] == "completed"
