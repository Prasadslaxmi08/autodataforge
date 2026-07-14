"""V2-20 agent architecture — lifecycle tests (no LLM, no backend run).

Covers the phase's TESTING checklist: agent registration, goal->plan generation,
message protocol, orchestration + tool binding, session lifecycle, pause/resume,
failure recovery, and the GUI view surface. Handlers are no-ops, so nothing here
touches the frozen V1 backend.
"""

from __future__ import annotations

import pytest

from vds.v2 import (
    AgentMessage,
    DatasetEngineerAgent,
    MessageBus,
    Planner,
    SessionStatus,
    ToolRegistry,
    default_registry,
    default_tools,
    new_goal,
)
from vds.v2.agent_base import BaseAgent
from vds.v2.orchestrator import TaskOrchestrator
from vds.v2.planner import StepStatus


class _FakeController:
    """Enough attributes for default_tools() to bind. Methods are never called
    because agent handlers are no-ops this phase."""

    def __getattr__(self, name):  # any attribute resolves to a callable stub
        return lambda *a, **k: None


def _engineer() -> DatasetEngineerAgent:
    return DatasetEngineerAgent(_FakeController())  # type: ignore[arg-type]


# --- registration & tools ---------------------------------------------
def test_agent_registration():
    reg = default_registry()
    assert len(reg) == 9
    assert "PlannerAgent" in reg and "ExportAgent" in reg
    names = {i.name for i in reg.list()}
    assert "DetectionAgent" in names


def test_registry_rejects_duplicate():
    reg = default_registry()
    with pytest.raises(ValueError):
        reg.register(reg.get("PlannerAgent"))


def test_tools_bind_to_controller():
    tools = default_tools(_FakeController())  # type: ignore[arg-type]
    assert "import_images" in tools and "export_dataset" in tools
    assert len(tools) == 10


# --- plan generation ---------------------------------------------------
def test_goal_produces_serializable_plan():
    plan = Planner().plan(new_goal("Create a vehicle detection dataset", source="x"))
    assert [s.id for s in plan.steps][:2] == ["input_analysis", "dataset_inspection"]
    assert any(s.requires_approval for s in plan.steps)  # a human gate exists
    # depends_on forms a chain
    assert plan.steps[1].depends_on == ["input_analysis"]
    # serializable
    assert plan.model_dump_json()


# --- message protocol --------------------------------------------------
def test_message_bus_stamps_and_filters():
    bus = MessageBus()
    bus.post(AgentMessage(sender="a", receiver="b", task="t"))
    bus.post(AgentMessage(sender="a", receiver="c", task="t"))
    assert len(bus) == 2
    assert bus.history(receiver="b")[0].timestamp > 0
    assert len(bus.history(sender="a")) == 2


# --- orchestration + lifecycle ----------------------------------------
def test_runs_to_approval_then_completes():
    eng = _engineer()
    s = eng.submit_goal(new_goal("build a dataset"))
    assert s.status == SessionStatus.AWAITING_APPROVAL
    assert s.current_step == "human_approval"
    # steps before the gate ran
    assert "detection" in s.completed_steps and "human_approval" not in s.completed_steps

    s = eng.approve(s.id, "human_approval")
    assert s.status == SessionStatus.COMPLETED
    assert {"export", "record_memory"} <= set(s.completed_steps)
    assert s.finished_at is not None


def test_pause_blocks_and_resume_continues():
    eng = _engineer()
    s = eng.submit_goal(new_goal("g"))  # parks at approval gate
    eng.pause(s.id)
    assert eng.status(s.id).status == SessionStatus.PAUSED
    # resume from a paused approval-gate session re-evaluates and parks again
    s = eng.resume(s.id)
    assert s.status == SessionStatus.AWAITING_APPROVAL


def test_failure_recovery_then_retry():
    # A registry whose DetectionAgent raises — step FAILED, run halts, retry recovers.
    reg = default_registry()

    class Boom(BaseAgent):
        info = reg.get("DetectionAgent").info
        _armed = True

        def handle(self, message: AgentMessage) -> dict:
            if Boom._armed:
                Boom._armed = False
                raise RuntimeError("gpu oom")
            return {"status": "ok"}

    # swap the DetectionAgent instance
    reg._agents["DetectionAgent"] = Boom()

    bus = MessageBus()
    orch = TaskOrchestrator(reg, ToolRegistry(), bus)
    plan = Planner().plan(new_goal("g"))
    from vds.v2.state import SessionState

    s = SessionState(id="s1", goal=new_goal("g"), plan=plan)
    s = orch.run_ready(s)
    assert s.status == SessionStatus.FAILED
    assert "detection" in s.failed_steps
    assert plan.get("detection").status == StepStatus.FAILED

    s = orch.retry(s, "detection")
    assert s.status == SessionStatus.AWAITING_APPROVAL  # recovered, back at the gate
    assert "detection" in s.completed_steps and "detection" not in s.failed_steps


def test_view_surface():
    eng = _engineer()
    s = eng.submit_goal(new_goal("make a dataset"))
    v = eng.view(s.id)
    assert v["current_agent"] == "ReviewAgent"
    assert len(v["timeline"]) == 11 and len(v["agents"]) == 9
    assert eng.report(s.id).startswith("# Dataset Engineer Report")
