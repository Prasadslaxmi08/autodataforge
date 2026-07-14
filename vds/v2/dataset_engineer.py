"""Dataset Engineer Agent — the master agent (V2-20 §DATASET ENGINEER AGENT).

The one V2 component the GUI talks to. It receives a goal, plans it, opens a
session, and drives the orchestrator; it exposes progress, approval, and a report.
Think of it as the experienced engineer the user hired: the user says *what*, this
agent coordinates *how* — but never executes a major action silently (the plan has
an explicit human-approval gate), and every action is explainable from the session.

Scope this phase: coordination only. No intent parsing, no autonomous choices — the
planner is a fixed template and agent handlers are no-ops.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

from vds.v2.goal import Goal
from vds.v2.orchestrator import TaskOrchestrator
from vds.v2.planner import Planner
from vds.v2.registry import AgentRegistry, default_registry
from vds.v2.state import SessionState, SessionStatus
from vds.v2.tool_registry import default_tools

if TYPE_CHECKING:
    from vds.gui.controller import BackendController

Event = Callable[[str, dict], None]


class DatasetEngineerAgent:
    def __init__(
        self,
        controller: BackendController,
        registry: AgentRegistry | None = None,
        on_event: Event | None = None,
    ) -> None:
        from vds.v2.decision import DecisionAgent
        from vds.v2.execution import ExecutionAgent
        from vds.v2.memory_agent import MemoryAgent
        from vds.v2.messages import MessageBus
        from vds.v2.planner_agent import PlannerAgent

        self._planner = Planner()
        self._planner_agent = PlannerAgent()  # V2-21 intelligent planner (plan-only)
        self._decider = DecisionAgent()  # V2-23 optimization layer
        self._memory = MemoryAgent()  # V2-24 experience recall/store (no tools, no planning)
        self._registry = registry or default_registry()
        self._tools = default_tools(controller)
        self._executor = ExecutionAgent(self._tools, on_event=on_event)  # V2-22
        self._bus = MessageBus()
        self._orch = TaskOrchestrator(self._registry, self._tools, self._bus, on_event)
        self._sessions: dict[str, SessionState] = {}

    # --- lifecycle -----------------------------------------------------
    def submit_goal(self, goal: Goal) -> SessionState:
        """Plan the goal, open a session, and run to the first gate."""
        plan = self._planner.plan(goal)
        session = SessionState(
            id=uuid.uuid4().hex,
            goal=goal,
            plan=plan,
            status=SessionStatus.PLANNING,
            pending_steps=[s.id for s in plan.steps],
            started_at=time.time(),
        )
        self._sessions[session.id] = session
        return self._orch.run_ready(session)

    def status(self, session_id: str) -> SessionState:
        return self._sessions[session_id]

    def approve(self, session_id: str, step_id: str) -> SessionState:
        """Clear a human-approval gate and continue the run."""
        from vds.v2.planner import StepStatus

        session = self._sessions[session_id]
        assert session.plan is not None
        step = session.plan.get(step_id)
        if step is not None and step.status == StepStatus.AWAITING_APPROVAL:
            step.status = StepStatus.DONE
            session.completed_steps.append(step.id)
        return self._orch.run_ready(session)

    def pause(self, session_id: str) -> SessionState:
        return self._orch.pause(self._sessions[session_id])

    def resume(self, session_id: str) -> SessionState:
        return self._orch.resume(self._sessions[session_id])

    def cancel(self, session_id: str) -> SessionState:
        return self._orch.cancel(self._sessions[session_id])

    def retry(self, session_id: str, step_id: str) -> SessionState:
        return self._orch.retry(self._sessions[session_id], step_id)

    # --- planning (V2-21; plan-only, nothing executes) -----------------
    def generate_plan(self, goal: Goal, **context: object):
        """Produce an explainable ExecutionPlan for a goal via the intelligent
        PlannerAgent. Plan-only: no tools run. This is the GUI's 'Generate Plan'
        entry. ``context`` accepts project/dataset/preferences/context (PlanContext)."""
        from vds.v2.recommendations import PlanContext

        ctx = context.pop("context", None)
        if isinstance(ctx, dict):
            ctx = PlanContext(**ctx)
        return self._planner_agent.create_plan(goal, context=ctx, **context)  # type: ignore[arg-type]

    # --- decision (V2-23; optimize a plan before execution) ------------
    def optimize_plan(self, plan, metadata):
        """Refine a plan's execution parameters from real dataset metadata.
        Returns (enriched_plan, DecisionReport). Optimizes only — never executes
        and never changes intent. ``metadata`` is a DatasetMetadata or a dict."""
        from vds.v2.decision import DatasetMetadata

        if isinstance(metadata, dict):
            metadata = DatasetMetadata(**metadata)
        return self._decider.decide(plan, metadata)

    # --- memory (V2-24; recall before planning, record after a run) ----
    def recall_experience(self, goal: Goal, metadata=None, top_k: int = 3):
        """Similar past projects + the settings that worked, before planning.
        Advice only — nothing here plans, executes, or runs a model. ``metadata``
        is a DatasetMetadata or a dict (or None for a goal-only recall)."""
        from vds.v2.decision import DatasetMetadata

        if isinstance(metadata, dict):
            metadata = DatasetMetadata(**metadata)
        return self._memory.recall(goal, metadata, top_k=top_k)

    def record_experience(self, goal: Goal, decision_report, execution_summary, *,
                          project_id: str, metadata=None, plan=None, export_summary=None):
        """After a completed run, persist reusable experience into the shared
        Engineering Memory. Returns the stored EngineeringMemory, or None if the run
        did not complete. The clock is stamped here to keep MemoryAgent deterministic."""
        from datetime import UTC, datetime

        from vds.v2.decision import DatasetMetadata

        if isinstance(metadata, dict):
            metadata = DatasetMetadata(**metadata)
        return self._memory.record(
            goal, decision_report, execution_summary, project_id=project_id,
            created_at=datetime.now(UTC).isoformat(), metadata=metadata,
            plan=plan, export_summary=export_summary)

    @property
    def memory(self):
        """The MemoryAgent — for direct recall/record and GUI memory_view."""
        return self._memory

    # --- execution (V2-22; delegates to ExecutionAgent) ----------------
    def execute_plan(self, plan, inputs: dict | None = None, *, require_approval: bool = True):
        """Execute an approved ExecutionPlan via the ExecutionAgent (tools only).
        Returns the ExecutionContext. The GUI polls ``execution_view``."""
        return self._executor.execute(plan, inputs, require_approval=require_approval)

    def execution_view(self, plan_id: str) -> dict:
        return self._executor.view(plan_id)

    @property
    def executor(self):
        """The ExecutionAgent — for pause/resume/cancel/approve/retry from the GUI."""
        return self._executor

    # --- GUI surface (data only; no Qt this phase) ---------------------
    def view(self, session_id: str) -> dict:
        """Everything the V2 GUI panels bind to: current agent/task, reasoning,
        execution timeline, and agent status. A future phase renders this in Qt."""
        session = self._sessions[session_id]
        steps = session.plan.steps if session.plan else []
        current = session.current_step
        current_step = session.plan.get(current) if (session.plan and current) else None
        return {
            "goal": session.goal.text,
            "status": session.status.value,
            "current_agent": current_step.agent if current_step else None,
            "current_task": current_step.name if current_step else None,
            "reasoning": current_step.task if current_step else "",
            "timeline": [
                {"id": s.id, "name": s.name, "agent": s.agent, "status": s.status.value}
                for s in steps
            ],
            "agents": [
                {"name": i.name, "status": i.status.value} for i in self._registry.list()
            ],
            "errors": list(session.errors),
        }

    def report(self, session_id: str) -> str:
        """A plain-Markdown run summary, from the session's own recorded facts."""
        session = self._sessions[session_id]
        lines = [
            "# Dataset Engineer Report",
            "",
            f"**Goal:** {session.goal.text}",
            f"**Status:** {session.status.value}",
            f"**Steps completed:** {len(session.completed_steps)}"
            + (f" / {len(session.plan.steps)}" if session.plan else ""),
            "",
            "## Timeline",
        ]
        for s in session.plan.steps if session.plan else []:
            lines.append(f"- [{s.status.value}] {s.name} ({s.agent})")
        if session.errors:
            lines += ["", "## Errors", *(f"- {e}" for e in session.errors)]
        return "\n".join(lines)
