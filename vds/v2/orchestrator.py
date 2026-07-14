"""Task orchestrator (V2-20 §TASK ORCHESTRATOR).

Deterministic state machine. It receives a plan on a SessionState, walks ready
steps in dependency order, dispatches each to its assigned agent via the message
bus, records timing and results, and stops cleanly at the human-approval gate. It
makes **no autonomous decisions** — it only advances what the plan already says.

Controls: pause, resume, cancel, retry. Every state change emits an event (for the
GUI timeline) through the optional ``on_event`` callback.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from vds.v2.agent_base import AgentStatus
from vds.v2.messages import AgentMessage, MessageBus, MessageStatus
from vds.v2.planner import PlanStep, StepStatus
from vds.v2.registry import AgentRegistry
from vds.v2.state import SessionState, SessionStatus
from vds.v2.tool_registry import ToolRegistry

Event = Callable[[str, dict], None]


class TaskOrchestrator:
    def __init__(
        self,
        registry: AgentRegistry,
        tools: ToolRegistry,
        bus: MessageBus,
        on_event: Event | None = None,
    ) -> None:
        self._registry = registry
        self._tools = tools
        self._bus = bus
        self._on_event = on_event

    def _emit(self, kind: str, payload: dict) -> None:
        if self._on_event is not None:
            self._on_event(kind, payload)

    def run_ready(self, session: SessionState) -> SessionState:
        """Advance the session as far as the plan allows right now: dispatch every
        ready step until one needs approval, one fails, or the plan is done."""
        if session.plan is None:
            return session
        if session.status in (SessionStatus.PAUSED, SessionStatus.CANCELLED):
            return session

        session.status = SessionStatus.RUNNING
        done = set(session.completed_steps)
        while True:
            ready = session.plan.ready(done)
            if not ready:
                break
            step = ready[0]  # linear plan; deterministic pick keeps runs reproducible
            if step.requires_approval:
                step.status = StepStatus.AWAITING_APPROVAL
                session.current_step = step.id
                session.status = SessionStatus.AWAITING_APPROVAL
                self._emit("awaiting_approval", {"step": step.id})
                return session
            ok = self._dispatch(session, step)
            if not ok:
                session.status = SessionStatus.FAILED
                return session
            done.add(step.id)

        # No ready steps: either parked at an approval gate, or genuinely done.
        awaiting = next(
            (s for s in session.plan.steps if s.status == StepStatus.AWAITING_APPROVAL),
            None,
        )
        if awaiting is not None:
            session.current_step = awaiting.id
            session.status = SessionStatus.AWAITING_APPROVAL
            return session

        session.current_step = None
        session.finished_at = time.time()
        session.status = SessionStatus.COMPLETED
        self._emit("completed", {"session": session.id})
        return session

    def _dispatch(self, session: SessionState, step: PlanStep) -> bool:
        assert session.plan is not None
        agent = self._registry.get(step.agent)
        step.status = StepStatus.RUNNING
        session.current_step = step.id
        agent.info.status = AgentStatus.RUNNING
        self._emit("step_started", {"step": step.id, "agent": step.agent})

        msg = self._bus.post(
            AgentMessage(
                sender="TaskOrchestrator",
                receiver=step.agent,
                task=step.task,
                arguments=step.arguments,
                status=MessageStatus.IN_PROGRESS,
            )
        )
        session.agent_activity.append(msg.id)

        started = time.perf_counter()
        try:
            result = agent.handle(msg)
        except Exception as exc:  # failure recovery: mark FAILED, keep the run resumable
            msg.execution_time_ms = (time.perf_counter() - started) * 1000
            msg.status = MessageStatus.FAILED
            msg.errors.append(str(exc))
            step.status = StepStatus.FAILED
            session.failed_steps.append(step.id)
            session.errors.append(f"{step.id}: {exc}")
            agent.info.status = AgentStatus.READY
            self._emit("step_failed", {"step": step.id, "error": str(exc)})
            return False

        msg.execution_time_ms = (time.perf_counter() - started) * 1000
        msg.status = MessageStatus.DONE
        msg.result = result
        step.status = StepStatus.DONE
        session.completed_steps.append(step.id)
        agent.info.status = AgentStatus.READY
        self._emit("step_done", {"step": step.id})
        return True

    # --- controls ------------------------------------------------------
    def pause(self, session: SessionState) -> SessionState:
        session.status = SessionStatus.PAUSED
        self._emit("paused", {"session": session.id})
        return session

    def resume(self, session: SessionState) -> SessionState:
        if session.status == SessionStatus.PAUSED:
            session.status = SessionStatus.RUNNING
        return self.run_ready(session)

    def cancel(self, session: SessionState) -> SessionState:
        session.status = SessionStatus.CANCELLED
        session.finished_at = time.time()
        self._emit("cancelled", {"session": session.id})
        return session

    def retry(self, session: SessionState, step_id: str) -> SessionState:
        """Reset a failed step to PENDING so a resume re-dispatches it."""
        if session.plan is None:
            return session
        step = session.plan.get(step_id)
        if step is None or step.status != StepStatus.FAILED:
            return session
        step.status = StepStatus.PENDING
        if step_id in session.failed_steps:
            session.failed_steps.remove(step_id)
        session.errors = [e for e in session.errors if not e.startswith(f"{step_id}:")]
        return self.run_ready(session)
