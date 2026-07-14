"""TaskOrchestrator — the single entry point for a dataset-generation task (V2-25).

The GUI used to call PlannerAgent -> MemoryAgent -> DecisionAgent -> ExecutionAgent
-> MemoryAgent by hand. This orchestrator collapses that into one call::

    orch.execute(goal, project)   # runs to the approval gate, then pauses
    orch.approve(task_id)         # runs execution + records memory
    orch.report(task_id)          # final report

It **only coordinates**. Every piece of real work stays in the four existing
agents — reached through the ``DatasetEngineerAgent`` facade, which already wires
them and exposes the coordinated verbs (generate_plan / recall_experience /
optimize_plan / execute_plan / record_experience). No agent internals are touched
and no agent logic is duplicated here: this file is a state machine, an event
stream, a timeline, and the failure policy the brief mandates, nothing more.

Failure policy (brief §FAILURE HANDLING):
  Planner failed      -> abort
  Memory unavailable  -> continue (advice is optional)
  Decision failed     -> use planner defaults
  Execution failed    -> retry, then abort
  Memory save failed  -> warn only
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, Field

from vds.logging import get_logger
from vds.v2.decision import DecisionReport
from vds.v2.execution import ExecutionSummary
from vds.v2.goal import Goal, new_goal
from vds.v2.memory_agent import MemoryExperience
from vds.v2.planner import ExecutionPlan, PlanStatus, StepStatus
from vds.v2.state import SessionStatus

Event = Callable[[str, dict], None]
log = get_logger(__name__)


class TaskState(StrEnum):
    IDLE = "idle"
    PLANNING = "planning"
    MEMORY_RETRIEVAL = "memory_retrieval"
    DECISION_MAKING = "decision_making"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    RECORDING_MEMORY = "recording_memory"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskEvent(StrEnum):
    PLANNING_STARTED = "PlanningStarted"
    PLANNING_COMPLETED = "PlanningCompleted"
    MEMORY_LOADED = "MemoryLoaded"
    DECISION_COMPLETED = "DecisionCompleted"
    APPROVAL_REQUESTED = "ApprovalRequested"
    EXECUTION_STARTED = "ExecutionStarted"
    EXECUTION_COMPLETED = "ExecutionCompleted"
    MEMORY_STORED = "MemoryStored"
    TASK_COMPLETED = "TaskCompleted"
    TASK_FAILED = "TaskFailed"
    TASK_CANCELLED = "TaskCancelled"


# The GUI orchestration timeline (brief §GUI) — one row per stage, in order.
STAGES = ["Planning", "Memory", "Decision", "Approval", "Execution", "Memory Save", "Complete"]


class TaskContext(BaseModel):
    """All state for one orchestrated task. Serializable — the whole run is
    reconstructable from this (goal, plan, decision, memory, summary, logs, timing)."""

    id: str
    goal: Goal
    project: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    inputs: dict = Field(default_factory=dict)
    state: TaskState = TaskState.IDLE

    plan: ExecutionPlan | None = None
    plan_id: str | None = None
    decision_report: DecisionReport | None = None
    memory: MemoryExperience | None = None
    execution_summary: ExecutionSummary | None = None
    stored_memory_id: str | None = None

    stages: dict[str, str] = Field(default_factory=lambda: {s: "pending" for s in STAGES})
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    logs: list[dict] = Field(default_factory=list)  # structured: ts, agent, action, duration_ms, status
    timeline: list[dict] = Field(default_factory=list)  # emitted events, for the live view
    stage_ms: dict[str, float] = Field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float | None = None

    def elapsed_seconds(self) -> float:
        return round((self.finished_at or time.time()) - self.started_at, 3)


class TaskOrchestrator:
    """Coordinates the four agents through the fixed workflow. Owns no agent logic."""

    def __init__(
        self,
        controller=None,
        *,
        engineer=None,
        on_event: Event | None = None,
        max_execution_retries: int = 1,
    ) -> None:
        # The DatasetEngineerAgent is the agent-access layer (the "Agent Coordinator"):
        # it already wires Planner/Memory/Decision/Execution and exposes their verbs.
        if engineer is None:
            from vds.v2.dataset_engineer import DatasetEngineerAgent

            engineer = DatasetEngineerAgent(controller)
        self._engineer = engineer
        self._on_event = on_event
        self._max_exec_retries = max_execution_retries
        self._tasks: dict[str, TaskContext] = {}

    @property
    def coordinator(self):
        """The DatasetEngineerAgent the orchestrator drives (the agent-access layer)."""
        return self._engineer

    # --- entry points --------------------------------------------------
    def execute(self, goal, project: dict | None = None, *, inputs: dict | None = None,
                metadata: dict | None = None, auto_approve: bool = False) -> TaskContext:
        """THE entry point. Plans, recalls memory, decides, then parks at the
        approval gate (or runs straight through when ``auto_approve``). ``project``
        carries project/dataset metadata; ``inputs`` are the execution tool args."""
        g = goal if isinstance(goal, Goal) else new_goal(str(goal))
        project = dict(project or {})
        ctx = TaskContext(
            id=uuid.uuid4().hex, goal=g, project=project,
            metadata=dict(metadata if metadata is not None else project.get("metadata", project)),
            inputs=dict(inputs if inputs is not None else project.get("inputs", {})),
            started_at=time.time(),
        )
        self._tasks[ctx.id] = ctx

        if not self._plan(ctx):
            return self._fail(ctx, "planning failed")
        self._recall(ctx)   # memory unavailable -> continue
        self._decide(ctx)   # decision failed -> planner defaults

        ctx.state = TaskState.AWAITING_APPROVAL
        self._set_stage(ctx, "Approval", "active")
        self._emit(ctx, TaskEvent.APPROVAL_REQUESTED, "approval requested")
        if auto_approve:
            return self._run_execution(ctx)
        return ctx

    def approve(self, task_id: str) -> TaskContext:
        """Clear the approval gate and run execution + memory recording."""
        ctx = self._tasks[task_id]
        if ctx.state != TaskState.AWAITING_APPROVAL:
            return ctx
        self._set_stage(ctx, "Approval", "done")
        return self._run_execution(ctx)

    def cancel(self, task_id: str) -> TaskContext:
        ctx = self._tasks[task_id]
        if ctx.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
            return ctx
        if ctx.plan_id is not None:
            try:
                self._engineer.executor.cancel(ctx.plan_id)
            except Exception:  # nothing running yet — cancel is best-effort
                pass
        ctx.state = TaskState.CANCELLED
        ctx.finished_at = time.time()
        self._emit(ctx, TaskEvent.TASK_CANCELLED, "cancelled")
        return ctx

    def status(self, task_id: str) -> TaskContext:
        return self._tasks[task_id]

    # --- stages --------------------------------------------------------
    def _plan(self, ctx: TaskContext) -> bool:
        ctx.state = TaskState.PLANNING
        self._set_stage(ctx, "Planning", "active")
        self._emit(ctx, TaskEvent.PLANNING_STARTED)
        t0 = time.perf_counter()
        try:
            ctx.plan = self._engineer.generate_plan(ctx.goal, context=self._plan_context(ctx))
        except Exception as exc:
            self._finish_stage(ctx, "Planning", "PlannerAgent", "plan", "failed", t0, str(exc))
            ctx.errors.append(f"planner: {exc}")
            self._set_stage(ctx, "Planning", "failed")
            return False
        self._finish_stage(ctx, "Planning", "PlannerAgent", "plan", "ok", t0,
                           f"{len(ctx.plan.steps)} steps")
        self._emit(ctx, TaskEvent.PLANNING_COMPLETED, ctx.plan.summary)
        return True

    def _recall(self, ctx: TaskContext) -> None:
        ctx.state = TaskState.MEMORY_RETRIEVAL
        self._set_stage(ctx, "Memory", "active")
        t0 = time.perf_counter()
        try:
            ctx.memory = self._engineer.recall_experience(ctx.goal, ctx.metadata or None)
            self._finish_stage(ctx, "Memory", "MemoryAgent", "recall", "ok", t0,
                               f"{len(ctx.memory.matches)} matches")
            self._emit(ctx, TaskEvent.MEMORY_LOADED,
                       f"{len(ctx.memory.matches)} similar project(s)")
        except Exception as exc:  # memory unavailable -> continue without it
            ctx.warnings.append(f"memory unavailable: {exc}")
            self._finish_stage(ctx, "Memory", "MemoryAgent", "recall", "skipped", t0, str(exc))
            self._set_stage(ctx, "Memory", "skipped")

    def _decide(self, ctx: TaskContext) -> None:
        ctx.state = TaskState.DECISION_MAKING
        self._set_stage(ctx, "Decision", "active")
        t0 = time.perf_counter()
        try:
            enriched, report = self._engineer.optimize_plan(ctx.plan, ctx.metadata or {})
            ctx.plan, ctx.decision_report = enriched, report
            self._finish_stage(ctx, "Decision", "DecisionAgent", "decide", "ok", t0,
                               f"{len(report.decisions)} decisions")
            self._emit(ctx, TaskEvent.DECISION_COMPLETED,
                       f"confidence {report.overall_confidence}")
        except Exception as exc:  # decision failed -> keep planner defaults
            ctx.warnings.append(f"decision failed; using planner defaults: {exc}")
            self._finish_stage(ctx, "Decision", "DecisionAgent", "decide", "skipped", t0, str(exc))
            self._set_stage(ctx, "Decision", "skipped")

    def _run_execution(self, ctx: TaskContext) -> TaskContext:
        ctx.state = TaskState.EXECUTING
        self._set_stage(ctx, "Execution", "active")
        self._emit(ctx, TaskEvent.EXECUTION_STARTED)
        ctx.plan.status = PlanStatus.APPROVED  # user approved at the orchestrator gate
        ctx.plan_id = ctx.plan.id
        t0 = time.perf_counter()
        try:
            exec_ctx = self._engineer.execute_plan(ctx.plan, ctx.inputs, require_approval=True)
            exec_ctx = self._drive(ctx, exec_ctx)
        except Exception as exc:
            self._finish_stage(ctx, "Execution", "ExecutionAgent", "execute", "failed", t0, str(exc))
            ctx.errors.append(f"execution: {exc}")
            self._set_stage(ctx, "Execution", "failed")
            return self._fail(ctx, "execution failed")

        ctx.execution_summary = self._engineer.executor.summary(ctx.plan_id)
        if ctx.execution_summary.status != SessionStatus.COMPLETED.value:
            self._finish_stage(ctx, "Execution", "ExecutionAgent", "execute", "failed", t0,
                               ctx.execution_summary.status)
            ctx.errors.extend(ctx.execution_summary.errors or [ctx.execution_summary.status])
            self._set_stage(ctx, "Execution", "failed")
            return self._fail(ctx, "execution failed after retries")

        self._finish_stage(ctx, "Execution", "ExecutionAgent", "execute", "ok", t0,
                           f"{ctx.execution_summary.completed} steps")
        self._emit(ctx, TaskEvent.EXECUTION_COMPLETED)
        self._record(ctx)   # memory save failed -> warn only

        ctx.state = TaskState.COMPLETED
        ctx.finished_at = time.time()
        self._set_stage(ctx, "Complete", "done")
        self._emit(ctx, TaskEvent.TASK_COMPLETED)
        return ctx

    def _drive(self, ctx: TaskContext, exec_ctx):
        """Run the executor to a terminal state. The human already approved at the
        orchestrator gate, so any of the executor's own approval gates are cleared
        automatically. A FAILED run is retried up to ``max_execution_retries``, then
        left FAILED (the caller aborts)."""
        exec_ctx = self._clear_gates(ctx, exec_ctx)
        retries = 0
        while exec_ctx.status == SessionStatus.FAILED and retries < self._max_exec_retries:
            retries += 1
            ctx.warnings.append(f"execution retry {retries}/{self._max_exec_retries}")
            for sid, st in list(exec_ctx.step_status.items()):
                if st == StepStatus.FAILED:
                    exec_ctx = self._engineer.executor.retry(ctx.plan_id, sid)
            exec_ctx = self._clear_gates(ctx, exec_ctx)
        return exec_ctx

    def _clear_gates(self, ctx: TaskContext, exec_ctx):
        guard = 0
        while exec_ctx.status == SessionStatus.AWAITING_APPROVAL and guard < 50:
            exec_ctx = self._engineer.executor.approve(ctx.plan_id)
            guard += 1
        return exec_ctx

    def _record(self, ctx: TaskContext) -> None:
        ctx.state = TaskState.RECORDING_MEMORY
        self._set_stage(ctx, "Memory Save", "active")
        t0 = time.perf_counter()
        if ctx.decision_report is None:  # nothing structured to record from
            ctx.warnings.append("no decision report; skipping memory record")
            self._finish_stage(ctx, "Memory Save", "MemoryAgent", "record", "skipped", t0)
            self._set_stage(ctx, "Memory Save", "skipped")
            return
        try:
            mem = self._engineer.record_experience(
                ctx.goal, ctx.decision_report, ctx.execution_summary,
                project_id=self._project_id(ctx), metadata=ctx.metadata or None,
                plan=ctx.plan, export_summary=ctx.project.get("export_summary"))
            if mem is not None:
                ctx.stored_memory_id = mem.id
                self._finish_stage(ctx, "Memory Save", "MemoryAgent", "record", "ok", t0, mem.id)
                self._emit(ctx, TaskEvent.MEMORY_STORED, mem.id)
            else:
                ctx.warnings.append("run not recorded (incomplete)")
                self._finish_stage(ctx, "Memory Save", "MemoryAgent", "record", "skipped", t0)
                self._set_stage(ctx, "Memory Save", "skipped")
        except Exception as exc:  # memory save failed -> warn only, never fail the task
            ctx.warnings.append(f"memory save failed: {exc}")
            self._finish_stage(ctx, "Memory Save", "MemoryAgent", "record", "failed", t0, str(exc))
            self._set_stage(ctx, "Memory Save", "failed")

    def _fail(self, ctx: TaskContext, reason: str) -> TaskContext:
        ctx.state = TaskState.FAILED
        ctx.finished_at = time.time()
        self._emit(ctx, TaskEvent.TASK_FAILED, reason)
        return ctx

    # --- report / view -------------------------------------------------
    def report(self, task_id: str) -> dict:
        """Final report (brief §REPORT): execution summary, decision report, memory
        matches, warnings, timing, statistics, artifacts."""
        ctx = self._tasks[task_id]
        artifacts = {}
        if ctx.plan_id is not None:
            try:
                v = self._engineer.execution_view(ctx.plan_id)
                artifacts = {"dataset": v.get("active_dataset"), "export": v.get("current_export")}
            except Exception:
                pass
        s = ctx.execution_summary
        return {
            "task_id": ctx.id,
            "state": ctx.state.value,
            "goal": ctx.goal.text,
            "execution_summary": s.model_dump() if s else None,
            "decision_report": ctx.decision_report.model_dump() if ctx.decision_report else None,
            "memory_matches": [
                {"id": m.memory.id, "similarity": m.score} for m in (ctx.memory.matches if ctx.memory else [])
            ],
            "warnings": list(ctx.warnings),
            "errors": list(ctx.errors),
            "timing": {"elapsed_seconds": ctx.elapsed_seconds(), "stages_ms": dict(ctx.stage_ms)},
            "statistics": {
                "steps_total": s.total if s else 0,
                "steps_completed": s.completed if s else 0,
                "steps_failed": s.failed if s else 0,
                "steps_retried": s.retried if s else 0,
                "memory_stored": ctx.stored_memory_id is not None,
            },
            "artifacts": artifacts,
        }

    def view(self, task_id: str) -> dict:
        """GUI orchestration surface (brief §GUI): the live stage timeline + context."""
        ctx = self._tasks[task_id]
        return {
            "task_id": ctx.id,
            "goal": ctx.goal.text,
            "state": ctx.state.value,
            "stages": [{"name": s, "status": ctx.stages[s]} for s in STAGES],
            "events": list(ctx.timeline),
            "logs": list(ctx.logs),
            "warnings": list(ctx.warnings),
            "errors": list(ctx.errors),
            "elapsed_seconds": ctx.elapsed_seconds(),
        }

    # --- helpers -------------------------------------------------------
    @staticmethod
    def _project_id(ctx: TaskContext) -> str:
        p = ctx.project
        return str(p.get("project_id") or p.get("id") or p.get("name") or ctx.goal.id)

    @staticmethod
    def _plan_context(ctx: TaskContext) -> dict:
        """Map known metadata onto the PlannerAgent's PlanContext fields."""
        m = ctx.metadata
        keys = ("image_count", "video_duration_seconds", "fps", "resolution",
                "expected_density", "small_objects")
        return {k: m[k] for k in keys if k in m}

    def _set_stage(self, ctx: TaskContext, stage: str, status: str) -> None:
        ctx.stages[stage] = status

    def _finish_stage(self, ctx: TaskContext, stage: str, agent: str, action: str,
                      status: str, t0: float, message: str = "") -> None:
        ms = round((time.perf_counter() - t0) * 1000, 2)
        ctx.stage_ms[stage] = ms
        if status == "ok":
            ctx.stages[stage] = "done"
        ctx.logs.append({"ts": time.time(), "agent": agent, "action": action,
                         "duration_ms": ms, "status": status, "message": message})
        log.info("orchestrator.stage", task=ctx.id, agent=agent, action=action,
                 status=status, duration_ms=ms)

    def _emit(self, ctx: TaskContext, event: TaskEvent, message: str = "") -> None:
        ctx.timeline.append({"ts": time.time(), "event": event.value,
                             "state": ctx.state.value, "message": message})
        if self._on_event is not None:
            self._on_event(event.value, {"task": ctx.id, "state": ctx.state.value,
                                         "event": event.value, "message": message})


def task_view(ctx: TaskContext) -> dict:
    """Standalone GUI surface for a TaskContext (mirrors decision_view/memory_view)."""
    return {
        "task_id": ctx.id,
        "goal": ctx.goal.text,
        "state": ctx.state.value,
        "stages": [{"name": s, "status": ctx.stages[s]} for s in STAGES],
        "events": list(ctx.timeline),
        "warnings": list(ctx.warnings),
        "errors": list(ctx.errors),
        "elapsed_seconds": ctx.elapsed_seconds(),
    }
