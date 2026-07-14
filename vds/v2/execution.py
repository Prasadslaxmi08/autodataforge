"""ExecutionAgent — runs an approved ExecutionPlan (V2-22).

Given an approved ``ExecutionPlan`` (from the V2-21 PlannerAgent) it executes every
step by invoking the existing ``BackendController`` tools (via the V2-20
``ToolRegistry``). It **only coordinates**: no detection, annotation, verification,
review, or export logic lives here — that stays in the frozen backend. It also never
plans, reasons, or **modifies the Planner's output**: execution state is tracked in a
separate ``ExecutionContext``; the plan is read-only.

Pieces (V2-22 deliverables): ``ExecutionContext`` (runtime state), ``ExecutionRunner``
(the step loop), ``ProgressTracker``, ``ApprovalHandler``, ``RecoveryHandler``,
``ExecutionTimeline``, and ``ExecutionSummary``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, Field

from vds.v2.planner import ExecutionPlan, PlanStatus, PlanStep, StepStatus
from vds.v2.state import SessionStatus
from vds.v2.tool_registry import ToolRegistry

Event = Callable[[str, dict], None]

# Plan-step task -> BackendController tool name. Tasks with no entry are planning /
# approval steps and are skipped during execution (the Agent never plans).
_TASK_TOOL: dict[str, str] = {
    "import_images": "import_images",
    "import_video": "import_video",
    "extract_frames": "extract_frames",
    "run_detection": "run_detection",
    "run_segmentation": "run_segmentation",
    "review_dataset": "review_dataset",  # Run Verification / Quality Review
    "export_dataset": "export_dataset",
    "generate_report": "generate_report",
    "inspect_dataset": "open_project",
}
_LARGE_DATASET = 2000


class GateReason(StrEnum):
    HUMAN_REVIEW = "human_review"
    LOW_CONFIDENCE = "low_confidence"
    LARGE_DATASET = "large_dataset"
    EXPORT_CONFIRMATION = "export_confirmation"
    DELETION = "deletion"


class FailureCategory(StrEnum):
    MODEL_MISSING = "model_missing"
    FOLDER_UNAVAILABLE = "folder_unavailable"
    VIDEO_DECODE = "video_decode"
    EXPORT_FAILURE = "export_failure"
    DISK_FULL = "disk_full"
    FATAL = "fatal"


class ExecutionError(RuntimeError):
    pass


class ExecutionContext(BaseModel):
    """All runtime state for one execution. Serializable; the plan is referenced by
    id only — execution never mutates the plan."""

    plan_id: str
    status: SessionStatus = SessionStatus.RUNNING
    current_step: str | None = None
    step_status: dict[str, StepStatus] = Field(default_factory=dict)
    attempts: dict[str, int] = Field(default_factory=dict)
    results: dict[str, str] = Field(default_factory=dict)  # step_id -> short summary
    approved: list[str] = Field(default_factory=list)
    gate_reason: str | None = None
    progress: float = 0.0
    completed: int = 0
    total: int = 0
    started_at: float = 0.0
    finished_at: float | None = None
    active_dataset: str | None = None
    current_image: str | None = None
    current_video: str | None = None
    current_export: str | None = None
    inputs: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    timeline: list[dict] = Field(default_factory=list)

    def elapsed_seconds(self) -> float:
        return round((self.finished_at or time.time()) - self.started_at, 3)


class ExecutionSummary(BaseModel):
    plan_id: str
    status: str
    total: int
    completed: int
    skipped: int
    failed: int
    cancelled: int
    retried: int
    elapsed_seconds: float
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ExecutionTimeline:
    """Append-only execution log (for the GUI timeline / live updates)."""

    @staticmethod
    def record(ctx: ExecutionContext, step_id: str | None, status: str, message: str = "") -> None:
        ctx.timeline.append({"ts": time.time(), "step": step_id or "", "status": status, "message": message})


class ProgressTracker:
    """Derives progress from step statuses. No side effects on the plan."""

    _TERMINAL = {StepStatus.DONE, StepStatus.SKIPPED, StepStatus.CANCELLED, StepStatus.FAILED}

    def update(self, ctx: ExecutionContext) -> None:
        done = sum(1 for s in ctx.step_status.values() if s in (StepStatus.DONE, StepStatus.SKIPPED))
        ctx.completed = done
        ctx.progress = round(done / ctx.total, 4) if ctx.total else 0.0


class ApprovalHandler:
    """Decides when execution must pause for a human (V2-22 §APPROVAL GATES)."""

    def __init__(self, gates: set[GateReason] | None = None) -> None:
        # Default: only the plan's explicit Manual-Review gate.
        self.gates = gates if gates is not None else {GateReason.HUMAN_REVIEW}

    def gate_for(self, step: PlanStep, plan: ExecutionPlan, ctx: ExecutionContext) -> GateReason | None:
        if step.id in ctx.approved:
            return None
        if step.requires_approval and GateReason.HUMAN_REVIEW in self.gates:
            return GateReason.HUMAN_REVIEW
        if (GateReason.LARGE_DATASET in self.gates
                and step.task in ("import_images", "import_video")
                and plan.estimated_dataset_size > _LARGE_DATASET):
            return GateReason.LARGE_DATASET
        if GateReason.EXPORT_CONFIRMATION in self.gates and step.task == "export_dataset":
            return GateReason.EXPORT_CONFIRMATION
        if GateReason.DELETION in self.gates and step.task in ("archive_project", "delete_project"):
            return GateReason.DELETION
        return None


class RecoveryHandler:
    """Classifies a step failure as recoverable or fatal (V2-22 §RECOVERY)."""

    _KEYWORDS: list[tuple[FailureCategory, tuple[str, ...]]] = [
        (FailureCategory.DISK_FULL, ("no space", "disk full", "enospc")),
        (FailureCategory.MODEL_MISSING, ("model", "weights", "checkpoint")),
        (FailureCategory.VIDEO_DECODE, ("decode", "codec", "ffmpeg", "ffprobe")),
        (FailureCategory.FOLDER_UNAVAILABLE, ("folder", "not found", "no such file", "unavailable", "directory")),
        (FailureCategory.EXPORT_FAILURE, ("export",)),
    ]

    def classify(self, exc: Exception) -> tuple[FailureCategory, bool]:
        if isinstance(exc, FileNotFoundError):
            return FailureCategory.FOLDER_UNAVAILABLE, True
        msg = str(exc).lower()
        for category, keys in self._KEYWORDS:
            if any(k in msg for k in keys):
                return category, True
        return FailureCategory.FATAL, False


def _summarize(result: object) -> str:
    if result is None:
        return "done"
    for attr in ("imported", "detections", "exported"):
        if hasattr(result, attr):
            return f"{attr}={getattr(result, attr)}"
    text = str(result)
    return text[:120] if text else "done"


class ExecutionRunner:
    """The step loop. Executes ready steps by invoking tools; pauses at gates; retries
    recoverable failures. Reads step readiness from ``ctx`` — never from plan status."""

    def __init__(self, tools: ToolRegistry, approval: ApprovalHandler, recovery: RecoveryHandler,
                 progress: ProgressTracker, max_retries: int, emit: Event) -> None:
        self._tools = tools
        self._approval = approval
        self._recovery = recovery
        self._progress = progress
        self._max_retries = max_retries
        self._emit = emit

    def run(self, ctx: ExecutionContext, plan: ExecutionPlan) -> ExecutionContext:
        if ctx.status in (SessionStatus.PAUSED, SessionStatus.CANCELLED):
            return ctx
        ctx.status = SessionStatus.RUNNING
        while True:
            ready = self._ready(plan, ctx)
            if not ready:
                break
            step = ready[0]
            reason = self._approval.gate_for(step, plan, ctx)
            if reason is not None:
                ctx.step_status[step.id] = StepStatus.AWAITING_APPROVAL
                ctx.current_step = step.id
                ctx.gate_reason = reason.value
                ctx.status = SessionStatus.AWAITING_APPROVAL
                ExecutionTimeline.record(ctx, step.id, "awaiting_approval", reason.value)
                self._emit("awaiting_approval", {"step": step.id, "reason": reason.value})
                return ctx
            if not self._run_step(ctx, step):
                ctx.status = SessionStatus.FAILED
                ctx.finished_at = time.time()
                return ctx
        # No ready steps: either parked at an approval gate, or genuinely done.
        awaiting = next((s for s in plan.steps
                         if ctx.step_status.get(s.id) == StepStatus.AWAITING_APPROVAL), None)
        if awaiting is not None:
            ctx.current_step = awaiting.id
            ctx.status = SessionStatus.AWAITING_APPROVAL
            return ctx
        ctx.current_step = None
        ctx.finished_at = time.time()
        ctx.status = SessionStatus.COMPLETED
        self._progress.update(ctx)
        self._emit("completed", {"plan": ctx.plan_id})
        return ctx

    def _ready(self, plan: ExecutionPlan, ctx: ExecutionContext) -> list[PlanStep]:
        done = {sid for sid, st in ctx.step_status.items()
                if st in (StepStatus.DONE, StepStatus.SKIPPED)}
        return [s for s in plan.steps
                if ctx.step_status.get(s.id, StepStatus.PENDING) == StepStatus.PENDING
                and set(s.depends_on) <= done]

    def _run_step(self, ctx: ExecutionContext, step: PlanStep) -> bool:
        ctx.current_step = step.id
        self._set_current(ctx, step)
        tool = _TASK_TOOL.get(step.task)
        if tool is None:  # planning / approval step — the Agent never runs logic itself
            status = StepStatus.DONE if step.requires_approval else StepStatus.SKIPPED
            ctx.step_status[step.id] = status
            ExecutionTimeline.record(ctx, step.id, status.value, "no tool")
            self._progress.update(ctx)
            return True

        ctx.step_status[step.id] = StepStatus.RUNNING
        ExecutionTimeline.record(ctx, step.id, "running")
        self._emit("step_started", {"step": step.id, "tool": tool})
        runner = self._tools.get(tool).run  # call .run directly: invoke() reserves `name`
        while True:
            try:
                result = runner(**self._build_args(step, ctx))
            except Exception as exc:
                category, recoverable = self._recovery.classify(exc)
                attempts = ctx.attempts.get(step.id, 0) + 1
                ctx.attempts[step.id] = attempts
                if recoverable and attempts <= self._max_retries:
                    ctx.step_status[step.id] = StepStatus.RETRYING
                    ctx.warnings.append(f"{step.id}: retry {attempts}/{self._max_retries} ({category.value})")
                    ExecutionTimeline.record(ctx, step.id, "retrying", f"{category.value}: {exc}")
                    self._emit("step_retry", {"step": step.id, "category": category.value})
                    continue
                ctx.step_status[step.id] = StepStatus.FAILED
                ctx.errors.append(f"{step.id}: {exc}")
                ExecutionTimeline.record(ctx, step.id, "failed", f"{category.value}: {exc}")
                self._emit("step_failed", {"step": step.id, "category": category.value})
                return False
            # Import steps create the project under a fresh id (pipeline.run); rebind the
            # downstream project key to it so quality/export/report target the real data.
            if step.task in ("import_video", "import_images"):
                rep = result[0] if isinstance(result, tuple) else result
                pid = getattr(rep, "project_id", None)
                if pid:
                    ctx.inputs["project_id"] = pid
                    ctx.active_dataset = pid
                if rep is not None:  # feed the run report to the generate_report step
                    ctx.inputs["report"] = rep
            ctx.results[step.id] = _summarize(result)
            ctx.step_status[step.id] = StepStatus.DONE
            ExecutionTimeline.record(ctx, step.id, "done", ctx.results[step.id])
            self._progress.update(ctx)
            self._emit("step_done", {"step": step.id})
            return True

    @staticmethod
    def _set_current(ctx: ExecutionContext, step: PlanStep) -> None:
        inp = ctx.inputs
        if ctx.active_dataset is None:
            ctx.active_dataset = inp.get("name") or inp.get("project_id")
        if step.task == "import_video":
            ctx.current_video = inp.get("source")
        elif step.task in ("import_images", "extract_frames", "run_detection", "run_segmentation"):
            ctx.current_image = inp.get("image_id") or ctx.current_image
        elif step.task == "export_dataset":
            ctx.current_export = step.arguments.get("format", "coco")

    @staticmethod
    def _build_args(step: PlanStep, ctx: ExecutionContext) -> dict:
        inp = ctx.inputs
        project = inp.get("project_id") or ctx.active_dataset
        return {
            "import_images": lambda: {"source": inp.get("source"), "name": inp.get("name")},
            "import_video": lambda: {"video_path": inp.get("source"), "name": inp.get("name"),
                                     "config": inp.get("config"), "dedup": inp.get("dedup", True),
                                     "export_format": inp.get("export_format", "coco")},
            "extract_frames": lambda: {"path": inp.get("source")},
            "run_detection": lambda: {"image_id": inp.get("image_id") or ctx.current_image},
            "run_segmentation": lambda: {"image_id": inp.get("image_id") or ctx.current_image,
                                         "box": inp.get("box")},
            "review_dataset": lambda: {"project_id": project},
            "export_dataset": lambda: {"project_id": project, "fmt": step.arguments.get("format", "coco")},
            "generate_report": lambda: {"report": inp.get("report")},
            "inspect_dataset": lambda: {"project_id": project},
        }[step.task]()


class ExecutionAgent:
    """Facade: receive an approved plan, execute it, expose controls + summary.

    Holds active runs keyed by plan id so the GUI can poll/drive them live.
    """

    def __init__(self, tools: ToolRegistry, *, on_event: Event | None = None,
                 max_retries: int = 2, gates: set[GateReason] | None = None) -> None:
        self._on_event = on_event
        self._progress = ProgressTracker()
        self._runner = ExecutionRunner(
            tools, ApprovalHandler(gates), RecoveryHandler(), self._progress, max_retries, self._emit)
        self._runs: dict[str, tuple[ExecutionContext, ExecutionPlan]] = {}

    def _emit(self, kind: str, payload: dict) -> None:
        if self._on_event is not None:
            self._on_event(kind, payload)

    # --- lifecycle -----------------------------------------------------
    def execute(self, plan: ExecutionPlan, inputs: dict | None = None,
                *, require_approval: bool = True) -> ExecutionContext:
        if require_approval and plan.status != PlanStatus.APPROVED:
            raise ExecutionError(f"plan {plan.id} is not approved (status={plan.status.value})")
        ctx = ExecutionContext(plan_id=plan.id, total=len(plan.steps),
                               inputs=dict(inputs or {}), started_at=time.time())
        self._runs[plan.id] = (ctx, plan)
        return self._runner.run(ctx, plan)

    def _run(self, plan_id: str) -> tuple[ExecutionContext, ExecutionPlan]:
        return self._runs[plan_id]

    def context(self, plan_id: str) -> ExecutionContext:
        return self._runs[plan_id][0]

    def pause(self, plan_id: str) -> ExecutionContext:
        ctx = self.context(plan_id)
        ctx.status = SessionStatus.PAUSED
        ExecutionTimeline.record(ctx, ctx.current_step, "paused")
        self._emit("paused", {"plan": plan_id})
        return ctx

    def resume(self, plan_id: str) -> ExecutionContext:
        ctx, plan = self._run(plan_id)
        if ctx.status == SessionStatus.PAUSED:
            ctx.status = SessionStatus.RUNNING
        return self._runner.run(ctx, plan)

    def approve(self, plan_id: str, step_id: str | None = None) -> ExecutionContext:
        """Clear the current approval gate (or a named step) and continue."""
        ctx, plan = self._run(plan_id)
        target = step_id or ctx.current_step
        if target is not None and target not in ctx.approved:
            ctx.approved.append(target)
            ctx.step_status.pop(target, None)  # was AWAITING_APPROVAL -> let the runner re-drive it
        ctx.gate_reason = None
        ctx.status = SessionStatus.RUNNING
        return self._runner.run(ctx, plan)

    def cancel(self, plan_id: str) -> ExecutionContext:
        ctx, plan = self._run(plan_id)
        for s in plan.steps:
            if ctx.step_status.get(s.id, StepStatus.PENDING) in (StepStatus.PENDING, StepStatus.AWAITING_APPROVAL):
                ctx.step_status[s.id] = StepStatus.CANCELLED
        ctx.status = SessionStatus.CANCELLED
        ctx.finished_at = time.time()
        ExecutionTimeline.record(ctx, ctx.current_step, "cancelled")
        self._emit("cancelled", {"plan": plan_id})
        return ctx

    def retry(self, plan_id: str, step_id: str) -> ExecutionContext:
        """Reset a failed step and resume."""
        ctx, plan = self._run(plan_id)
        if ctx.step_status.get(step_id) != StepStatus.FAILED:
            return ctx
        ctx.step_status.pop(step_id, None)
        ctx.attempts.pop(step_id, None)
        ctx.errors = [e for e in ctx.errors if not e.startswith(f"{step_id}:")]
        ctx.status = SessionStatus.RUNNING
        return self._runner.run(ctx, plan)

    # --- reporting -----------------------------------------------------
    def summary(self, plan_id: str) -> ExecutionSummary:
        ctx = self.context(plan_id)
        counts = {st: sum(1 for v in ctx.step_status.values() if v == st) for st in StepStatus}
        return ExecutionSummary(
            plan_id=plan_id, status=ctx.status.value, total=ctx.total,
            completed=counts[StepStatus.DONE], skipped=counts[StepStatus.SKIPPED],
            failed=counts[StepStatus.FAILED], cancelled=counts[StepStatus.CANCELLED],
            retried=sum(1 for a in ctx.attempts.values() if a),
            elapsed_seconds=ctx.elapsed_seconds(), warnings=list(ctx.warnings), errors=list(ctx.errors))

    def view(self, plan_id: str) -> dict:
        """GUI data surface (V2-22 §GUI): live execution progress, steps, log."""
        ctx, plan = self._run(plan_id)
        current = plan.get(ctx.current_step) if ctx.current_step else None
        upcoming = [s.name for s in plan.steps
                    if ctx.step_status.get(s.id, StepStatus.PENDING) == StepStatus.PENDING]
        return {
            "status": ctx.status.value,
            "progress": ctx.progress,
            "completed": ctx.completed,
            "total": ctx.total,
            "elapsed_seconds": ctx.elapsed_seconds(),
            "current_step": current.name if current else None,
            "gate_reason": ctx.gate_reason,
            "upcoming": upcoming,
            "steps": [
                {"id": s.id, "title": s.name,
                 "status": ctx.step_status.get(s.id, StepStatus.PENDING).value}
                for s in plan.steps
            ],
            "log": list(ctx.timeline),
            "warnings": list(ctx.warnings),
            "errors": list(ctx.errors),
            "active_dataset": ctx.active_dataset,
            "current_export": ctx.current_export,
        }

    def report(self, plan_id: str) -> str:
        s = self.summary(plan_id)
        lines = [
            "# Execution Summary", "",
            f"**Plan:** {s.plan_id}", f"**Status:** {s.status}",
            f"**Steps:** {s.completed} done, {s.skipped} skipped, {s.failed} failed, "
            f"{s.cancelled} cancelled of {s.total}",
            f"**Elapsed:** {s.elapsed_seconds}s",
        ]
        if s.warnings:
            lines += ["", "## Warnings", *(f"- {w}" for w in s.warnings)]
        if s.errors:
            lines += ["", "## Errors", *(f"- {e}" for e in s.errors)]
        return "\n".join(lines)
