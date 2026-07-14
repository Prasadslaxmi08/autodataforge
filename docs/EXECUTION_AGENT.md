# ExecutionAgent (V2-22)

Runs an **approved** `ExecutionPlan` (from the V2-21 PlannerAgent) by invoking the
existing `BackendController` tools (the V2-20 `ToolRegistry`). It **only coordinates**:
no detection, annotation, verification, review, or export logic lives here — that
stays in the frozen backend. It also never plans, reasons, chooses models, or
**modifies the Planner's output**. Execution state is tracked in a separate
`ExecutionContext`; the plan is read-only.

Deliverables: `ExecutionAgent` (facade), `ExecutionRunner` (step loop),
`ExecutionContext`, `ProgressTracker`, `ApprovalHandler`, `RecoveryHandler`,
`ExecutionTimeline`, `ExecutionSummary`.

## Execution lifecycle

```
approved ExecutionPlan ─▶ ExecutionAgent.execute(plan, inputs)
        │  (prereq check: plan.status == APPROVED)
        ▼
   ExecutionContext (status=RUNNING, step_status={}, timeline=[])
        │
        ▼
   ExecutionRunner loop ── for each ready step (deps satisfied, per ctx):
        ├─ approval gate?  ─▶ pause (AWAITING_APPROVAL), return
        ├─ no tool (plan/approval step) ─▶ SKIPPED / DONE
        └─ tool step ─▶ BackendController tool ─▶ DONE | (retry) | FAILED
        ▼
   COMPLETED | FAILED | (paused at gate)  ─▶ ExecutionSummary
```

The GUI polls `view(plan_id)` for live progress; controls are `pause`, `resume`,
`approve`, `cancel`, `retry`.

## State machine

Per-step status (`StepStatus`, tracked in `ExecutionContext.step_status` — never on
the plan): `pending → running → done`, or `skipped` (plan/approval steps),
`awaiting_approval` (gate), `retrying` (recoverable failure), `failed`, `cancelled`.

Run status (`SessionStatus`, reused from V2-20): `running`, `paused`,
`awaiting_approval`, `completed`, `failed`, `cancelled`.

Step readiness is computed from `ctx.step_status` + `depends_on`, so the plan object
is never mutated (satisfying "never modify Planner output").

## Tool mapping

Each plan step's `task` maps to a `ToolRegistry` tool, invoked with arguments resolved
from `ExecutionContext.inputs` + the step's own arguments. No business logic — only
argument marshaling.

| Step task | Tool | BackendController method |
|-----------|------|--------------------------|
| import_images | import_images | `import_dataset` |
| import_video | import_video | `import_video_dataset` |
| extract_frames | extract_frames | `probe_video` |
| run_detection | run_detection | `ai_annotate` |
| run_segmentation | run_segmentation | `resegment` |
| review_dataset | review_dataset | `object_verdicts` |
| export_dataset | export_dataset | `export_project` |
| generate_report | generate_report | `report_markdown` |
| inspect_dataset | open_project | `dataset_detail` |

Tasks with **no** tool (`analyze_input`, `select_models`, `await_approval`,
`archive_project`) are planning/approval steps: they are skipped during execution
(the Agent never runs logic itself) — except `await_approval`, which is an approval
gate.

## Failure recovery

`RecoveryHandler.classify(exc)` labels a failure and decides if it is recoverable:

| Category | Recoverable | Detected by |
|----------|-------------|-------------|
| folder_unavailable | yes | `FileNotFoundError`; "folder / not found / no such file / unavailable" |
| model_missing | yes | "model / weights / checkpoint" |
| video_decode | yes | "decode / codec / ffmpeg / ffprobe" |
| export_failure | yes | "export" |
| disk_full | yes | "no space / disk full / enospc" |
| fatal | no | anything else |

Recoverable failures retry up to `max_retries` (status `retrying`); on exhaustion the
step is `failed` and the run halts. `retry(plan_id, step_id)` resets a failed step and
resumes — the manual recovery path after fixing the cause.

## Approval workflow

`ApprovalHandler` pauses the run when a gate applies. Gate reasons: `human_review`
(the plan's Manual Review step — the default), `large_dataset`, `export_confirmation`,
`deletion`, `low_confidence`. Enable extra gates via the `gates=` argument. Execution
pauses (`awaiting_approval`) until `approve(plan_id[, step_id])`, then resumes — a
gated tool step still executes after approval; a pure approval step just completes.

## Execution examples

```python
from vds.v2 import ExecutionAgent, PlannerAgent, PlanStatus, default_tools, new_goal
from vds.gui.controller import BackendController

plan = PlannerAgent().create_plan(new_goal("detect cars from images", source="imgs/"))
plan.status = PlanStatus.APPROVED                      # approved via the plan session

agent = ExecutionAgent(default_tools(BackendController()))
ctx = agent.execute(plan, {"source": "imgs/", "name": "cars", "project_id": "cars"})
# ctx.status == AWAITING_APPROVAL  (parked at Manual Review)

agent.view(plan.id)          # live progress / steps / log for the GUI
agent.approve(plan.id)       # clear the gate; runs on to Export + Report
agent.summary(plan.id)       # ExecutionSummary(completed=..., failed=..., elapsed=...)
```

Via the master agent (the GUI's single entry): `DatasetEngineerAgent.generate_plan()`
→ approve → `execute_plan(plan, inputs)` → `execution_view(plan_id)`; drive controls
through `DatasetEngineerAgent.executor`.

## GUI integration (this phase)

Data surface only: `ExecutionAgent.view(plan_id)` returns everything the execution UI
binds to (status, progress, current step, upcoming steps, per-step statuses, live log,
warnings/errors). Events fire through the `on_event` callback for live updates. No Qt
widgets and no existing pages changed — matching the V2-20/V2-21 defer-Qt decision.

## Rules

The Execution Agent never plans, reasons, chooses models, generates recommendations,
or modifies Planner output. It only executes — by invoking existing BackendController
tools.
