# TaskOrchestrator (V2-25)

The **single entry point** for a dataset-generation task. The GUI used to call
`PlannerAgent → MemoryAgent → DecisionAgent → ExecutionAgent → MemoryAgent` by
hand; now it makes one call:

```python
orch = TaskOrchestrator(controller)
ctx  = orch.execute(goal, project)   # plans, recalls, decides -> pauses at approval
ctx  = orch.approve(ctx.id)          # executes -> records memory -> completes
rep  = orch.report(ctx.id)           # final report
```

The orchestrator **only coordinates**. Every piece of real work stays in the four
existing agents, reached through the `DatasetEngineerAgent` facade (the
agent-access layer / "agent coordinator") which already wires them and exposes the
verbs `generate_plan`, `recall_experience`, `optimize_plan`, `execute_plan`, and
`record_experience`. No agent internals are touched and **no agent logic is
duplicated** — this component is a state machine, an event stream, a timeline, and
the failure policy, nothing more. `BackendController` is untouched.

## Architecture

```
GUI ──▶ TaskOrchestrator ──▶ DatasetEngineerAgent ──┬─▶ PlannerAgent   (generate_plan)
        (state machine,      (agent coordinator)    ├─▶ MemoryAgent    (recall / record)
         events, timeline,                          ├─▶ DecisionAgent  (optimize_plan)
         failure policy)                            └─▶ ExecutionAgent (execute_plan)
                                                              │
                                                     ToolRegistry ─▶ BackendController ─▶ V1 services
```

The GUI talks **only** to the TaskOrchestrator — never to the four agents directly.

## Sequence

```
User ─▶ execute(goal, project)
          │  PLANNING          PlannerAgent.generate_plan        ▶ PlanningStarted / PlanningCompleted
          │  MEMORY_RETRIEVAL  MemoryAgent.recall_experience     ▶ MemoryLoaded
          │  DECISION_MAKING   DecisionAgent.optimize_plan       ▶ DecisionCompleted
          │  AWAITING_APPROVAL (pause; return context)           ▶ ApprovalRequested
User ─▶ approve(task_id)
          │  EXECUTING         ExecutionAgent.execute_plan       ▶ ExecutionStarted / ExecutionCompleted
          │  RECORDING_MEMORY  MemoryAgent.record_experience     ▶ MemoryStored
          │  COMPLETED                                           ▶ TaskCompleted
        report(task_id) ─▶ Execution Summary · Decision Report · Memory Matches · Warnings · Timing · Statistics · Artifacts
```

`execute(..., auto_approve=True)` runs straight through the approval gate (for
headless/automated use).

## State machine

```
Idle ─▶ Planning ─▶ Memory Retrieval ─▶ Decision Making ─▶ Awaiting Approval ─▶ Executing ─▶ Recording Memory ─▶ Completed
                                                                 │                 │
                                                             Cancelled          Failed
```

`TaskState`: `idle, planning, memory_retrieval, decision_making, awaiting_approval,
executing, recording_memory, completed, failed, cancelled`. `cancel(task_id)` moves
any non-terminal task to `cancelled` (and cancels the executor if a run is live).

## Agent interaction / context

`TaskContext` (serializable) carries the whole run: current goal, plan, decision
report, memory experience, execution summary, stored memory id, per-stage status,
warnings, errors, structured logs, timeline, and timing. Retrieve it any time with
`orch.status(task_id)`.

## Failure recovery

| Failure | Policy | Result |
|---|---|---|
| Planner failed | **Abort** | state → `failed`, `TaskFailed` |
| Memory unavailable | **Continue** | warning; Memory stage `skipped`; planning proceeds |
| Decision failed | **Planner defaults** | warning; keep the planner plan; Decision stage `skipped` |
| Execution failed | **Retry, then abort** | retry failed steps up to `max_execution_retries`, then state → `failed` |
| Memory save failed | **Warn only** | warning; task still `completed` |

The ExecutionAgent's own internal approval gates are cleared automatically after the
user approves at the orchestrator gate (approval is asked **once**).

## Events

`PlanningStarted, PlanningCompleted, MemoryLoaded, DecisionCompleted,
ApprovalRequested, ExecutionStarted, ExecutionCompleted, MemoryStored,
TaskCompleted, TaskFailed` (+ `TaskCancelled`). Subscribe via the `on_event`
callback: `TaskOrchestrator(controller, on_event=lambda event, payload: ...)`.

## Logging

Every stage appends a structured log row: `ts, agent, action, duration_ms, status,
message` (and mirrors to the app logger as `orchestrator.stage`).

## GUI

`orch.view(task_id)` (or `task_view(ctx)`) returns the live **orchestration
timeline** — one row per stage with status (`pending / active / done / skipped /
failed`):

```
Planning · Memory · Decision · Approval · Execution · Memory Save · Complete
```

plus the emitted events, logs, warnings, errors, and elapsed time. Data only (no Qt
this phase — same pattern as `decision_view` / `memory_view`).

## Report

`orch.report(task_id)` returns: `execution_summary`, `decision_report`,
`memory_matches`, `warnings`, `errors`, `timing` (elapsed + per-stage ms),
`statistics` (steps total/completed/failed/retried, memory_stored), and `artifacts`
(dataset, export).

## Example

```python
orch = TaskOrchestrator(controller)
ctx  = orch.execute("create thermal drone vehicle dataset",
                    project={"project_id": "af_thermal", "image_count": 800, "resolution": "high"})
# ctx.state == awaiting_approval; ctx.memory shows similar past projects
orch.approve(ctx.id)
print(orch.report(ctx.id)["statistics"])
```

## Note on naming

A lower-level `TaskOrchestrator` from V2-20 (a plan-step walker over the nine
registry stub agents) still lives at `vds.v2.orchestrator` and is now exported as
`PlanStepOrchestrator`. The package-level `vds.v2.TaskOrchestrator` is this V2-25
agent coordinator.
