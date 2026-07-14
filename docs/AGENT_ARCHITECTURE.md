# Version 2 — Agent Architecture

Version 1 is a **workflow** app: the user drives Import → Annotate → Review →
Export by hand. Version 2 makes it **goal-driven**: the user states *what* they
want, and a team of agents decides *how*, driving the existing V1 services as
tools. This phase (V2-20) builds the **architecture only** — the seams, contracts,
registries, and the deterministic state machine. No planning intelligence, no
autonomous decisions, no model selection, no memory reasoning, no MCP, no LLM
calls yet. Every agent handler is a declared-metadata no-op.

Everything below `BackendController` is **frozen**. The V2 layer never
reimplements a backend service; it wraps them.

## Architecture

```
GUI
 └─ DatasetEngineerAgent        (the only V2 component the GUI touches)
      ├─ Planner                (Goal -> serializable ExecutionPlan)
      ├─ TaskOrchestrator       (deterministic state machine over the plan)
      │    ├─ AgentRegistry     (name -> agent)
      │    ├─ ToolRegistry      (tool name -> BackendController method)
      │    └─ MessageBus        (append-only AgentMessage log)
      └─ SessionState           (fully serializable run state)

TaskOrchestrator dispatch ─▶ BackendController ─▶ V1 services ─▶ YOLO / Segmentation / Export
```

The package lives at `vds/v2/` (the brief called it `vde/agents/`; renamed to stay
inside the one installed `vds` package and to avoid colliding with the V1
`vds/agents` LLM layer).

## Agents (`registry.py`)

Nine agents register metadata (capabilities, supported tasks, dependencies,
status, version, description). Their `handle()` is a no-op this phase; task names
already line up with the tools so wiring handler → tool is a later, mechanical step.

| Agent | Supported tasks | Purpose |
|-------|-----------------|---------|
| PlannerAgent | analyze_input, select_models | Understand goal/input; pick models (future) |
| DatasetAnalysisAgent | inspect_dataset | Inspect the incoming/existing dataset |
| ImportAgent | import_images, import_video, extract_frames | Ingest and extract frames |
| DetectionAgent | run_detection | Object detection |
| SegmentationAgent | run_segmentation | Masks for detected boxes |
| QualityAgent | review_dataset | Annotation quality / verdicts |
| ReviewAgent | await_approval | Holds the human-approval gate |
| MemoryAgent | record_memory | Record the run into engineering memory (future) |
| ExportAgent | export_dataset | Export approved dataset (COCO/YOLO) |

## Tools (`tool_registry.py`)

Each tool is a thin binding to a real, frozen `BackendController` method — no
duplicated implementations:

| Tool | BackendController method |
|------|--------------------------|
| import_images | `import_dataset` |
| import_video | `import_video_dataset` |
| extract_frames | `probe_video` |
| run_detection | `ai_annotate` |
| run_segmentation | `resegment` |
| review_dataset | `object_verdicts` |
| export_dataset | `export_project` |
| generate_report | `report_markdown` |
| open_project | `dataset_detail` |
| list_projects | `list_datasets` |

## Message flow

Agents never call each other. The orchestrator posts an `AgentMessage`
(id, sender, receiver, task, arguments, priority, timestamp, reasoning, status,
result, execution_time_ms, errors) to the `MessageBus`; the assigned agent handles
it; the result and timing are written back on the same message. The bus is an
append-only log, so a whole run is auditable from the messages alone.

## Execution lifecycle

1. GUI submits a `Goal` (natural-language text + known params) to
   `DatasetEngineerAgent.submit_goal()`.
2. `Planner` produces a serializable `ExecutionPlan` — the fixed V1 pipeline as a
   linear chain of agent steps:
   Input Analysis → Dataset Inspection → Import → Model Selection → Frame
   Extraction → Detection → Segmentation → Quality Review → **Human Approval** →
   Export → Record Memory.
3. A `SessionState` opens and the `TaskOrchestrator` walks ready steps in
   dependency order, dispatching each to its agent via the bus.
4. At the **Human Approval** gate the run parks (`AWAITING_APPROVAL`) — no major
   action executes silently. `approve()` clears the gate and the run continues.
5. On a handler error a step is marked `FAILED` and the run halts; `retry(step_id)`
   resets it and resumes. `pause`/`resume`/`cancel` control the run.
6. `report()` renders a Markdown summary from the session's recorded facts;
   `view()` returns the data a future Qt page binds to (current agent/task,
   reasoning, timeline, agent status).

State is fully serializable (pydantic), so a run can be persisted and reloaded
from `SessionState` alone.

## GUI integration (this phase)

Only the **data surface** exists: `DatasetEngineerAgent.view()`. No new Qt page and
no edits to existing pages — the closing scope of V2-20 is architecture only. A
future phase renders `view()` into panels (Goal Input, Current Agent, Current Task,
Reasoning, Execution Timeline, Agent Status).

## Future MCP integration points

- **ToolRegistry** is the natural MCP boundary: each `Tool` already has a name,
  description, and callable — expose the registry as an MCP server and remote
  agents drive the same frozen backend.
- **AgentMessage** carries `tool` semantics (task + arguments + result) that map
  onto MCP tool calls without a schema change.
- **AgentRegistry** entries (capabilities, supported tasks) are the raw material
  for advertising agents/tools to an MCP client.

## Not in this phase

No planning intelligence, no autonomous decisions, no model selection, no memory
reasoning, no MCP, no LLM calls. The goal is a clean, extensible foundation that
preserves 100% Version 1 compatibility.
