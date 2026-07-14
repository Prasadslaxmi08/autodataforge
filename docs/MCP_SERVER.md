# MCP Server (V2-26)

Exposes the AutoDataForge as a set of [Model Context Protocol](https://modelcontextprotocol.io)
tools callable by any MCP client (Claude Desktop, Cursor, VS Code, …). Every task
flows through the existing `TaskOrchestrator` — the server **never** re-implements
planning, execution, or memory:

```
Client → MCP Server → TaskOrchestrator → Planner → Memory → Decision → Execution → BackendController
```

## Architecture

Two layers, so the SDK stays optional and the logic stays testable:

- **Core** — `vds/mcp/server.py` (`VdsMcpServer`). Protocol-agnostic, imports **no**
  `mcp` package. It is the tool registry, the pydantic-backed input/output JSON
  Schemas, request validation, structured errors, progress events, path security,
  and the routing into `TaskOrchestrator`. This is what the tests exercise.
- **Stdio binding** — `vds/mcp/stdio.py`. Thin glue that registers the core tools
  with the official `mcp` SDK `Server`, forwards `tools/call` to
  `VdsMcpServer.call_tool`, maps `McpToolError` onto MCP errors, and streams
  progress. The `mcp` SDK is an **optional** dependency.

```
             ┌─────────────────────────── vds/mcp/stdio.py (mcp SDK glue) ───────────┐
 MCP client ─┤  list_tools / call_tool / progress                                    │
             └───────────────▼───────────────────────────────────────────────────────┘
                   vds/mcp/server.py  VdsMcpServer   (validate · route · errors)
                                   │
                        TaskOrchestrator  (V2-25, unchanged)
                                   │  reads via the frozen BackendController tool bindings
```

## Setup

```bash
pip install "vds[mcp]"          # installs the optional MCP SDK
```

Claude Desktop / Cursor config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "autodataforge": {
      "command": "vds-mcp",
      "env": { "VDS_PROJECT_ROOT": "/path/to/your/projects" }
    }
  }
}
```

`VDS_PROJECT_ROOT` confines every filesystem `source` argument (see Security).

## Tools

| Tool | Input | Output | Routes to |
|---|---|---|---|
| `create_dataset` | `goal`, `source?`, `classes?`, `export_format?`, `name?`, `auto_approve?` | `task_id`, `status`, `estimated_time` | `orchestrator.execute` |
| `review_dataset` | `project_id`, `review_level?`, `auto_approve?` | `task_id`, `status` | `orchestrator.execute` |
| `export_dataset` | `project_id`, `export_format?`, `auto_approve?` | `task_id`, `status` | `orchestrator.execute` |
| `generate_report` | `task_id` | full report | `orchestrator.report` |
| `search_memory` | `query`, `top_k?`, `metadata?` | matches + recommendations | `MemoryAgent.recall` (via coordinator) |
| `list_projects` | — | `projects` | `BackendController.list_datasets` |
| `load_project` | `project_id` | `project` | `BackendController.dataset_detail` |
| `resume_task` | `task_id` | status | `orchestrator.approve` |
| `cancel_task` | `task_id` | status | `orchestrator.cancel` |
| `task_status` | `task_id` | state, stages, progress, errors | `orchestrator.status` |
| `health` | — | `status`, `name`, `version`, `tools` | — |

Every tool exposes a **description**, an **input schema** and (where structured) an
**output schema** — all generated from pydantic models (`inputSchema` is JSON
Schema with `additionalProperties: false`, so unknown fields are rejected).

## Approval flow

`create_dataset` (without `auto_approve`) plans → recalls memory → decides, then
**pauses** at the approval gate and returns `status: "awaiting_approval"`. Call
`resume_task` to approve and run execution + memory recording. `generate_report`
on a still-awaiting task returns an `approval_required` error.

## Streaming

During a task the server emits progress events to the client:

```
Planning → Memory Recall → Decision → (Approval) → Execution → Export → Completed
```

(mapped from the orchestrator's event stream).

## Errors

Structured `McpToolError` with a stable `code` and a JSON-RPC numeric code:

| code | when |
|---|---|
| `invalid_input` | bad/missing/unknown arguments, bad path, unknown tool |
| `task_not_found` | no task with that id |
| `execution_failed` | the run failed after retries |
| `export_failed` | export step failed |
| `approval_required` | action needs the task approved first |
| `internal_error` | unexpected fault (never leaks a stack across the boundary) |

## Security

- **No arbitrary filesystem access.** Every `source` path is resolved against
  `VDS_PROJECT_ROOT` and rejected if it escapes the root (absolute paths outside
  the root, `..` traversal).
- **Every request is validated** against its schema before anything runs (unknown
  fields rejected, types/ranges enforced).

## Example (JSON)

Request:

```json
{ "method": "tools/call",
  "params": { "name": "create_dataset",
    "arguments": { "goal": "create a thermal drone vehicle dataset",
                   "source": "clips/patrol.mp4", "classes": ["vehicle"],
                   "export_format": "coco", "auto_approve": true } } }
```

Result:

```json
{ "task_id": "a1b2c3…", "status": "completed", "estimated_time": 42.0 }
```

## Example client (Python, core directly)

```python
from vds.mcp import VdsMcpServer
from vds.gui.controller import BackendController

srv = VdsMcpServer(BackendController(), project_root="/data/projects")
out = srv.call_tool("create_dataset",
                    {"goal": "detect cars from images", "source": "cars/", "auto_approve": True},
                    on_progress=print)
print(srv.call_tool("generate_report", {"task_id": out["task_id"]}))
```
