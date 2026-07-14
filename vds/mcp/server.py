"""AutoDataForge MCP server — core (V2-26).

Framework-agnostic. This module speaks **no** MCP wire protocol and imports **no**
`mcp` package: it is the tool registry, the JSON-Schema-backed input/output
contracts (via pydantic), request validation, structured errors, progress events,
and the routing that funnels every task through the existing ``TaskOrchestrator``.
The thin stdio/SDK binding that turns this into a real MCP server lives in
``vds.mcp.stdio`` and just forwards to :class:`VdsMcpServer`.

Rules honoured (brief §RULES): no duplicated execution, planning, or memory logic.
Task-generating tools (create/review/export) go through ``TaskOrchestrator.execute``;
control tools (resume/cancel/status/report) call the orchestrator's own methods;
memory search reuses the orchestrator's MemoryAgent; read tools (list/load) are thin
bindings to the frozen ``BackendController`` (the same tools ExecutionAgent uses).
Nothing here re-implements the pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from vds.v2.goal import new_goal
from vds.v2.memory_agent import memory_view
from vds.v2.task_orchestrator import TaskContext, TaskState

Progress = Callable[[dict], None]


# --- errors -----------------------------------------------------------
class McpErrorCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    TASK_NOT_FOUND = "task_not_found"
    EXECUTION_FAILED = "execution_failed"
    EXPORT_FAILED = "export_failed"
    APPROVAL_REQUIRED = "approval_required"
    INTERNAL_ERROR = "internal_error"


# JSON-RPC numeric codes for the SDK layer (MCP is JSON-RPC 2.0).
_RPC_CODE = {
    McpErrorCode.INVALID_INPUT: -32602,   # Invalid params
    McpErrorCode.TASK_NOT_FOUND: -32001,
    McpErrorCode.EXECUTION_FAILED: -32002,
    McpErrorCode.EXPORT_FAILED: -32003,
    McpErrorCode.APPROVAL_REQUIRED: -32004,
    McpErrorCode.INTERNAL_ERROR: -32603,  # Internal error
}


class McpToolError(Exception):
    """A structured MCP tool error. Carries a stable string ``code`` (brief §ERRORS)
    plus the JSON-RPC numeric code the transport layer needs."""

    def __init__(self, code: McpErrorCode, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    @property
    def rpc_code(self) -> int:
        return _RPC_CODE[self.code]

    def to_dict(self) -> dict:
        return {"code": self.code.value, "rpc_code": self.rpc_code,
                "message": self.message, "data": self.data}


# --- tool input/output contracts (pydantic -> JSON Schema) ------------
class _In(BaseModel):
    model_config = {"extra": "forbid"}  # reject unknown fields (validation)


class CreateDatasetInput(_In):
    goal: str = Field(..., min_length=1, description="Natural-language dataset goal.")
    source: str | None = Field(None, description="Image folder or video path (under the project root).")
    classes: list[str] = Field(default_factory=list, description="Target object classes.")
    export_format: str = Field("coco", description="coco | yolo | voc.")
    name: str | None = Field(None, description="Project name / id.")
    auto_approve: bool = Field(False, description="Skip the human approval gate.")
    frame_strategy: str = Field(
        "every_n",
        description="Video only: every_frame | every_n | every_seconds | fixed_count | scene_change.")
    frame_param: float | None = Field(
        None, description="Video only: strategy parameter (n / seconds / count). Ignored for every_frame.")
    dedup: bool = Field(
        True, description="Video only: drop near-duplicate frames. Turn off for static-camera footage.")


class ReviewDatasetInput(_In):
    project_id: str = Field(..., min_length=1)
    review_level: str | None = Field(None, description="low | medium | high.")
    auto_approve: bool = False


class ExportDatasetInput(_In):
    project_id: str = Field(..., min_length=1)
    export_format: str = Field("coco", description="coco | yolo | voc.")
    auto_approve: bool = True


class TaskIdInput(_In):
    task_id: str = Field(..., min_length=1)


class SearchMemoryInput(_In):
    query: str = Field(..., min_length=1, description="Goal text to find similar past projects.")
    top_k: int = Field(3, ge=1, le=20)
    metadata: dict = Field(default_factory=dict)


class LoadProjectInput(_In):
    project_id: str = Field(..., min_length=1)


class EmptyInput(_In):
    pass


class TaskRef(BaseModel):
    task_id: str
    status: str


class CreateDatasetOutput(TaskRef):
    estimated_time: float = Field(0.0, description="Estimated runtime in seconds.")


class TaskStatusOutput(TaskRef):
    stages: list[dict]
    progress: float
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class HealthOutput(BaseModel):
    status: str
    name: str
    version: str
    tools: int


# --- tool definition --------------------------------------------------
class Tool:
    def __init__(self, name: str, description: str, input_model: type[BaseModel],
                 output_model: type[BaseModel] | None, handler: Callable) -> None:
        self.name = name
        self.description = description
        self.input_model = input_model
        self.output_model = output_model
        self.handler = handler

    def spec(self) -> dict:
        """MCP tool descriptor: name, description, input + output JSON Schema."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_model.model_json_schema(),
            "outputSchema": self.output_model.model_json_schema() if self.output_model else None,
        }


_TERMINAL = {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}


class VdsMcpServer:
    """The AutoDataForge as MCP tools, over one shared TaskOrchestrator."""

    NAME = "autodataforge"
    VERSION = "2.0.0"

    def __init__(self, controller=None, *, orchestrator=None, project_root: str | Path = ".",
                 tools=None) -> None:
        from vds.v2.task_orchestrator import TaskOrchestrator

        self._root = Path(project_root).resolve()
        if orchestrator is None:
            orchestrator = TaskOrchestrator(controller, on_event=self._on_orch_event)
        else:
            orchestrator._on_event = self._on_orch_event  # subscribe to the event stream
        self._orch = orchestrator
        if tools is None and controller is not None:
            from vds.v2.tool_registry import default_tools
            tools = default_tools(controller)
        self._backend = tools  # frozen BackendController, wrapped as read tools
        self._progress: Progress | None = None
        self._registry: dict[str, Tool] = {}
        self._register_all()

    # --- registry ------------------------------------------------------
    def _register_all(self) -> None:
        t = self._registry
        reg = lambda name, desc, im, om, h: t.__setitem__(name, Tool(name, desc, im, om, h))  # noqa: E731
        reg("create_dataset", "Create an annotated dataset from a goal (plans, decides, "
            "pauses for approval, then executes via the orchestrator).",
            CreateDatasetInput, CreateDatasetOutput, self._create_dataset)
        reg("review_dataset", "Run quality review / re-annotation on an existing dataset.",
            ReviewDatasetInput, TaskRef, self._review_dataset)
        reg("export_dataset", "Export an existing dataset to COCO/YOLO/VOC.",
            ExportDatasetInput, TaskRef, self._export_dataset)
        reg("generate_report", "Return the final report for a completed task.",
            TaskIdInput, None, self._generate_report)
        reg("search_memory", "Find similar past projects and the settings that worked.",
            SearchMemoryInput, None, self._search_memory)
        reg("list_projects", "List existing datasets/projects.",
            EmptyInput, None, self._list_projects)
        reg("load_project", "Load a project's summary/detail.",
            LoadProjectInput, None, self._load_project)
        reg("resume_task", "Approve and resume a task waiting at the approval gate.",
            TaskIdInput, TaskStatusOutput, self._resume_task)
        reg("cancel_task", "Cancel a running or pending task.",
            TaskIdInput, TaskStatusOutput, self._cancel_task)
        reg("task_status", "Get a task's state, stage timeline, progress and errors.",
            TaskIdInput, TaskStatusOutput, self._task_status)
        reg("health", "Server + orchestrator health.",
            EmptyInput, HealthOutput, self._health)

    def list_tools(self) -> list[dict]:
        return [tool.spec() for tool in self._registry.values()]

    def tool_names(self) -> list[str]:
        return list(self._registry)

    # --- dispatch ------------------------------------------------------
    def call_tool(self, name: str, arguments: dict | None = None, *,
                  on_progress: Progress | None = None) -> dict:
        """Validate arguments, run the handler, return a JSON-able result dict.
        Raises :class:`McpToolError` on any structured failure."""
        tool = self._registry.get(name)
        if tool is None:
            raise McpToolError(McpErrorCode.INVALID_INPUT, f"unknown tool: {name}")
        try:
            args = tool.input_model.model_validate(arguments or {})
        except ValidationError as exc:
            raise McpToolError(McpErrorCode.INVALID_INPUT, f"invalid input for {name}",
                               data=exc.errors(include_url=False)) from exc
        self._progress = on_progress
        try:
            return tool.handler(args)
        except McpToolError:
            raise
        except Exception as exc:  # never leak a raw stack across the protocol boundary
            raise McpToolError(McpErrorCode.INTERNAL_ERROR, str(exc)) from exc
        finally:
            self._progress = None

    # --- task-generating tools (through TaskOrchestrator.execute) ------
    def _create_dataset(self, a: CreateDatasetInput) -> dict:
        self._check_source(a.source)
        project = {"project_id": a.name or _slug(a.goal), "export_summary": {"format": a.export_format}}
        metadata = {"project_id": project["project_id"], "existing_classes": a.classes}
        goal = new_goal(a.goal, source=a.source, name=project["project_id"])
        inputs = {"source": a.source, "name": project["project_id"], "project_id": project["project_id"],
                  "config": _extraction_config(a.frame_strategy, a.frame_param),
                  "dedup": a.dedup, "export_format": a.export_format}
        ctx = self._orch.execute(goal, project=project, metadata=metadata, inputs=inputs,
                                 auto_approve=a.auto_approve)
        est = ctx.decision_report.estimated_runtime_seconds if ctx.decision_report else (
            ctx.plan.estimated_runtime_seconds if ctx.plan else 0.0)
        self._raise_if_failed(ctx)
        return CreateDatasetOutput(task_id=ctx.id, status=ctx.state.value, estimated_time=est).model_dump()

    def _review_dataset(self, a: ReviewDatasetInput) -> dict:
        goal = new_goal(f"review and improve dataset {a.project_id}",
                        project=a.project_id, dataset=a.project_id)
        ctx = self._orch.execute(goal, project={"project_id": a.project_id},
                                 inputs={"project_id": a.project_id}, auto_approve=a.auto_approve)
        self._raise_if_failed(ctx)
        return TaskRef(task_id=ctx.id, status=ctx.state.value).model_dump()

    def _export_dataset(self, a: ExportDatasetInput) -> dict:
        goal = new_goal(f"export dataset {a.project_id} to {a.export_format}", project=a.project_id)
        ctx = self._orch.execute(
            goal, project={"project_id": a.project_id, "export_summary": {"format": a.export_format}},
            inputs={"project_id": a.project_id}, auto_approve=a.auto_approve)
        if ctx.state == TaskState.FAILED and any("export" in e.lower() for e in ctx.errors):
            raise McpToolError(McpErrorCode.EXPORT_FAILED, "; ".join(ctx.errors), data={"task_id": ctx.id})
        self._raise_if_failed(ctx)
        return TaskRef(task_id=ctx.id, status=ctx.state.value).model_dump()

    # --- control tools (orchestrator's own methods) --------------------
    def _resume_task(self, a: TaskIdInput) -> dict:
        ctx = self._task(a.task_id)
        if ctx.state in _TERMINAL:
            return self._status_dict(ctx)
        ctx = self._orch.approve(a.task_id)
        self._raise_if_failed(ctx)
        return self._status_dict(ctx)

    def _cancel_task(self, a: TaskIdInput) -> dict:
        self._task(a.task_id)  # existence check -> TASK_NOT_FOUND
        return self._status_dict(self._orch.cancel(a.task_id))

    def _task_status(self, a: TaskIdInput) -> dict:
        return self._status_dict(self._task(a.task_id))

    def _generate_report(self, a: TaskIdInput) -> dict:
        ctx = self._task(a.task_id)
        if ctx.state == TaskState.AWAITING_APPROVAL:
            raise McpToolError(McpErrorCode.APPROVAL_REQUIRED,
                               "task is awaiting approval; call resume_task first",
                               data={"task_id": ctx.id})
        return self._orch.report(a.task_id)

    # --- read / utility tools -----------------------------------------
    def _search_memory(self, a: SearchMemoryInput) -> dict:
        exp = self._orch.coordinator.recall_experience(
            new_goal(a.query), metadata=a.metadata or None, top_k=a.top_k)
        return memory_view(exp)

    def _list_projects(self, a: EmptyInput) -> dict:
        return {"projects": _jsonable(self._backend_run("list_projects"))}

    def _load_project(self, a: LoadProjectInput) -> dict:
        return {"project": _jsonable(self._backend_run("open_project", project_id=a.project_id))}

    def _health(self, a: EmptyInput) -> dict:
        return HealthOutput(status="ok", name=self.NAME, version=self.VERSION,
                            tools=len(self._registry)).model_dump()

    # --- helpers -------------------------------------------------------
    def _task(self, task_id: str) -> TaskContext:
        try:
            return self._orch.status(task_id)
        except KeyError as exc:
            raise McpToolError(McpErrorCode.TASK_NOT_FOUND, f"no such task: {task_id}") from exc

    def _raise_if_failed(self, ctx: TaskContext) -> None:
        if ctx.state == TaskState.FAILED:
            raise McpToolError(McpErrorCode.EXECUTION_FAILED,
                               "; ".join(ctx.errors) or "execution failed", data={"task_id": ctx.id})

    def _status_dict(self, ctx: TaskContext) -> dict:
        v = self._orch.view(ctx.id)
        done = sum(1 for s in v["stages"] if s["status"] == "done")
        progress = round(done / len(v["stages"]), 4) if v["stages"] else 0.0
        return TaskStatusOutput(task_id=ctx.id, status=ctx.state.value, stages=v["stages"],
                                progress=progress, warnings=v["warnings"], errors=v["errors"]).model_dump()

    def _backend_run(self, tool: str, **kwargs) -> Any:
        if self._backend is None or tool not in self._backend:
            raise McpToolError(McpErrorCode.INTERNAL_ERROR, f"backend tool unavailable: {tool}")
        return self._backend.get(tool).run(**kwargs)

    def _check_source(self, source: str | None) -> None:
        """Security (brief §SECURITY): confine any filesystem source to the project
        root. No absolute escapes, no ``..`` traversal outside the root."""
        if not source:
            return
        p = Path(source)
        try:
            full = p.resolve() if p.is_absolute() else (self._root / p).resolve()
        except (OSError, ValueError) as exc:
            raise McpToolError(McpErrorCode.INVALID_INPUT, f"invalid path: {source}") from exc
        if full != self._root and self._root not in full.parents:
            raise McpToolError(McpErrorCode.INVALID_INPUT, f"path outside project root: {source}")

    # --- progress event stream (brief §STREAMING) ----------------------
    _STAGE = {
        "PlanningStarted": "Planning", "PlanningCompleted": "Planning",
        "MemoryLoaded": "Memory Recall", "DecisionCompleted": "Decision",
        "ApprovalRequested": "Approval", "ExecutionStarted": "Execution",
        "ExecutionCompleted": "Execution", "MemoryStored": "Memory Save",
        "TaskCompleted": "Completed", "TaskFailed": "Failed", "TaskCancelled": "Cancelled",
    }

    def _on_orch_event(self, event: str, payload: dict) -> None:
        if self._progress is None:
            return
        self._progress({"stage": self._STAGE.get(event, event), "event": event,
                        "task_id": payload.get("task"), "message": payload.get("message", "")})


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower())[:40].strip("_") or "dataset"


def _extraction_config(strategy: str, param: float | None):
    """Build a video ExtractionConfig from the tool's frame_strategy/frame_param. The
    param maps onto whichever field the strategy reads (every_frame ignores it)."""
    from vds.video import ExtractionConfig

    cfg = ExtractionConfig(strategy=strategy)
    if param is not None:
        if strategy == "every_n":
            cfg.every_n = int(param)
        elif strategy == "every_seconds":
            cfg.seconds = float(param)
        elif strategy == "fixed_count":
            cfg.count = int(param)
        elif strategy == "scene_change":
            cfg.scene_threshold = int(param)
    return cfg


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
