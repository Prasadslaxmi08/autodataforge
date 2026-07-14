"""V2-26 MCP server (core) — tool registration, schemas, dispatch, validation,
structured errors, streaming progress, security, and TaskOrchestrator/memory
integration. The core is protocol-agnostic; the mcp SDK is not needed here.
"""

from __future__ import annotations

import pytest

from vds.mcp import McpErrorCode, McpToolError, VdsMcpServer
from vds.memory.store import MemoryStore
from vds.v2 import TaskState


class _FakeController:
    def __getattr__(self, name):
        return lambda *a, **k: None

    def list_datasets(self):
        return ["proj_a", "proj_b"]

    def dataset_detail(self, project_id=None):
        return {"id": project_id, "images": 10}


class _FailingDetection(_FakeController):
    def ai_annotate(self, *a, **k):
        raise ValueError("boom")


def _server(tmp_path, controller=None):
    ctrl = controller or _FakeController()
    srv = VdsMcpServer(ctrl, project_root=tmp_path)
    srv._orch.coordinator.memory._store = MemoryStore(tmp_path / "mem.json")  # isolate memory
    return srv


_EXPECTED_TOOLS = {
    "create_dataset", "review_dataset", "export_dataset", "generate_report", "search_memory",
    "list_projects", "load_project", "resume_task", "cancel_task", "task_status", "health",
}


# --- registration + schemas -------------------------------------------
def test_all_tools_registered(tmp_path):
    assert set(_server(tmp_path).tool_names()) == _EXPECTED_TOOLS


def test_every_tool_exposes_a_schema(tmp_path):
    for spec in _server(tmp_path).list_tools():
        assert spec["name"] and spec["description"]
        assert spec["inputSchema"]["type"] == "object"  # JSON Schema
        assert "properties" in spec["inputSchema"]


def test_health(tmp_path):
    out = _server(tmp_path).call_tool("health", {})
    assert out["status"] == "ok" and out["tools"] == len(_EXPECTED_TOOLS)
    assert out["name"] == "autodataforge"


# --- validation + errors ----------------------------------------------
def test_unknown_tool_is_invalid_input(tmp_path):
    with pytest.raises(McpToolError) as e:
        _server(tmp_path).call_tool("nope", {})
    assert e.value.code == McpErrorCode.INVALID_INPUT


def test_missing_required_field_rejected(tmp_path):
    with pytest.raises(McpToolError) as e:
        _server(tmp_path).call_tool("create_dataset", {})  # goal is required
    assert e.value.code == McpErrorCode.INVALID_INPUT
    assert e.value.rpc_code == -32602 and e.value.data  # carries pydantic error details


def test_unknown_field_rejected(tmp_path):
    with pytest.raises(McpToolError) as e:
        _server(tmp_path).call_tool("create_dataset", {"goal": "x", "bogus": 1})
    assert e.value.code == McpErrorCode.INVALID_INPUT


def test_task_not_found(tmp_path):
    with pytest.raises(McpToolError) as e:
        _server(tmp_path).call_tool("task_status", {"task_id": "missing"})
    assert e.value.code == McpErrorCode.TASK_NOT_FOUND


# --- security ---------------------------------------------------------
def test_path_outside_root_rejected(tmp_path):
    srv = _server(tmp_path)
    with pytest.raises(McpToolError) as e:
        srv.call_tool("create_dataset", {"goal": "detect cars", "source": "../../etc/passwd"})
    assert e.value.code == McpErrorCode.INVALID_INPUT
    assert "outside project root" in e.value.message


def test_source_under_root_allowed(tmp_path):
    (tmp_path / "imgs").mkdir()
    srv = _server(tmp_path)
    out = srv.call_tool("create_dataset", {"goal": "detect cars from images",
                                           "source": "imgs", "auto_approve": True})
    assert out["task_id"] and out["status"] == TaskState.COMPLETED.value


# --- full orchestration through the server ----------------------------
def test_create_dataset_runs_through_orchestrator(tmp_path):
    srv = _server(tmp_path)
    out = srv.call_tool("create_dataset", {
        "goal": "create thermal drone vehicle dataset", "classes": ["vehicle"],
        "export_format": "coco", "auto_approve": True})
    assert out["status"] == TaskState.COMPLETED.value
    assert out["estimated_time"] >= 0
    # a report is now available and memory was recorded
    rep = srv.call_tool("generate_report", {"task_id": out["task_id"]})
    assert rep["statistics"]["memory_stored"] is True


def test_approval_pause_then_resume(tmp_path):
    srv = _server(tmp_path)
    out = srv.call_tool("create_dataset", {"goal": "detect cars from images"})  # no auto_approve
    assert out["status"] == TaskState.AWAITING_APPROVAL.value
    # report before approval is refused
    with pytest.raises(McpToolError) as e:
        srv.call_tool("generate_report", {"task_id": out["task_id"]})
    assert e.value.code == McpErrorCode.APPROVAL_REQUIRED
    # resume drives it to completion
    status = srv.call_tool("resume_task", {"task_id": out["task_id"]})
    assert status["status"] == TaskState.COMPLETED.value and status["progress"] > 0


def test_task_status_and_cancel(tmp_path):
    srv = _server(tmp_path)
    out = srv.call_tool("create_dataset", {"goal": "detect cars from images"})
    st = srv.call_tool("task_status", {"task_id": out["task_id"]})
    assert st["status"] == TaskState.AWAITING_APPROVAL.value
    assert [s["name"] for s in st["stages"]][0] == "Planning"
    cancelled = srv.call_tool("cancel_task", {"task_id": out["task_id"]})
    assert cancelled["status"] == TaskState.CANCELLED.value


def test_execution_failure_maps_to_execution_error(tmp_path):
    srv = _server(tmp_path, controller=_FailingDetection())
    with pytest.raises(McpToolError) as e:
        srv.call_tool("create_dataset", {"goal": "detect cars from images", "auto_approve": True})
    assert e.value.code == McpErrorCode.EXECUTION_FAILED


# --- streaming progress ----------------------------------------------
def test_progress_events_streamed(tmp_path):
    srv = _server(tmp_path)
    events: list[dict] = []
    srv.call_tool("create_dataset", {"goal": "detect cars from images", "auto_approve": True},
                  on_progress=events.append)
    stages = [e["stage"] for e in events]
    for expected in ("Planning", "Memory Recall", "Decision", "Execution", "Completed"):
        assert expected in stages


# --- memory + read tools ---------------------------------------------
def test_search_memory_reuses_memory_agent(tmp_path):
    srv = _server(tmp_path)
    srv.call_tool("create_dataset", {"goal": "thermal drone vehicle dataset", "auto_approve": True})
    res = srv.call_tool("search_memory", {"query": "thermal drone dataset"})
    assert res["similar_projects"] and res["confidence"] >= 0


def test_list_and_load_project(tmp_path):
    srv = _server(tmp_path)
    assert srv.call_tool("list_projects", {})["projects"] == ["proj_a", "proj_b"]
    assert srv.call_tool("load_project", {"project_id": "proj_a"})["project"] == {"id": "proj_a", "images": 10}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
