"""AutoDataForge MCP server (V2-26).

Exposes the platform as MCP tools for any MCP client. Every task flows through the
existing ``TaskOrchestrator`` — no duplicated planning/execution/memory logic. The
core (:mod:`vds.mcp.server`) is protocol-agnostic and dependency-free; the stdio
SDK binding (:mod:`vds.mcp.stdio`, optional ``mcp`` dependency) is thin glue.
"""

from vds.mcp.server import (
    McpErrorCode,
    McpToolError,
    Tool,
    VdsMcpServer,
)

__all__ = ["McpErrorCode", "McpToolError", "Tool", "VdsMcpServer"]
