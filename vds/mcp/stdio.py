"""MCP stdio binding (V2-26) — the only place the ``mcp`` SDK is imported.

Thin adapter: it turns :class:`vds.mcp.server.VdsMcpServer` into a real
Model-Context-Protocol server over stdio, callable by Claude Desktop, Cursor,
VS Code, etc. All logic lives in the core; this file is glue — it registers the
core tools with the SDK, forwards ``tools/call`` to ``VdsMcpServer.call_tool``,
maps :class:`McpToolError` onto MCP errors, and streams progress notifications.

The SDK is an **optional** dependency (``pip install "vds[mcp]"``). Importing this
module without it raises a clear message; the core (and its tests) never need it.
"""

from __future__ import annotations

import json
import os

from vds.mcp.server import McpToolError, VdsMcpServer


def _require_sdk():
    try:
        import mcp.server  # noqa: F401
        import mcp.types  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional dep
        raise SystemExit(
            "The MCP SDK is not installed. Install it with:  pip install \"vds[mcp]\""
        ) from exc


def build_sdk_server(core: VdsMcpServer):
    """Wrap a core VdsMcpServer in an SDK ``mcp.server.Server`` (low-level API)."""
    _require_sdk()
    import mcp.types as types
    from mcp.server import Server

    server = Server(core.NAME)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(name=s["name"], description=s["description"], inputSchema=s["inputSchema"])
            for s in core.list_tools()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        ctx = server.request_context

        def on_progress(ev: dict) -> None:
            # Best-effort structured progress log; clients receive it as a message.
            try:
                ctx.session.send_log_message(level="info", data=ev)  # type: ignore[attr-defined]
            except Exception:
                pass

        try:
            result = core.call_tool(name, arguments, on_progress=on_progress)
        except McpToolError as exc:
            # Structured tool error surfaced to the client (isError content).
            return [types.TextContent(type="text", text=json.dumps({"error": exc.to_dict()}))]
        return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    return server


def main() -> None:  # pragma: no cover - process entry point
    """Console entry: serve over stdio. Project root from ``VDS_PROJECT_ROOT``."""
    _require_sdk()
    import anyio
    from mcp.server.stdio import stdio_server

    from vds.gui.controller import BackendController  # the frozen backend seam

    core = VdsMcpServer(BackendController(), project_root=os.environ.get("VDS_PROJECT_ROOT", "."))
    sdk = build_sdk_server(core)

    async def _run() -> None:
        async with stdio_server() as (read, write):
            await sdk.run(read, write, sdk.create_initialization_options())

    anyio.run(_run)


if __name__ == "__main__":  # pragma: no cover
    main()
