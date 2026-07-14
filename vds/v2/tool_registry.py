"""Tool registry — the existing backend services, as tools (V2-20 §TOOL REGISTRY).

No duplicated implementations: every tool is a thin binding to a real
``BackendController`` method (the frozen GUI↔backend seam). Agents will invoke
tools by name in a future phase; today the registry just declares the surface and
proves the bindings resolve.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vds.gui.controller import BackendController


@dataclass
class Tool:
    name: str
    description: str
    run: Callable[..., Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def invoke(self, name: str, **kwargs: Any) -> Any:
        return self.get(name).run(**kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def default_tools(controller: BackendController) -> ToolRegistry:
    """Wrap the frozen BackendController methods as named tools. Each entry is
    (name, description, bound method) — no logic, just a reference."""
    reg = ToolRegistry()
    bindings: list[tuple[str, str, Callable[..., Any]]] = [
        ("import_images", "Import an image folder through the full pipeline.", controller.import_dataset),
        ("import_video", "Import a video: extract frames, then run the pipeline.", controller.import_video_dataset),
        ("extract_frames", "Probe video metadata / frame plan.", controller.probe_video),
        ("run_detection", "Run the configured detector on one image.", controller.ai_annotate),
        ("run_segmentation", "Regenerate a box's mask with the configured segmenter.", controller.resegment),
        ("review_dataset", "Reproduce per-annotation verdicts for a dataset.", controller.object_verdicts),
        ("export_dataset", "Re-export an existing dataset (COCO/YOLO).", controller.export_project),
        ("generate_report", "Render a run's report as Markdown.", controller.report_markdown),
        ("open_project", "Load a project's summary/detail.", controller.dataset_detail),
        ("list_projects", "List all datasets.", controller.list_datasets),
    ]
    for name, desc, fn in bindings:
        reg.register(Tool(name=name, description=desc, run=fn))
    return reg
