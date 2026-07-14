"""Agent registry + the nine V2 agents (V2-20 §AGENT REGISTRY).

The registry is the orchestrator's directory: given a step's ``agent`` name it
returns the object to handle the message. The nine agents below declare their
metadata (capabilities, supported tasks, dependencies) but ship **no logic** —
``BaseAgent.handle`` is a no-op this phase. Their task names line up with the tools
in ``tool_registry`` so a future phase can wire handler -> tool with no re-shuffle.
"""

from __future__ import annotations

from vds.v2.agent_base import AgentInfo, BaseAgent


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        name = agent.info.name
        if name in self._agents:
            raise ValueError(f"agent already registered: {name}")
        self._agents[name] = agent

    def get(self, name: str) -> BaseAgent:
        if name not in self._agents:
            raise KeyError(f"unknown agent: {name}")
        return self._agents[name]

    def list(self) -> list[AgentInfo]:
        return [a.info for a in self._agents.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __len__(self) -> int:
        return len(self._agents)


class PlannerAgent(BaseAgent):
    info = AgentInfo(
        name="PlannerAgent",
        capabilities=["intent", "planning", "model_selection"],
        supported_tasks=["analyze_input", "select_models"],
        description="Analyzes the goal and input; selects models (future).",
    )


class DatasetAnalysisAgent(BaseAgent):
    info = AgentInfo(
        name="DatasetAnalysisAgent",
        capabilities=["inspection", "statistics"],
        supported_tasks=["inspect_dataset"],
        description="Inspects an existing/incoming dataset.",
    )


class ImportAgent(BaseAgent):
    info = AgentInfo(
        name="ImportAgent",
        capabilities=["import", "frame_extraction"],
        supported_tasks=["import_images", "import_video", "extract_frames"],
        description="Imports images/video and extracts frames.",
    )


class DetectionAgent(BaseAgent):
    info = AgentInfo(
        name="DetectionAgent",
        capabilities=["detection"],
        supported_tasks=["run_detection"],
        dependencies=["ImportAgent"],
        description="Runs object detection over imported images.",
    )


class SegmentationAgent(BaseAgent):
    info = AgentInfo(
        name="SegmentationAgent",
        capabilities=["segmentation"],
        supported_tasks=["run_segmentation"],
        dependencies=["DetectionAgent"],
        description="Generates masks for detected boxes.",
    )


class QualityAgent(BaseAgent):
    info = AgentInfo(
        name="QualityAgent",
        capabilities=["quality", "verification"],
        supported_tasks=["review_dataset"],
        description="Scores annotation quality / verdicts.",
    )


class ReviewAgent(BaseAgent):
    info = AgentInfo(
        name="ReviewAgent",
        capabilities=["human_in_the_loop"],
        supported_tasks=["await_approval"],
        description="Holds the run at the human-approval gate.",
    )


class MemoryAgent(BaseAgent):
    info = AgentInfo(
        name="MemoryAgent",
        capabilities=["memory"],
        supported_tasks=["record_memory"],
        description="Records the run into engineering memory (future).",
    )


class ExportAgent(BaseAgent):
    info = AgentInfo(
        name="ExportAgent",
        capabilities=["export"],
        supported_tasks=["export_dataset"],
        dependencies=["QualityAgent"],
        description="Exports the approved dataset (COCO/YOLO).",
    )


def default_registry() -> AgentRegistry:
    """A registry with all nine V2 agents registered."""
    reg = AgentRegistry()
    for agent in (
        PlannerAgent(),
        DatasetAnalysisAgent(),
        ImportAgent(),
        DetectionAgent(),
        SegmentationAgent(),
        QualityAgent(),
        ReviewAgent(),
        MemoryAgent(),
        ExportAgent(),
    ):
        reg.register(agent)
    return reg
