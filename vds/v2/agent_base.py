"""Agent base + registration metadata (V2-20 §AGENT REGISTRY).

Every V2 agent subclasses ``BaseAgent`` and declares an ``AgentInfo``. This phase
builds the seam only: ``handle`` is a no-op that echoes the task. Concrete agents
gain real behaviour (driving tools from the registry) in later phases — the base
gives them a uniform contract without any of that logic today.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from vds.v2.messages import AgentMessage


class AgentStatus(StrEnum):
    READY = "ready"
    RUNNING = "running"
    DISABLED = "disabled"


class AgentInfo(BaseModel):
    """What an agent advertises to the registry. Serializable for status views."""

    name: str
    capabilities: list[str] = Field(default_factory=list)
    supported_tasks: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: AgentStatus = AgentStatus.READY
    version: str = "2.0.0"
    description: str = ""


class BaseAgent:
    """Base for every V2 agent. Subclasses set ``info``; the framework ships no logic."""

    #: Subclasses override with their own AgentInfo.
    info: AgentInfo = AgentInfo(name="base")

    def handle(self, message: AgentMessage) -> dict:
        """Execute one task. Architecture phase: no-op that reports what it *would*
        do. Real tool-driven behaviour lands in a future phase."""
        return {"agent": self.info.name, "task": message.task, "status": "noop"}
