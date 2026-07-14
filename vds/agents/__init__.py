"""L3 — the three agents (Planner, Verifier, Analyst) + orchestrator.

The only module permitted to call an LLM for a *decision*. Quarantining judgment
here keeps every other module deterministic. Agents never message each other;
they read and write typed contract rows arbitrated by the orchestrator.
"""

from vds.agents.analyst import AnalystAgent
from vds.agents.base import Agent
from vds.agents.llm import LLMClient
from vds.agents.messages import (
    CompletionRequest,
    CompletionResponse,
    Conversation,
    ToolSpec,
)
from vds.agents.orchestrator import Orchestrator, guard_phase
from vds.agents.planner import PlannerAgent
from vds.agents.providers.base import LLMProvider, load_provider
from vds.agents.verifier import VerifierAgent
from vds.agents.vlm_verifier import LLMVerifier

__all__ = [
    "Agent",
    "AnalystAgent",
    "CompletionRequest",
    "CompletionResponse",
    "Conversation",
    "LLMClient",
    "LLMProvider",
    "LLMVerifier",
    "Orchestrator",
    "PlannerAgent",
    "ToolSpec",
    "VerifierAgent",
    "guard_phase",
    "load_provider",
]
