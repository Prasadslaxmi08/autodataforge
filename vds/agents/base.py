"""Agent base class (Agent Framework).

The foundation future agents (Planner, Analyst, ...) subclass. It owns an
LLMClient and an optional system prompt, and exposes plain completion helpers.
It contains **no reasoning and no prompts** — those belong to the concrete agents
built in later phases. This class exists so those agents inherit provider-agnostic
transport, retries, and structured output without reimplementing any of it.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from vds.agents.llm import LLMClient
from vds.agents.messages import CompletionResponse, Conversation, ToolSpec

T = TypeVar("T", bound=BaseModel)


class Agent:
    #: Subclasses set their system prompt here (Phase 2+). Empty by default —
    #: the framework ships no prompts.
    system_prompt: str = ""

    def __init__(self, client: LLMClient, system_prompt: str | None = None) -> None:
        self._client = client
        self._system = system_prompt if system_prompt is not None else self.system_prompt

    def new_conversation(self) -> Conversation:
        """A fresh conversation seeded with the agent's system prompt (if any)."""
        convo = Conversation()
        if self._system:
            convo.system(self._system)
        return convo

    def complete(
        self, user_prompt: str, *, tools: list[ToolSpec] | None = None
    ) -> CompletionResponse:
        return self._client.complete(self.new_conversation().user(user_prompt), tools=tools)

    def complete_structured(self, user_prompt: str, schema: type[T]) -> T:
        return self._client.complete_structured(
            self.new_conversation().user(user_prompt), schema
        )
