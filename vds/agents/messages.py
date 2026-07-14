"""Agent-framework contracts — provider-agnostic request/response models.

These are the vocabulary every agent and every provider speaks. Nothing here
knows which LLM is behind the provider interface. Tool-calling and streaming
types are defined now so adding them later needs no schema change (future-ready,
per the phase brief).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ToolSpec(BaseModel):
    """A tool exposed to the model (function calling / MCP — future-ready)."""

    name: str
    description: str = ""
    parameters: dict = Field(default_factory=dict)  # JSON Schema


class ToolCall(BaseModel):
    id: str = ""
    name: str
    arguments: dict = Field(default_factory=dict)


class ImageContent(BaseModel):
    """A base64 image attached to a user message (multimodal / VLM input).

    Provider-neutral: each provider adapter renders this into its own image block
    format. Providers without vision simply ignore it (they only read `content`).
    """

    media_type: str = "image/png"
    data_base64: str


class Message(BaseModel):
    role: Role
    content: str = ""
    images: list[ImageContent] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class Conversation(BaseModel):
    """An ordered message list with fluent builders. Providers read `.messages`;
    they never construct this themselves."""

    messages: list[Message] = Field(default_factory=list)

    def _add(self, role: Role, content: str) -> Conversation:
        self.messages.append(Message(role=role, content=content))
        return self

    def system(self, content: str) -> Conversation:
        return self._add("system", content)

    def user(
        self, content: str, images: list[ImageContent] | None = None
    ) -> Conversation:
        self.messages.append(Message(role="user", content=content, images=images or []))
        return self

    def assistant(self, content: str) -> Conversation:
        return self._add("assistant", content)

    def tool_result(self, content: str, tool_call_id: str) -> Conversation:
        self.messages.append(
            Message(role="tool", content=content, tool_call_id=tool_call_id)
        )
        return self

    def last_user(self) -> str:
        return next((m.content for m in reversed(self.messages) if m.role == "user"), "")


class CompletionRequest(BaseModel):
    """Everything a provider needs for one completion. Provider-neutral."""

    model: str
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int | None = None
    response_schema: dict | None = None  # JSON Schema for structured output
    tools: list[ToolSpec] = Field(default_factory=list)
    stream: bool = False
    timeout_seconds: float = 60.0


class CompletionResponse(BaseModel):
    text: str = ""
    structured: dict | None = None  # set only if the provider parsed JSON itself
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    model: str = ""
    provider: str = ""
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0


class StreamChunk(BaseModel):
    delta: str = ""
    done: bool = False
