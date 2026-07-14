"""Message bus — the internal communication protocol (V2-20 §MESSAGE BUS).

Every agent speaks one object: ``AgentMessage``. Agents never call each other
directly; the orchestrator posts a message to the ``MessageBus``, the assigned
agent handles it, and the result is written back on the same message. The bus is
an append-only log, so a whole run is auditable from the messages alone — which is
the point of not hiding agent traffic inside method calls.
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum

from pydantic import BaseModel, Field


class MessageStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


class AgentMessage(BaseModel):
    """One unit of agent work. Serializable — the run's audit trail is a list of these."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    sender: str
    receiver: str
    task: str
    arguments: dict = Field(default_factory=dict)
    priority: int = 0
    timestamp: float = 0.0  # epoch seconds; stamped by the bus on post if unset
    reasoning: str = ""
    status: MessageStatus = MessageStatus.PENDING
    result: dict | None = None
    execution_time_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)


class MessageBus:
    """In-process append-only record of every AgentMessage."""

    def __init__(self) -> None:
        self._log: list[AgentMessage] = []

    def post(self, message: AgentMessage) -> AgentMessage:
        """Record a message, stamping the time if the caller left it unset."""
        if not message.timestamp:
            message.timestamp = time.time()
        self._log.append(message)
        return message

    def history(
        self, *, sender: str | None = None, receiver: str | None = None
    ) -> list[AgentMessage]:
        return [
            m
            for m in self._log
            if (sender is None or m.sender == sender)
            and (receiver is None or m.receiver == receiver)
        ]

    def __len__(self) -> int:
        return len(self._log)
