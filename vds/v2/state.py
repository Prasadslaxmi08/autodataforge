"""Session state (V2-20 §SESSION STATE).

Every goal submission becomes a session. The state is fully serializable: goal,
plan, per-step progress, agent activity (message ids), timing, errors, and memory
references. A run can be persisted and reloaded from this object alone.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from vds.v2.goal import Goal
from vds.v2.planner import ExecutionPlan


class SessionStatus(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SessionState(BaseModel):
    id: str
    goal: Goal
    plan: ExecutionPlan | None = None
    status: SessionStatus = SessionStatus.CREATED
    current_step: str | None = None
    completed_steps: list[str] = Field(default_factory=list)
    pending_steps: list[str] = Field(default_factory=list)
    failed_steps: list[str] = Field(default_factory=list)
    agent_activity: list[str] = Field(default_factory=list)  # AgentMessage ids
    started_at: float = 0.0
    finished_at: float | None = None
    errors: list[str] = Field(default_factory=list)
    memory_references: list[str] = Field(default_factory=list)
