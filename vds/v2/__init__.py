"""Version 2 — the Goal-Driven Multi-Agent layer (Phase V2-20).

This package sits *above* the frozen Version 1 platform. Nothing here reimplements
a backend service: the existing services are wrapped as **tools** (see
``tool_registry``) and driven by agents an orchestrator schedules from a plan.

Layering (top calls down, never up)::

    GUI -> DatasetEngineerAgent -> TaskOrchestrator -> AgentRegistry
                                                     -> ToolRegistry -> BackendController -> V1 services

Scope of this phase is *architecture only*: the seams, contracts and state
machine. No planning intelligence, no autonomous decisions, no model selection,
no memory reasoning, no MCP — those are future phases. Every agent handler is a
declared-metadata no-op today.

Note: the phase brief named this package ``vde/agents/``. It lives at ``vds/v2/``
so it stays inside the one installed ``vds`` package (no second top-level package,
no pyproject change) and never collides with the V1 ``vds/agents`` LLM layer.
"""

from vds.v2.agent_base import AgentInfo, AgentStatus, BaseAgent
from vds.v2.dataset_engineer import DatasetEngineerAgent
from vds.v2.decision import (
    DatasetMetadata,
    Decision,
    DecisionAgent,
    DecisionArea,
    DecisionReport,
    decision_view,
)
from vds.v2.execution import (
    ApprovalHandler,
    ExecutionAgent,
    ExecutionContext,
    ExecutionError,
    ExecutionRunner,
    ExecutionSummary,
    ExecutionTimeline,
    FailureCategory,
    GateReason,
    ProgressTracker,
    RecoveryHandler,
)
from vds.v2.goal import Goal, new_goal
from vds.v2.goal_parser import GoalParser, ParsedGoal
from vds.v2.memory_agent import MemoryAgent, MemoryExperience, memory_view
from vds.v2.messages import AgentMessage, MessageBus, MessageStatus
from vds.v2.orchestrator import TaskOrchestrator as PlanStepOrchestrator
from vds.v2.planner import (
    Alternative,
    ExecutionPlan,
    FrameStrategy,
    Planner,
    PlanStatus,
    PlanStep,
    Recommendation,
    RequiredInput,
    ReviewLevel,
    StepStatus,
    TaskType,
)
from vds.v2.planner_agent import (
    PlannerAgent,
    PlanSessionStore,
    PlanValidationError,
    ValidationEngine,
    plan_view,
)
from vds.v2.recommendations import PlanContext, RecommendationEngine, RecommendationResult
from vds.v2.registry import AgentRegistry, default_registry
from vds.v2.state import SessionState, SessionStatus
from vds.v2.task_orchestrator import (
    STAGES,
    TaskContext,
    TaskEvent,
    TaskOrchestrator,
    TaskState,
    task_view,
)
from vds.v2.tool_registry import Tool, ToolRegistry, default_tools

__all__ = [
    "AgentInfo",
    "AgentMessage",
    "AgentRegistry",
    "AgentStatus",
    "Alternative",
    "ApprovalHandler",
    "BaseAgent",
    "DatasetEngineerAgent",
    "DatasetMetadata",
    "Decision",
    "DecisionAgent",
    "DecisionArea",
    "DecisionReport",
    "ExecutionAgent",
    "ExecutionContext",
    "ExecutionError",
    "ExecutionPlan",
    "ExecutionRunner",
    "ExecutionSummary",
    "ExecutionTimeline",
    "FailureCategory",
    "FrameStrategy",
    "GateReason",
    "Goal",
    "GoalParser",
    "MemoryAgent",
    "MemoryExperience",
    "MessageBus",
    "MessageStatus",
    "ParsedGoal",
    "PlanContext",
    "PlanSessionStore",
    "PlanStatus",
    "PlanStep",
    "PlanStepOrchestrator",
    "PlanValidationError",
    "Planner",
    "PlannerAgent",
    "ProgressTracker",
    "Recommendation",
    "RecommendationEngine",
    "RecommendationResult",
    "RecoveryHandler",
    "RequiredInput",
    "ReviewLevel",
    "STAGES",
    "SessionState",
    "SessionStatus",
    "StepStatus",
    "TaskContext",
    "TaskEvent",
    "TaskOrchestrator",
    "TaskState",
    "TaskType",
    "Tool",
    "ToolRegistry",
    "ValidationEngine",
    "decision_view",
    "default_registry",
    "default_tools",
    "memory_view",
    "new_goal",
    "plan_view",
    "task_view",
]
