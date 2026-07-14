"""Execution plan + deterministic template planner (V2-20 §EXECUTION PLAN).

The V2 Planner turns a Goal into a serializable ExecutionPlan the orchestrator
walks. This phase has **no planning intelligence**: it emits the fixed V1 pipeline
as a linear chain of agent steps, every step depending on the previous one, with a
single human-approval gate before export. Branching on input type, real model
selection, and frame-extraction strategy are future phases.

Note: this is *not* the V1 ``vds.agents.planner.PlannerAgent`` (which drafts a
LabelingPlan via the LLM). Different layer, different output.
"""

from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, Field

from vds.v2.goal import Goal


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"  # completed
    FAILED = "failed"
    SKIPPED = "skipped"
    AWAITING_APPROVAL = "awaiting_approval"
    RETRYING = "retrying"  # V2-22 execution
    CANCELLED = "cancelled"  # V2-22 execution


class TaskType(StrEnum):
    """What kind of dataset work the goal asks for (V2-21 §GOAL PARSER)."""

    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    REVIEW = "review"
    EXPORT = "export"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ReviewLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FrameStrategy(StrEnum):
    """Planner *recommendation* for video frame sampling (not the V1 extractor enum)."""

    NONE = "none"  # image input — no extraction
    EVERY_FRAME = "every_frame"
    EVERY_2 = "every_2"
    EVERY_5 = "every_5"
    EVERY_10 = "every_10"
    SCENE_CHANGE = "scene_change"
    ADAPTIVE = "adaptive"


class PlanStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"


class Recommendation(BaseModel):
    """One explainable recommendation. No hidden reasoning (V2-21 §REASONING)."""

    topic: str  # "model", "confidence", "frame_strategy", "dedup", ...
    value: str
    reason: str
    impact: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    alternative: str | None = None


class Alternative(BaseModel):
    """A recommended-vs-alternative tradeoff (V2-21 §ALTERNATIVES)."""

    topic: str
    recommended: str
    alternative: str
    tradeoff: str


class RequiredInput(BaseModel):
    name: str
    provided: bool
    note: str = ""


class PlanStep(BaseModel):
    id: str
    name: str  # Title
    agent: str  # Responsible Agent (must exist in the AgentRegistry)
    task: str  # the task/tool the agent runs
    arguments: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)  # Dependencies
    status: StepStatus = StepStatus.PENDING
    requires_approval: bool = False
    # --- V2-21 explainability (additive; defaulted for V2-20 compatibility) ---
    description: str = ""
    expected_output: str = ""
    reason: str = ""


class ExecutionPlan(BaseModel):
    """An ordered, serializable plan. The orchestrator advances it; nothing here
    executes anything itself. V2-21 enriched it with the reasoning the Planner
    produces — all fields default so V2-20's template plans stay valid."""

    goal_id: str
    steps: list[PlanStep] = Field(default_factory=list)
    # --- V2-21 plan metadata (additive) ---
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    status: PlanStatus = PlanStatus.DRAFT
    goal_text: str = ""
    task_type: TaskType = TaskType.UNKNOWN
    summary: str = ""
    current_state: str = ""
    required_inputs: list[RequiredInput] = Field(default_factory=list)
    recommended_model: str = ""
    recommended_segmentation: bool = False
    recommended_confidence: float = 0.0
    recommended_iou: float = 0.0
    frame_strategy: FrameStrategy = FrameStrategy.NONE
    estimated_dataset_size: int = 0
    estimated_runtime_seconds: float = 0.0
    estimated_review: ReviewLevel = ReviewLevel.MEDIUM
    warnings: list[str] = Field(default_factory=list)
    approvals_required: list[str] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    alternatives: list[Alternative] = Field(default_factory=list)
    reasoning: str = ""

    def ready(self, done: set[str]) -> list[PlanStep]:
        """Pending steps whose dependencies are all satisfied."""
        return [
            s
            for s in self.steps
            if s.status == StepStatus.PENDING and set(s.depends_on) <= done
        ]

    def get(self, step_id: str) -> PlanStep | None:
        return next((s for s in self.steps if s.id == step_id), None)


# The canonical V1 pipeline expressed as agent steps: (id, name, agent, task, approval).
# One deterministic template — real branching is a future phase.
_TEMPLATE: list[tuple[str, str, str, str, bool]] = [
    ("input_analysis", "Input Analysis", "PlannerAgent", "analyze_input", False),
    ("dataset_inspection", "Dataset Inspection", "DatasetAnalysisAgent", "inspect_dataset", False),
    ("import", "Import Strategy", "ImportAgent", "import_images", False),
    ("model_selection", "Model Selection", "PlannerAgent", "select_models", False),
    ("frame_extraction", "Frame Extraction Strategy", "ImportAgent", "extract_frames", False),
    ("detection", "Detection", "DetectionAgent", "run_detection", False),
    ("segmentation", "Segmentation", "SegmentationAgent", "run_segmentation", False),
    ("quality_review", "Quality Review", "QualityAgent", "review_dataset", False),
    ("human_approval", "Human Approval", "ReviewAgent", "await_approval", True),
    ("export", "Export", "ExportAgent", "export_dataset", False),
    ("record_memory", "Record Memory", "MemoryAgent", "record_memory", False),
]


class Planner:
    """Deterministic template planner. Same goal -> same plan."""

    def plan(self, goal: Goal) -> ExecutionPlan:
        steps: list[PlanStep] = []
        prev: str | None = None
        for step_id, name, agent, task, approval in _TEMPLATE:
            steps.append(
                PlanStep(
                    id=step_id,
                    name=name,
                    agent=agent,
                    task=task,
                    arguments=dict(goal.params),
                    depends_on=[prev] if prev else [],
                    requires_approval=approval,
                )
            )
            prev = step_id
        return ExecutionPlan(goal_id=goal.id, steps=steps)
