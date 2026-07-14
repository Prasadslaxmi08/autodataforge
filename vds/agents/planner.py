"""Planner Agent (System Design §2.9) — the flagship.

Turns a brief + image samples into an approved LabelingPlan through grounded,
tool-using dialogue; revises the plan in response to feedback or low yield. This
is genuine agency: ambiguity, tool use (VisionJudge inspection, embedding
search), and plan revision.

The LLM PlannerAgent (draft/revise a LabelingPlan) is Phase 2. Phase 1 ships the
deterministic ExecutionPlanner below, which produces the ProcessingPlan every
later stage follows: counts, batches, model selection, GPU budget, time estimate,
export format.
"""

from __future__ import annotations

import math
from typing import Protocol

from vds.config.settings import Settings
from vds.core.contracts import FeedbackSummary, LabelingPlan, ProcessingPlan, ProjectId


class PlannerAgent(Protocol):
    def draft(self, project_id: ProjectId) -> LabelingPlan:
        """Propose a LabelingPlan (with open questions) from the brief + samples."""
        ...

    def revise(self, plan: LabelingPlan, feedback: FeedbackSummary) -> LabelingPlan:
        """Produce the next plan version addressing systematic failures."""
        ...


# Per-image time estimate (seconds) for the builtin CPU backend, used only to
# give the user an ETA. Calibrated from benchmark runs; see benchmarks/.
_SECONDS_PER_IMAGE = 0.05
_DETECTOR_CONF_FLOOR = 0.30  # drop detections below this before segmentation


class ExecutionPlanner:
    """Deterministic Phase-1 planner. Same inputs -> same plan."""

    def __init__(self, settings: Settings) -> None:
        self._s = settings

    def plan(
        self, project_id: ProjectId, image_count: int, classes: list[str] | None = None
    ) -> ProcessingPlan:
        batch = max(1, self._s.runtime.batch_size)
        return ProcessingPlan(
            project_id=project_id,
            version=1,
            image_count=image_count,
            classes=classes or ["object"],
            detector=self._s.models.detector,
            segmenter=self._s.models.segmenter,
            confidence_threshold=_DETECTOR_CONF_FLOOR,
            batch_size=batch,
            num_batches=math.ceil(image_count / batch) if image_count else 0,
            gpu_budget_mb=self._s.gpu.vram_budget_mb,
            export_format=self._s.export.default_format,
            estimated_seconds=round(image_count * _SECONDS_PER_IMAGE, 2),
        )
