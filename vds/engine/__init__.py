"""L2 service — the Labeling Engine (deterministic; the demoted agent).

Responsibility: execute an approved LabelingPlan mechanically — per class run the
planned detector prompts/thresholds, NMS/merge, box-prompt the segmenter, embed
crops, classify, tile where the plan says so. It never invents strategy; it
escalates low yield to the orchestrator, which invokes the Planner.

Bootstrap scope: the service interface. Implementation is Phase 1.
"""

from __future__ import annotations

from typing import Protocol

from vds.core.contracts import ImageId, LabelingPlan


class YieldReport(Protocol):
    per_class_counts: dict[str, int]


class LabelingEngine(Protocol):
    def label_batch(self, plan: LabelingPlan, image_ids: list[ImageId]) -> YieldReport:
        ...
