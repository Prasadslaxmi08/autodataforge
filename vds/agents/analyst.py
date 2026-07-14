"""Analyst Agent (System Design §2.9) — curation + quality, one persona.

Two entry points: `synthesize_feedback` (continuous during review, routed to the
Planner) and `audit` (pre-snapshot gate). Both are the same cognitive job — read
metrics/corrections, judge significance, write prioritized recommendations —
which is why the review merged Curator and Quality Auditor into one agent.

Bootstrap scope: the interface. Interpretation logic is Phase 3.
"""

from __future__ import annotations

from typing import Protocol

from vds.core.contracts import (
    FeedbackSummary,
    ProjectId,
    QualityMetrics,
    QualityReport,
    Verdict,
)


class AnalystAgent(Protocol):
    def synthesize_feedback(
        self, project_id: ProjectId, corrections: list[dict], verdicts: list[Verdict]
    ) -> FeedbackSummary:
        ...

    def audit(self, metrics: QualityMetrics) -> QualityReport: ...
