"""L2 service — triage and splits (deterministic; it's math).

Responsibility: TriageScore per candidate = f(confidence, labeler-verifier
disagreement, rare-class boost, embedding diversity); cluster grouping for grid
review (pgvector); dedup-aware stratified train/val/test splits.

Bootstrap scope: the service interface. Scoring math is Phase 1.
"""

from __future__ import annotations

from typing import Protocol

from vds.core.contracts import ProjectId, TriageScore


class CurationService(Protocol):
    def score_candidates(self, project_id: ProjectId) -> list[TriageScore]: ...
    def build_clusters(self, project_id: ProjectId) -> list[list[str]]: ...
    def make_splits(self, project_id: ProjectId, ratios: tuple[float, float, float]) -> None: ...
