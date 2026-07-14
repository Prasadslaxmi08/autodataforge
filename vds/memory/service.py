"""Engineering Memory facade (Phase 10).

The single object the Planner and Analyst talk to. Owns the store, the similarity
engine, and the trend analyzer, and exposes the two integration verbs:

  recall(fingerprint)      -> what past experience is relevant to THIS dataset
  record_execution(...)    -> turn a finished run into reusable engineering knowledge

Deterministic and explainable throughout — every recalled memory comes with the
reasons it matched, and every stored memory is built only from measured outputs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from vds.core.contracts import ExecutionReport
from vds.memory.builder import build_memory
from vds.memory.schema import DatasetFingerprint, EngineeringMemory, MemoryRecommendation
from vds.memory.similarity import MemoryMatch, SimilarityEngine
from vds.memory.store import MEMORY_PATH, MemoryStore
from vds.memory.trends import TrendAnalyzer


class MemoryGuidance(BaseModel):
    """What the Planner receives from memory: ranked prior experience plus
    aggregated, evidence-backed hints. `note` is the sentence the Planner echoes
    to explain (or disclaim) memory influence."""

    matches: list[MemoryMatch] = Field(default_factory=list)
    successful_plans: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    historical_recommendations: list[MemoryRecommendation] = Field(default_factory=list)
    historical_review_rates: list[float] = Field(default_factory=list)
    note: str = "No similar datasets found in engineering memory."

    @property
    def has_experience(self) -> bool:
        return bool(self.matches)

    def render(self) -> str:
        if not self.matches:
            return self.note
        lines = [self.note, "", "Similar past datasets:"]
        for m in self.matches:
            d = m.memory.planner_decisions
            lines.append(
                f"- {m.explain()}; that run used detector={d.detector}, seg={d.segmentation_enabled}, "
                f"conf={d.confidence_threshold}, tiling={d.tiling} -> review_rate "
                f"{m.memory.execution_metrics.review_rate}, quality {m.memory.benchmark_summary.quality_score}"
            )
        if self.historical_recommendations:
            lines += ["", "Historical validated recommendations:"]
            lines += [f"- {r.action} (impact: {r.expected_impact})" for r in self.historical_recommendations[:5]]
        return "\n".join(lines)


class EngineeringMemoryService:
    def __init__(self, path=MEMORY_PATH, min_similarity: float = 0.5) -> None:
        self._store = MemoryStore(path)
        self._sim = SimilarityEngine(min_score=min_similarity)
        self._trends = TrendAnalyzer()

    # --- read / query ---
    def all(self) -> list[EngineeringMemory]:
        return self._store.all()

    def similar(self, fingerprint: DatasetFingerprint, top_k: int = 3) -> list[MemoryMatch]:
        return self._sim.search(fingerprint, self._store.all(), top_k=top_k)

    def recall(self, fingerprint: DatasetFingerprint, top_k: int = 3) -> MemoryGuidance:
        """Planner-facing: 'have we processed a similar dataset?' Returns ranked
        matches, aggregated successes/failures, and validated historical advice —
        or an explicit no-experience note."""
        matches = self.similar(fingerprint, top_k=top_k)
        if not matches:
            return MemoryGuidance()
        successful = [m.memory.planner_decisions.model_dump_json()
                      for m in matches if m.memory.benchmark_summary.quality_score >= 0.7]
        failures = [f"{m.memory.id}: quality {m.memory.benchmark_summary.quality_score}, "
                    f"review {m.memory.execution_metrics.review_rate}"
                    for m in matches if m.memory.benchmark_summary.quality_score < 0.5]
        recs: list[MemoryRecommendation] = []
        seen: set[str] = set()
        for m in matches:  # validated recs only were stored; dedup by action
            for r in m.memory.engineering_recommendations:
                if r.action not in seen:
                    seen.add(r.action)
                    recs.append(r)
        best = matches[0]
        return MemoryGuidance(
            matches=matches, successful_plans=successful, failures=failures,
            historical_recommendations=recs,
            historical_review_rates=[m.memory.execution_metrics.review_rate for m in matches],
            note=f"Found {len(matches)} similar past dataset(s); closest match {best.memory.id} "
                 f"(similarity {best.score}). Prior experience informs this plan.",
        )

    # --- write (Analyst integration) ---
    def record_execution(
        self, execution: ExecutionReport, created_at: str, **kwargs
    ) -> EngineeringMemory:
        """Analyst-facing: after a completed run, persist reusable engineering
        knowledge. Only validated recommendations are stored (builder gates them);
        history is never overwritten and duplicates are suppressed (store)."""
        memory = build_memory(execution, created_at, **kwargs)
        return self._store.add(memory)

    # --- reports ---
    def trend_report(self) -> str:
        return self._trends.trend_report(self._store.all())

    def engineering_report(self) -> str:
        return self._trends.engineering_report(self._store.all())
