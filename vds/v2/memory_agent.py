"""MemoryAgent — experience recall + storage layer (V2-24).

Sits beside the Planner/Decision/Execution agents but runs no tools, plans
nothing, and executes nothing: it only *remembers*. Before planning it recalls
similar past jobs; after a successful run it records the settings that worked.

It reuses the existing Engineering Memory stack wholesale — the same JSON store
(`vds.memory.store.MemoryStore`), the same deterministic `SimilarityEngine`, and
the same `EngineeringMemory` schema. No second database, no duplicated storage.
Fields the V2 pipeline produces that the shared schema has no column for (frame
strategy, review level, goal text, IoU, annotation count, success) ride in the
fingerprint's `environment` bag — serialized, but never scored by similarity — so
matching stays clean and the shared V1 schema is left untouched.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field

from vds.memory.schema import (
    AnalystConclusions,
    BenchmarkSummary,
    DatasetFingerprint,
    EngineeringMemory,
    ExecutionMetrics,
    PlannerDecisions,
    VerificationOutcomes,
)
from vds.memory.similarity import MemoryMatch, SimilarityEngine
from vds.memory.store import MemoryStore
from vds.v2.decision import DatasetMetadata, DecisionArea, DecisionReport
from vds.v2.execution import ExecutionSummary
from vds.v2.goal import Goal
from vds.v2.goal_parser import GoalParser, ParsedGoal
from vds.v2.planner import ExecutionPlan

_RES_MP = {"low": 0.3, "medium": 2.0, "high": 8.0}


def _res_mp(resolution: str | None) -> float:
    """Megapixels from a resolution tag or "WxH" string; -1 (unknown) if absent."""
    if not resolution:
        return -1.0
    r = resolution.lower()
    if r in _RES_MP:
        return _RES_MP[r]
    if "x" in r:
        try:
            w, h = (int(p) for p in r.split("x")[:2])
        except ValueError:
            return -1.0
        return round(w * h / 1_000_000, 2)
    return -1.0


def _scene_type(parsed: ParsedGoal) -> str:
    """Composite domain tag so like-domain jobs cluster on the (categorical)
    scene_type the SimilarityEngine already scores: e.g. thermal_aerial."""
    sensor = "thermal" if parsed.thermal else "rgb"
    platform = "aerial" if parsed.drone else "ground"
    return f"{sensor}_{platform}"


class MemoryExperience(BaseModel):
    """What the Planner/Decision agents receive from memory before a job. Ranked
    matches plus human-readable, evidence-backed hints; the caller may ignore all
    of it (memory advises, it never decides)."""

    matches: list[MemoryMatch] = Field(default_factory=list)
    similarity_score: float = 0.0  # best match
    confidence: float = 0.0  # mean similarity across matches
    recommendations: list[str] = Field(default_factory=list)
    successful_settings: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    lessons: list[str] = Field(default_factory=list)
    note: str = "No similar past projects in engineering memory."

    @property
    def has_experience(self) -> bool:
        return bool(self.matches)


class MemoryAgent:
    """Recall + store, over the existing Engineering Memory store. Never executes,
    never plans, never runs models."""

    def __init__(self, store: MemoryStore | None = None, min_similarity: float = 0.5) -> None:
        self._store = store or MemoryStore()  # shared engineering_memory.json by default
        self._sim = SimilarityEngine(min_score=min_similarity)
        self._parser = GoalParser()

    # --- fingerprint ---------------------------------------------------
    def _fingerprint(
        self, goal: Goal, metadata: DatasetMetadata | None = None, environment: dict | None = None
    ) -> DatasetFingerprint:
        parsed = self._parser.parse(goal)
        env = {
            "goal": goal.text,
            "sensor": "thermal" if parsed.thermal else "rgb",
            "platform": "aerial" if parsed.drone else "ground",
            "task_type": parsed.task_type.value,
        }
        if environment:
            env.update({k: str(v) for k, v in environment.items()})
        classes = {c: 1 for c in parsed.target_classes}
        size = -1
        density = -1.0
        if metadata:
            if not classes and metadata.existing_classes:
                classes = {c: 1 for c in metadata.existing_classes}
            size = metadata.image_count or -1
            density = float(metadata.historical_stats.get("avg_objects_per_image", -1.0))
        return DatasetFingerprint(
            resolution_mp=_res_mp(metadata.resolution if metadata else None),
            dataset_size=size,
            class_distribution=classes,
            scene_type=_scene_type(parsed),
            environment=env,
            scene_density=density,
        )

    # --- recall (pre-planning) -----------------------------------------
    def recall(self, goal: Goal, metadata: DatasetMetadata | None = None, top_k: int = 3) -> MemoryExperience:
        """Given a new goal (and whatever metadata is known), return the most
        similar past projects with the settings that worked and the mistakes to
        avoid. Deterministic — same inputs, same output."""
        fp = self._fingerprint(goal, metadata)
        matches = self._sim.search(fp, self._store.all(), top_k=top_k)
        if not matches:
            return MemoryExperience()
        recs, settings, warnings, lessons = self._aggregate(matches)
        best = matches[0]
        scores = [m.score for m in matches]
        return MemoryExperience(
            matches=matches,
            similarity_score=best.score,
            confidence=round(sum(scores) / len(scores), 3),
            recommendations=recs,
            successful_settings=settings,
            warnings=warnings,
            lessons=lessons,
            note=f"Found {len(matches)} similar past project(s); closest match {best.memory.id} "
            f"(similarity {best.score}). Prior experience is available.",
        )

    @staticmethod
    def _aggregate(matches: list[MemoryMatch]) -> tuple[list[str], dict, list[str], list[str]]:
        best = matches[0]
        d = best.memory.planner_decisions
        env = best.memory.dataset_fingerprint.environment
        rt = best.memory.execution_metrics.runtime_seconds
        settings = {
            "model": d.detector,
            "confidence": d.confidence_threshold,
            "iou": env.get("iou"),
            "segmentation": d.segmentation_enabled,
            "frame_strategy": env.get("frame_strategy"),
            "review_level": env.get("review_level"),
            "export_format": best.memory.execution_metrics.export_format or d.export_strategy,
        }
        recs = [f"Use detection confidence {d.confidence_threshold} (worked on {best.memory.id})"]
        if env.get("frame_strategy"):
            recs.append(f"Recommended frame sampling: {env['frame_strategy']}")
        if env.get("review_level"):
            recs.append(f"Suggested review level: {env['review_level']}")
        seen: set[str] = set()
        for m in matches:  # validated engineering recs across matches, dedup by action
            for r in m.memory.engineering_recommendations:
                if r.action not in seen:
                    seen.add(r.action)
                    recs.append(f"{r.action} — {r.expected_impact}")
        if rt:
            recs.append(f"Estimated runtime ~{round(rt / 60, 1)} min (from {best.memory.id})")
        warnings: list[str] = []
        lessons: list[str] = []
        for m in matches:
            warnings.extend(m.memory.analyst_conclusions.bottlenecks)
            lessons.extend(m.memory.analyst_conclusions.improvement_opportunities)
            if m.memory.benchmark_summary.quality_score < 0.5:
                lessons.append(
                    f"{m.memory.id}: low quality ({m.memory.benchmark_summary.quality_score}) — revisit settings"
                )
        dedup = lambda xs: list(dict.fromkeys(xs))  # noqa: E731 — order-preserving unique
        return recs, settings, dedup(warnings), dedup(lessons)

    # --- store (post-execution) ----------------------------------------
    def record(
        self,
        goal: Goal,
        decision_report: DecisionReport,
        execution_summary: ExecutionSummary,
        *,
        project_id: str,
        created_at: str,
        metadata: DatasetMetadata | None = None,
        plan: ExecutionPlan | None = None,
        export_summary: dict | None = None,
    ) -> EngineeringMemory | None:
        """Persist a finished run as reusable experience. Only completed runs are
        stored (brief: "after successful execution"); a cancelled/failed run returns
        None. ``created_at`` is passed in so this stays deterministic — the caller
        (DatasetEngineerAgent) stamps the clock."""
        if execution_summary.status != "completed":  # ponytail: SessionStatus.COMPLETED.value
            return None

        # Prefer the enriched plan for structured params; fall back to the report.
        conf = plan.recommended_confidence if plan else _dec_float(decision_report, DecisionArea.DETECTION_CONFIDENCE)
        iou = plan.recommended_iou if plan else _dec_float(decision_report, DecisionArea.IOU_THRESHOLD)
        seg = plan.recommended_segmentation if plan else _dec_bool(decision_report, DecisionArea.SEGMENTATION)
        model = (plan.recommended_model if plan and plan.recommended_model else "unknown")
        frame = plan.frame_strategy.value if plan else _dec_str(decision_report, DecisionArea.FRAME_SAMPLING)
        export_fmt = _export_format(export_summary) or _dec_str(decision_report, DecisionArea.EXPORT_FORMAT)
        success = execution_summary.failed == 0 and not execution_summary.errors
        warnings = list(dict.fromkeys([*decision_report.warnings, *execution_summary.warnings]))

        env = {
            "iou": f"{iou:.2f}",
            "frame_strategy": frame,
            "review_level": decision_report.recommended_review,
            "annotation_count": str(decision_report.expected_annotation_count),
            "success": str(success).lower(),
            "export_format": export_fmt,
        }
        fp = self._fingerprint(goal, metadata, environment=env)
        ident = hashlib.sha256(f"{fp.hash()}|{project_id}|{created_at}".encode()).hexdigest()[:12]
        memory = EngineeringMemory(
            id=f"mem_{ident}",
            created_at=created_at,
            project_id=project_id,
            source="memory_agent.v2",
            dataset_fingerprint=fp,
            planner_decisions=PlannerDecisions(
                detector=model,
                segmentation_enabled=seg,
                confidence_threshold=conf,
                batch_size=_dec_int(decision_report, DecisionArea.BATCH_SIZE, 1),
                export_strategy=export_fmt or "coco",
            ),
            execution_metrics=ExecutionMetrics(
                throughput_ips=0.0,
                runtime_seconds=execution_summary.elapsed_seconds,
                export_format=export_fmt,
                export_validated=success,
            ),
            analyst_conclusions=AnalystConclusions(
                bottlenecks=warnings,
                improvement_opportunities=list(decision_report.suggestions),
                confidence=decision_report.overall_confidence,
            ),
            verification_outcomes=VerificationOutcomes(),
            benchmark_summary=BenchmarkSummary(
                throughput_ips=0.0,
                review_rate=0.0,
                approval_rate=1.0 if success else 0.0,
                avg_confidence=conf,
                quality_score=1.0 if success else 0.3,
            ),
            validation_status="validated" if success else "provisional",
            confidence=decision_report.overall_confidence,
        )
        return self._store.add(memory)  # dedup + versioning handled by the shared store


# --- DecisionReport field readers (fallbacks when no enriched plan is passed) ---
def _dec_str(report: DecisionReport, area: DecisionArea, default: str = "") -> str:
    d = report.get(area)
    return d.value if d else default


def _dec_float(report: DecisionReport, area: DecisionArea, default: float = 0.0) -> float:
    d = report.get(area)
    try:
        return float(d.value) if d else default
    except ValueError:
        return default


def _dec_int(report: DecisionReport, area: DecisionArea, default: int = 0) -> int:
    d = report.get(area)
    try:
        return int(d.value) if d else default
    except ValueError:
        return default


def _dec_bool(report: DecisionReport, area: DecisionArea) -> bool:
    d = report.get(area)
    return bool(d) and d.value.lower() in ("true", "1", "yes", "enable")


def _export_format(export_summary: dict | None) -> str:
    if not export_summary:
        return ""
    val = export_summary.get("format") or export_summary.get("formats") or ""
    return ",".join(val) if isinstance(val, list) else str(val)


def memory_view(exp: MemoryExperience) -> dict:
    """GUI Memory Summary panel data (V2-24 §GUI). Data only — a future phase
    renders it in Qt, exactly like ``decision_view``."""
    return {
        "memory_summary": exp.note,
        "similar_projects": [
            {
                "id": m.memory.id,
                "similarity": m.score,
                "goal": m.memory.dataset_fingerprint.environment.get("goal", ""),
                "why": m.explain(),
            }
            for m in exp.matches
        ],
        "lessons_learned": list(exp.lessons),
        "recommendations": list(exp.recommendations),
        "previous_results": [
            {
                "id": m.memory.id,
                "success": m.memory.dataset_fingerprint.environment.get("success"),
                "runtime_seconds": m.memory.execution_metrics.runtime_seconds,
                "quality": m.memory.benchmark_summary.quality_score,
                "review_level": m.memory.dataset_fingerprint.environment.get("review_level"),
                "annotations": m.memory.dataset_fingerprint.environment.get("annotation_count"),
            }
            for m in exp.matches
        ],
        "successful_settings": exp.successful_settings,
        "confidence": exp.confidence,
        "similarity_score": exp.similarity_score,
    }
