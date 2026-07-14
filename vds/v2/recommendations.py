"""Recommendation engine (V2-21 §RECOMMENDATIONS, §MODEL/FRAME/REVIEW).

Rule-based, deterministic. The Planner *recommends* — it never executes. Every
recommendation carries reason / impact / confidence / alternative so nothing is
hidden. The Detection Agent (a future phase) is free to override.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from vds.v2.goal_parser import ParsedGoal
from vds.v2.planner import Alternative, FrameStrategy, Recommendation, ReviewLevel, TaskType

# Per-image labeling time (s) for the builtin CPU backend — an ETA only.
# Calibrated from benchmarks/; same figure the V1 ExecutionPlanner uses.
_SECONDS_PER_IMAGE = 0.05
_LARGE_DATASET = 2000  # above this, duplicate removal pays for itself
_LONG_VIDEO_S = 300.0

# Frame stride implied by each recommendation, for size estimation.
_STRIDE = {
    FrameStrategy.EVERY_FRAME: 1,
    FrameStrategy.EVERY_2: 2,
    FrameStrategy.EVERY_5: 5,
    FrameStrategy.EVERY_10: 10,
    FrameStrategy.SCENE_CHANGE: 10,  # rough: ~10% of frames survive the filter
    FrameStrategy.ADAPTIVE: 5,
}


class PlanContext(BaseModel):
    """What little the Planner knows about the inputs up front. All optional."""

    image_count: int | None = None
    video_duration_seconds: float | None = None
    fps: float | None = None
    expected_density: str = "medium"  # "low" | "medium" | "high"
    resolution: str = "medium"  # "low" | "medium" | "high"
    small_objects: bool = False


class RecommendationResult(BaseModel):
    model: str
    segmentation: bool
    confidence: float
    iou: float
    frame_strategy: FrameStrategy
    dedup: bool
    estimated_dataset_size: int
    estimated_runtime_seconds: float
    review_level: ReviewLevel
    recommendations: list[Recommendation] = Field(default_factory=list)
    alternatives: list[Alternative] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RecommendationEngine:
    def recommend(self, parse: ParsedGoal, ctx: PlanContext | None = None) -> RecommendationResult:
        ctx = ctx or PlanContext()
        recs: list[Recommendation] = []
        alts: list[Alternative] = []
        warnings: list[str] = []

        segmentation = parse.task_type in (TaskType.SEGMENTATION, TaskType.MIXED)
        model = self._model(parse, ctx, segmentation, recs, alts)
        confidence = self._confidence(parse, recs, warnings)
        iou = self._iou(parse, recs)
        frame_strategy = self._frames(parse, ctx, recs)
        size = self._size(parse, ctx, frame_strategy)
        dedup = self._dedup(size, recs)
        review = self._review(parse, segmentation)

        return RecommendationResult(
            model=model,
            segmentation=segmentation,
            confidence=confidence,
            iou=iou,
            frame_strategy=frame_strategy,
            dedup=dedup,
            estimated_dataset_size=size,
            estimated_runtime_seconds=round(size * _SECONDS_PER_IMAGE, 2),
            review_level=review,
            recommendations=recs,
            alternatives=alts,
            warnings=warnings,
        )

    # --- individual rules ---------------------------------------------
    def _model(self, parse, ctx, segmentation, recs, alts) -> str:
        if segmentation:
            model, alt, trade = "YOLO11-seg", "YOLO11n-seg", "Faster; lower mask quality"
            reason = "Goal requires masks."
        elif ctx.small_objects or ctx.resolution == "high":
            model, alt, trade = "YOLO11m", "YOLO11s", "Faster; lower accuracy on small objects"
            reason = "High-resolution / small objects need a larger backbone."
        else:
            model, alt, trade = "YOLO11s", "YOLO11n", "Faster; lower accuracy"
            reason = "General objects; balanced speed/accuracy."
        recs.append(Recommendation(
            topic="model", value=model, reason=reason,
            impact="Sets detection accuracy vs speed.", confidence=0.7, alternative=alt))
        alts.append(Alternative(topic="model", recommended=model, alternative=alt, tradeoff=trade))
        return model

    def _confidence(self, parse, recs, warnings) -> float:
        if parse.thermal:
            warnings.append("Thermal input: lower object contrast may reduce detection recall.")
            recs.append(Recommendation(
                topic="confidence", value="0.20",
                reason="Thermal images have lower contrast.",
                impact="Recovers low-confidence true positives.", confidence=0.65,
                alternative="0.30"))
            return 0.20
        recs.append(Recommendation(
            topic="confidence", value="0.30", reason="Default detector floor.",
            impact="Balances precision and recall.", confidence=0.6, alternative="0.25"))
        return 0.30

    def _iou(self, parse, recs) -> float:
        iou = 0.50 if (parse.drone or parse.task_type == TaskType.DETECTION) else 0.45
        recs.append(Recommendation(
            topic="iou", value=f"{iou:.2f}", reason="NMS overlap threshold for the task.",
            impact="Higher keeps more overlapping boxes.", confidence=0.55, alternative="0.45"))
        return iou

    def _frames(self, parse, ctx, recs) -> FrameStrategy:
        if parse.modality != "video":
            return FrameStrategy.NONE
        if ctx.expected_density == "high":
            strat, reason = FrameStrategy.EVERY_2, "Dense scene — sample more often."
        elif (ctx.video_duration_seconds or 0) > _LONG_VIDEO_S or ctx.expected_density == "low":
            strat, reason = FrameStrategy.EVERY_10, "Long or sparse video — sample sparsely."
        else:
            strat, reason = FrameStrategy.EVERY_5, "Balanced sampling for typical footage."
        recs.append(Recommendation(
            topic="frame_strategy", value=strat.value, reason=reason,
            impact="Controls dataset size and annotation cost.", confidence=0.6,
            alternative=FrameStrategy.ADAPTIVE.value))
        return strat

    def _size(self, parse, ctx, strat: FrameStrategy) -> int:
        if parse.modality == "video" and ctx.video_duration_seconds and ctx.fps:
            total = ctx.video_duration_seconds * ctx.fps
            return int(total / _STRIDE.get(strat, 5))
        return int(ctx.image_count or 0)

    def _dedup(self, size: int, recs) -> bool:
        if size > _LARGE_DATASET:
            recs.append(Recommendation(
                topic="dedup", value="enabled",
                reason="Large dataset — near-duplicate frames inflate annotation cost.",
                impact="Fewer redundant images to review.", confidence=0.7, alternative="disabled"))
            return True
        return False

    def _review(self, parse: ParsedGoal, segmentation: bool) -> ReviewLevel:
        score = 1  # 0 low, 1 medium, 2 high
        if segmentation or parse.thermal or len(parse.target_classes) > 3:
            score += 1
        if parse.task_type == TaskType.EXPORT:  # export-only rarely needs heavy review
            score -= 1
        return {0: ReviewLevel.LOW, 1: ReviewLevel.MEDIUM}.get(score, ReviewLevel.HIGH)
