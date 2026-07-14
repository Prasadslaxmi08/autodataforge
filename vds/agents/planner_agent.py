"""Planner Agent (Phase 7) — the platform's first true AI agent.

It analyses an imported dataset and produces an *execution* plan (which models,
thresholds, tiling, batching, review expectations, order) with a confidence and
justification for every decision. It reasons via the provider-agnostic
LLMClient — never a hardcoded provider — and its output is schema-validated
before use.

Safety contract (phase brief): the Planner never stops the pipeline. On invalid
output, provider failure, or missing credentials it falls back to the
deterministic ExecutionPlanner and logs why. With the default Echo provider
(no real LLM) it therefore always falls back — the correct safe default.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

from vds.agents.base import Agent
from vds.agents.cost import estimate_cost
from vds.agents.llm import LLMClient
from vds.agents.planner import ExecutionPlanner
from vds.config.settings import Settings
from vds.core.contracts import ProcessingPlan, ProjectId
from vds.logging import get_logger
from vds.memory import DatasetFingerprint, EngineeringMemoryService, MemoryGuidance
from vds.store.sqlite import ImageRepo

log = get_logger(__name__)

_VALID_FORMATS = {"coco", "yolo", "voc"}

# Candidate models the Planner may reason about. `available` filters this to what
# is actually installed; the Planner must only *choose* from the available set.
MODEL_CATALOG: dict[str, dict] = {
    "builtin": {"det": True, "seg": True, "vram_mb": 0, "speed": "fast",
                "accuracy": "low", "open_vocab": False},
    "yolo": {"det": True, "seg": False, "vram_mb": 2048, "speed": "fast",
             "accuracy": "medium", "open_vocab": False},
    "grounding_dino": {"det": True, "seg": False, "vram_mb": 4096, "speed": "medium",
                       "accuracy": "high", "open_vocab": True},
    "sam2": {"det": False, "seg": True, "vram_mb": 3072, "speed": "medium",
             "accuracy": "high", "open_vocab": False},
}


# --- schemas ---------------------------------------------------------------
class DatasetContext(BaseModel):
    """Everything the Planner reasons over. Built from the imported dataset."""

    project_id: ProjectId
    image_count: int
    resolution_summary: dict
    file_types: list[str]
    classes: list[str]
    available_detectors: list[str]
    available_segmenters: list[str]
    gpu_device: str
    vram_budget_mb: int
    export_format: str
    review_budget_hours: float
    model_catalog: dict = Field(default_factory=lambda: MODEL_CATALOG)
    user_preferences: dict = Field(default_factory=dict)


class DecisionRationale(BaseModel):
    decision: str
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str


class PlannerPlan(BaseModel):
    """The AI-generated, schema-validated execution plan. Constrained fields make
    invalid LLM output fail validation → retry → fallback."""

    detector: str
    segmenter: str
    run_segmentation: bool
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    tiling_required: bool
    batch_size: int = Field(gt=0, le=1024)
    worker_count: int = Field(gt=0, le=64)
    expected_processing_seconds: float = Field(ge=0.0)
    expected_gpu_mb: int = Field(ge=0)
    expected_review_percent: float = Field(ge=0.0, le=100.0)
    expected_annotation_density: float = Field(ge=0.0)
    export_format: str
    execution_order: list[str]
    rationale: list[DecisionRationale] = Field(default_factory=list)
    summary: str = ""


class PlannerResult(BaseModel):
    """What the Planner returns: the plan the pipeline runs, plus provenance and
    metrics (latency / retries / tokens / cost / fallback)."""

    processing_plan: ProcessingPlan
    source: Literal["ai", "deterministic"]
    ai_plan: PlannerPlan | None = None
    fallback_reason: str | None = None
    latency_ms: float = 0.0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float | None = None
    # Engineering-memory influence (Phase 10). memory_note always explains whether
    # prior experience informed the plan; memory_matches lists the memory ids used.
    memory_used: bool = False
    memory_note: str = ""
    memory_matches: list[str] = Field(default_factory=list)


PLANNER_SYSTEM_PROMPT = """\
You are a Senior Computer Vision Engineer planning an automated annotation run.
Given a dataset profile and the available models, choose the execution parameters
that best balance: accuracy, processing speed, GPU utilization, human-review
reduction, and annotation quality.

Rules:
- Choose `detector` and `segmenter` ONLY from the context's available lists.
- Respect the GPU VRAM budget; do not plan models that exceed it.
- Enable `tiling_required` for high-resolution images with small objects.
- Larger, denser datasets warrant larger batches and more workers, within limits.
- Estimate `expected_review_percent` honestly: low-accuracy models and dense or
  small-object scenes need more human review.
- Output ONLY a JSON object matching the required schema. Be concise. Do not add
  prose outside the JSON. Provide a short confidence (0-1) and one-sentence
  justification for each major decision in `rationale`.
"""


def _resolution_summary(res: list[tuple[int, int]]) -> dict:
    if not res:
        return {"count": 0}
    widths = [w for w, _ in res]
    heights = [h for _, h in res]
    from collections import Counter

    common = Counter(res).most_common(1)[0][0]
    return {
        "count": len(res),
        "distinct": len(set(res)),
        "min": [min(widths), min(heights)],
        "max": [max(widths), max(heights)],
        "common": list(common),
        "megapixels_max": round(max(w * h for w, h in res) / 1e6, 3),
    }


def build_dataset_context(
    project_id: ProjectId,
    settings: Settings,
    images: ImageRepo,
    *,
    classes: list[str] | None = None,
    user_preferences: dict | None = None,
) -> DatasetContext:
    recs = images.by_project(project_id)
    return DatasetContext(
        project_id=project_id,
        image_count=len(recs),
        resolution_summary=_resolution_summary([(r.width, r.height) for r in recs]),
        file_types=["png"],  # ingest normalizes to PNG (Phase 1)
        classes=classes or ["object"],
        available_detectors=["builtin"],  # grows as adapters are installed
        available_segmenters=["builtin"],
        gpu_device=settings.gpu.device,
        vram_budget_mb=settings.gpu.vram_budget_mb,
        export_format=settings.export.default_format,
        review_budget_hours=settings.runtime.review_budget_hours,
        user_preferences=user_preferences or {},
    )


class LLMPlanner(Agent):
    system_prompt = PLANNER_SYSTEM_PROMPT

    def __init__(
        self,
        client: LLMClient,
        deterministic: ExecutionPlanner,
        memory: EngineeringMemoryService | None = None,
    ) -> None:
        super().__init__(client)
        self._fallback_planner = deterministic
        self._memory = memory

    def _recall(self, context: DatasetContext) -> MemoryGuidance:
        """Ask engineering memory: have we processed a similar dataset? Uses only
        pre-run features the Planner actually knows (resolution, scale, scene type)."""
        if self._memory is None:
            return MemoryGuidance(note="Engineering memory is not enabled.")
        query = DatasetFingerprint(
            resolution_mp=context.resolution_summary.get("megapixels_max", -1.0),
            dataset_size=context.image_count,
            scene_type=context.user_preferences.get("scene_type", "unknown"),
        )
        return self._memory.recall(query)

    def plan(self, context: DatasetContext) -> PlannerResult:
        import time

        start = time.perf_counter()
        guidance = self._recall(context)
        convo = self.new_conversation().user(
            "Plan the annotation run for this dataset. Return one PlannerPlan JSON "
            "object. If prior engineering experience is provided, let it inform your "
            "choices and note it in `summary`.\n\nDATASET:\n"
            + context.model_dump_json(indent=2)
            + "\n\nENGINEERING MEMORY:\n" + guidance.render()
        )
        try:
            outcome = self._client.structured(convo, PlannerPlan)
        except Exception as exc:  # never propagate an LLM failure (phase brief)
            return self._fallback(context, f"{type(exc).__name__}: {exc}", start, guidance=guidance)

        invalid = self._invalid_choice(outcome.value, context)
        if invalid:
            return self._fallback(context, f"invalid plan: {invalid}", start, outcome.attempts, guidance)

        latency = round((time.perf_counter() - start) * 1000, 3)
        usage = outcome.response.usage
        log.info("planner.ai_plan", project_id=context.project_id, latency_ms=latency,
                 memory_used=guidance.has_experience)
        return PlannerResult(
            processing_plan=self._to_processing_plan(outcome.value, context),
            source="ai",
            ai_plan=outcome.value,
            latency_ms=latency,
            retries=outcome.attempts - 1,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            estimated_cost_usd=estimate_cost(outcome.response.model, usage),
            memory_used=guidance.has_experience,
            memory_note=guidance.note,
            memory_matches=[m.memory.id for m in guidance.matches],
        )

    def _invalid_choice(self, plan: PlannerPlan, ctx: DatasetContext) -> str | None:
        if plan.detector not in ctx.available_detectors:
            return f"detector {plan.detector!r} not available"
        if plan.run_segmentation and plan.segmenter not in ctx.available_segmenters:
            return f"segmenter {plan.segmenter!r} not available"
        if plan.export_format not in _VALID_FORMATS:
            return f"export_format {plan.export_format!r} unsupported"
        if plan.expected_gpu_mb > ctx.vram_budget_mb:
            return "plan exceeds the VRAM budget"
        return None

    def _to_processing_plan(self, plan: PlannerPlan, ctx: DatasetContext) -> ProcessingPlan:
        batch = plan.batch_size
        return ProcessingPlan(
            project_id=ctx.project_id,
            version=1,
            image_count=ctx.image_count,
            classes=ctx.classes,
            detector=plan.detector,
            segmenter=plan.segmenter,
            confidence_threshold=plan.confidence_threshold,
            batch_size=batch,
            num_batches=math.ceil(ctx.image_count / batch) if ctx.image_count else 0,
            gpu_budget_mb=ctx.vram_budget_mb,
            export_format=plan.export_format,
            estimated_seconds=plan.expected_processing_seconds,
        )

    def _fallback(
        self, ctx: DatasetContext, reason: str, start: float, retries: int = 0,
        guidance: MemoryGuidance | None = None,
    ) -> PlannerResult:
        import time

        log.warning("planner.fallback", project_id=ctx.project_id, reason=reason)
        plan = self._fallback_planner.plan(ctx.project_id, ctx.image_count, ctx.classes)
        g = guidance or MemoryGuidance(note="Engineering memory not consulted.")
        return PlannerResult(
            processing_plan=plan,
            source="deterministic",
            fallback_reason=reason,
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
            retries=retries,
            memory_used=g.has_experience,
            memory_note=g.note,
            memory_matches=[m.memory.id for m in g.matches],
        )
