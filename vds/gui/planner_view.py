"""Planner view-model (Phase 12) — plain data for the Planner Workspace.

No Qt here. This assembles everything the four workspace panels display, sourced
ENTIRELY from the existing Planner Agent, Engineering Memory, and quality metrics —
nothing is reimplemented. The engineer's manual overrides are their own inputs:
they are injected into the Planner's `user_preferences` (so a real AI provider sees
them) and reflected in the effective controls; the plan itself still comes from
`planner_agent.plan()` exactly as implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vds.agents.planner_agent import MODEL_CATALOG, build_dataset_context
from vds.container import Container
from vds.memory import DatasetFingerprint

DETECTOR_OPTIONS = list(MODEL_CATALOG.keys())
EXPORT_OPTIONS = ["coco", "yolo", "voc"]


def _short_detector(name: str) -> str:
    """Friendly detector label: 'vds...:BuiltinAdapter' -> 'builtin'; 'yolo' -> 'yolo'."""
    tail = name.split(":")[-1].split(".")[-1]
    label = tail[:-7] if tail.endswith("Adapter") else tail
    return label.lower() or name


@dataclass
class PlanControls:
    """The six engineer-editable settings. `None` means 'leave as the Planner chose'."""

    detector: str | None = None
    segmentation: bool | None = None
    confidence_threshold: float | None = None
    batch_size: int | None = None
    worker_count: int | None = None
    export_format: str | None = None


@dataclass
class Decision:
    name: str
    value: str
    reason: str
    confidence: str  # "0.85" or "—"
    expected_impact: str
    trade_offs: str
    validation: str


@dataclass
class MemoryMatchView:
    dataset: str
    similarity: float
    strategy: str
    review_rate: float
    runtime: str
    benchmark: str
    analyst_recommendation: str
    why: str


@dataclass
class PlanEvaluation:
    runtime_s: float
    throughput_ips: float
    review_rate: float
    quality: float
    gpu_util_pct: float | None
    memory_mb: float | None
    cost_usd: float | None
    confidence: float | None


@dataclass
class DatasetProfile:
    name: str
    phase: str
    image_count: int
    storage_mb: float
    avg_resolution: str
    avg_megapixels: float
    class_distribution: dict[str, int]
    small_object_pct: float | None
    duplicate_pct: float | None
    difficulty: str
    fingerprint: str
    import_date: str
    version: int


@dataclass
class PlanView:
    project_id: str
    profile: DatasetProfile
    decisions: list[Decision]
    effective_controls: PlanControls
    source: str  # "ai" | "deterministic"
    memory_used: bool
    memory_note: str
    memory_matches: list[MemoryMatchView]
    evaluation: PlanEvaluation
    warnings: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)


@dataclass
class PlanDiffRow:
    field: str
    original: str
    modified: str
    delta: str


# --- dataset profile -------------------------------------------------------
def _profile(container: Container, project_id: str, fingerprint: str) -> DatasetProfile:
    proj = container.projects.get(project_id)
    images = container.images.by_project(project_id)
    n = len(images)
    storage = 0
    for img in images:
        p = container.cas.path(img.sha256)
        if p.exists():
            storage += p.stat().st_size
    avg_w = round(sum(i.width for i in images) / n) if n else 0
    avg_h = round(sum(i.height for i in images) / n) if n else 0
    avg_mp = round((avg_w * avg_h) / 1e6, 3)

    quality = container.analyzer.quality(project_id)
    errors = container.analyzer.errors(project_id)
    dets = quality.detections
    small = next((c.count for c in errors.categories if c.name == "small_objects"), 0)
    small_pct = round(100 * small / dets, 1) if dets else None
    dup_pct = round(100 * quality.duplicate_detections / dets, 1) if dets else None

    fam = [m for m in container.memory.all() if m.dataset_fingerprint.hash() == fingerprint]
    fam.sort(key=lambda m: (m.created_at, m.version))
    import_date = fam[-1].created_at if fam else "—"
    version = fam[-1].version if fam else 1

    return DatasetProfile(
        name=proj.name if proj else project_id,
        phase=proj.phase if proj else "—",
        image_count=n,
        storage_mb=round(storage / (1024 * 1024), 2),
        avg_resolution=f"{avg_w}×{avg_h}",
        avg_megapixels=avg_mp,
        class_distribution={"object": dets},
        small_object_pct=small_pct,
        duplicate_pct=dup_pct,
        difficulty=_difficulty(small_pct, quality.annotation_density),
        fingerprint=fingerprint,
        import_date=import_date,
        version=version,
    )


def _difficulty(small_pct: float | None, density: float) -> str:
    if small_pct is None:
        return "Unknown (not yet processed)"
    score = (small_pct / 100) + min(density / 10, 1.0)
    if score >= 1.0:
        return "High"
    if score >= 0.5:
        return "Medium"
    return "Low"


# --- decisions -------------------------------------------------------------
def _rationale_for(ai, *keywords: str) -> tuple[str, str]:
    """(reason, confidence) from the Planner's own rationale, matched by keyword."""
    if ai is None:
        return ("", "—")
    for r in ai.rationale:
        text = r.decision.lower()
        if any(k in text for k in keywords):
            return (r.justification, f"{r.confidence:.2f}")
    return ("", "—")


def _decisions(result, controls: PlanControls, image_count: int) -> list[Decision]:
    plan = result.processing_plan
    ai = result.ai_plan
    validated = "Validated by Planner" if result.source == "ai" else "Deterministic fallback"

    def dec(name, value, kws, factual, impact, trade="—"):
        reason, conf = _rationale_for(ai, *kws)
        return Decision(name=name, value=str(value), reason=reason or factual,
                        confidence=conf, expected_impact=impact, trade_offs=trade,
                        validation=validated)

    review_pct = round(ai.expected_review_percent, 1) if ai else round(100 * _measured_review(result), 1)
    runtime = ai.expected_processing_seconds if ai else plan.estimated_seconds
    gpu = f"{ai.expected_gpu_mb} MB" if ai else "—"
    return [
        dec("Detector", controls.detector, ["detector", "model"],
            "The configured detector for this dataset.",
            "Drives detection accuracy and speed."),
        dec("Segmentation", "Enabled" if controls.segmentation else "Disabled",
            ["segment", "mask"], "Instance masks per detected object.",
            "Adds mask quality at extra compute."),
        dec("Confidence Threshold", controls.confidence_threshold, ["confidence", "threshold"],
            "Detections below this score are dropped.",
            "Higher → fewer false positives, more misses."),
        dec("Batch Size", controls.batch_size, ["batch"],
            "Images processed per model call.",
            "Larger → faster, more GPU memory."),
        dec("Worker Count", controls.worker_count, ["worker", "parallel"],
            "Parallel processing workers.", "More → higher throughput within limits."),
        dec("GPU Strategy", f"{plan.gpu_budget_mb} MB budget", ["gpu", "vram"],
            "VRAM budget the plan must respect.", "Bounds model choice and batch size."),
        dec("Tiling", "Enabled" if (ai and ai.tiling_required) else "Disabled",
            ["tile", "tiling"], "Split large images to recover small objects.",
            "Higher recall on small objects, slower."),
        dec("Export Format", controls.export_format, ["export", "format"],
            "Annotation export format.", "Downstream training compatibility."),
        dec("Estimated Runtime", f"{runtime}s", ["runtime", "time"],
            "Predicted total processing time.", "Planning/throughput reference."),
        dec("Estimated GPU Usage", gpu, ["gpu"],
            "Predicted peak GPU memory.", "Must fit the VRAM budget."),
        dec("Estimated Human Review", f"{review_pct}%", ["review"],
            "Predicted share of annotations needing a human.", "Drives labeling cost."),
    ]


def _measured_review(result) -> float:
    return 0.0  # deterministic plan carries no review estimate; shown as 0% baseline


# --- evaluation ------------------------------------------------------------
def _evaluation(result, image_count: int, quality) -> PlanEvaluation:
    plan, ai = result.processing_plan, result.ai_plan
    runtime = ai.expected_processing_seconds if ai else plan.estimated_seconds
    throughput = round(image_count / runtime, 3) if runtime > 0 else 0.0
    review = round(ai.expected_review_percent / 100, 4) if ai else quality.review_rate
    gpu_pct = (round(100 * ai.expected_gpu_mb / plan.gpu_budget_mb, 1)
               if ai and plan.gpu_budget_mb else None)
    conf = None
    if ai and ai.rationale:
        conf = round(sum(r.confidence for r in ai.rationale) / len(ai.rationale), 3)
    return PlanEvaluation(
        runtime_s=runtime, throughput_ips=throughput, review_rate=review,
        quality=round(max(0.0, quality.approval_rate - quality.rejection_rate), 4),
        gpu_util_pct=gpu_pct, memory_mb=None, cost_usd=result.estimated_cost_usd,
        confidence=conf,
    )


# --- memory ----------------------------------------------------------------
def _memory_matches(container: Container, fp: DatasetFingerprint) -> list[MemoryMatchView]:
    guidance = container.memory.recall(fp)
    out: list[MemoryMatchView] = []
    for m in guidance.matches:
        mem = m.memory
        d = mem.planner_decisions
        rec = (mem.engineering_recommendations[0].action
               if mem.engineering_recommendations else "—")
        out.append(MemoryMatchView(
            dataset=mem.project_id or mem.id,
            similarity=m.score,
            strategy=f"detector={d.detector}, conf={d.confidence_threshold}, tiling={d.tiling}",
            review_rate=mem.execution_metrics.review_rate,
            runtime=f"{mem.execution_metrics.runtime_seconds}s",
            benchmark=f"quality {mem.benchmark_summary.quality_score}, "
                      f"throughput {mem.benchmark_summary.throughput_ips} img/s",
            analyst_recommendation=rec,
            why=m.explain(),
        ))
    return out


def _warnings_risks(profile: DatasetProfile, result, controls: PlanControls,
                    ev: PlanEvaluation) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    risks: list[str] = []
    if profile.small_object_pct and profile.small_object_pct > 50 and not _tiling(result):
        warnings.append(
            f"{profile.small_object_pct:.0f}% of objects are small but tiling is disabled — "
            "small-object recall may suffer.")
    if profile.duplicate_pct and profile.duplicate_pct > 20:
        warnings.append(f"High duplicate rate ({profile.duplicate_pct:.0f}%) — consider deduplication.")
    if ev.review_rate > 0.4:
        warnings.append(f"High predicted human-review rate ({ev.review_rate:.0%}).")
    if result.source == "deterministic":
        risks.append("No AI provider configured — this plan is the deterministic fallback, "
                     "so per-decision rationale and estimates are limited.")
    if ev.confidence is not None and ev.confidence < 0.5:
        risks.append(f"Low Planner confidence ({ev.confidence}).")
    return warnings, risks


def _tiling(result) -> bool:
    return bool(result.ai_plan and result.ai_plan.tiling_required)


# --- assembly --------------------------------------------------------------
def build_plan_view(
    container: Container, project_id: str, overrides: PlanControls | None = None
) -> PlanView:
    """Run the Planner Agent for a dataset (optionally with engineer overrides) and
    assemble the four-panel view. The Agent is used exactly as implemented."""
    prefs: dict = {}
    if overrides is not None:
        prefs = {k: v for k, v in overrides.__dict__.items() if v is not None}

    context = build_dataset_context(
        project_id, container.settings, container.images, user_preferences=prefs
    )
    result = container.planner_agent.plan(context)

    fp = DatasetFingerprint(
        resolution_mp=context.resolution_summary.get("megapixels_max", -1.0),
        dataset_size=context.image_count,
        scene_type=prefs.get("scene_type", "unknown"),
    )
    profile = _profile(container, project_id, fp.hash())

    plan, ai = result.processing_plan, result.ai_plan
    base = PlanControls(
        detector=_short_detector(plan.detector),
        segmentation=(ai.run_segmentation if ai else True),
        confidence_threshold=plan.confidence_threshold,
        batch_size=plan.batch_size,
        worker_count=(ai.worker_count if ai else 1),
        export_format=plan.export_format,
    )
    effective = _apply_overrides(base, overrides)

    quality = container.analyzer.quality(project_id)
    ev = _evaluation(result, context.image_count, quality)
    warnings, risks = _warnings_risks(profile, result, effective, ev)
    return PlanView(
        project_id=project_id, profile=profile,
        decisions=_decisions(result, effective, context.image_count),
        effective_controls=effective, source=result.source,
        memory_used=result.memory_used, memory_note=result.memory_note,
        memory_matches=_memory_matches(container, fp),
        evaluation=ev, warnings=warnings, risks=risks,
    )


def _apply_overrides(base: PlanControls, overrides: PlanControls | None) -> PlanControls:
    if overrides is None:
        return base
    return PlanControls(
        detector=overrides.detector or base.detector,
        segmentation=base.segmentation if overrides.segmentation is None else overrides.segmentation,
        confidence_threshold=(base.confidence_threshold if overrides.confidence_threshold is None
                              else overrides.confidence_threshold),
        batch_size=overrides.batch_size or base.batch_size,
        worker_count=overrides.worker_count or base.worker_count,
        export_format=overrides.export_format or base.export_format,
    )


def diff_plans(original: PlanView, modified: PlanView) -> list[PlanDiffRow]:
    """What changed between the original AI plan and the modified plan."""
    rows: list[PlanDiffRow] = []

    def row(fieldname: str, a, b, numeric: bool = False):
        if str(a) == str(b):
            return
        delta = ""
        if numeric and isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = f"{b - a:+.4g}"
        rows.append(PlanDiffRow(field=fieldname, original=str(a), modified=str(b), delta=delta))

    oc, mc = original.effective_controls, modified.effective_controls
    row("Detector", oc.detector, mc.detector)
    row("Segmentation", oc.segmentation, mc.segmentation)
    row("Confidence Threshold", oc.confidence_threshold, mc.confidence_threshold, True)
    row("Batch Size", oc.batch_size, mc.batch_size, True)
    row("Worker Count", oc.worker_count, mc.worker_count, True)
    row("Export Format", oc.export_format, mc.export_format)

    oe, me = original.evaluation, modified.evaluation
    row("Runtime (s)", oe.runtime_s, me.runtime_s, True)
    row("Review Rate", oe.review_rate, me.review_rate, True)
    row("GPU Util %", oe.gpu_util_pct, me.gpu_util_pct, True)
    row("Quality", oe.quality, me.quality, True)
    row("Confidence", oe.confidence, me.confidence, True)
    return rows
