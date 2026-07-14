"""Pipeline monitoring view-model (Phase 13) — plain data for the Pipeline Workspace.

No Qt here. The existing `Phase1Pipeline.run()` is monolithic and exposes no live
per-stage hooks, so this module derives the eight-stage timeline, AI-model activity,
metrics, console events, and summary from the real ExecutionReport it returns —
every field maps to a measured number, nothing is invented. Live resource sampling
and elapsed time are handled on the UI side; the exact per-stage detail becomes
available when the pipeline reports it (at completion).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vds.agents.analyst_agent import AnalystContext, LLMAnalyst
from vds.agents.planner_agent import build_dataset_context
from vds.container import Container
from vds.core.contracts import ExecutionReport
from vds.memory import DatasetFingerprint
from vds.reporting import _recommendations

# The eight stages, in order, and which measured stage-timing key (if any) feeds each.
STAGES = [
    ("Dataset Import", "ingest"),
    ("Validation", "ingest"),
    ("Detection", "detection"),
    ("Segmentation", "segmentation"),
    ("Verification", "verification"),
    ("Quality Analysis", None),
    ("Engineering Memory Update", None),
    ("Export", "export"),
]


@dataclass
class StageStatus:
    name: str
    status: str  # Waiting | Running | Completed | Failed | Skipped
    duration_s: float | None
    items: int
    progress_pct: int


@dataclass
class ModelActivity:
    detection: dict
    segmentation: dict
    verification: dict
    quality: dict
    memory: dict


@dataclass
class PipelineMetrics:
    images_processed: int
    images_remaining: int
    images_per_second: float
    avg_latency_ms: float
    gpu_util: float | None
    gpu_mem_mb: float | None
    cpu_percent: float
    ram_mb: float
    elapsed_s: float
    throughput_ips: float
    export_count: int
    failed: int
    skipped: int
    duplicates_removed: int


@dataclass
class PreviewItem:
    name: str
    width: int
    height: int
    path: str
    boxes: list[tuple[float, float, float, float, str, float]] = field(default_factory=list)


@dataclass
class PipelineSummary:
    dataset: str
    execution_time_s: float
    total_images: int
    processed: int
    successful: int
    failed: int
    skipped: int
    duplicates: int
    avg_detection_ms: float
    avg_segmentation_ms: float
    avg_verification_ms: float
    avg_review_rate: float
    planner_strategy: str
    memory_influence: str
    analyst_summary: str
    export_stats: str


def _short(name: str) -> str:
    tail = name.split(":")[-1].split(".")[-1]
    return (tail[:-7] if tail.endswith("Adapter") else tail).lower() or name


# --- timeline --------------------------------------------------------------
def stage_timeline(report: ExecutionReport) -> list[StageStatus]:
    ss = report.benchmark.stage_seconds
    q = report.quality
    items = {
        "Dataset Import": report.imported,
        "Validation": report.imported,
        "Detection": report.detections,
        "Segmentation": q.masks,
        "Verification": report.detections,
        "Quality Analysis": q.images,
        "Engineering Memory Update": 0,
        "Export": report.export.annotations,
    }
    out: list[StageStatus] = []
    for name, key in STAGES:
        if name == "Engineering Memory Update":
            # Phase1Pipeline registers comparison KPIs but does not write a memory
            # record (that is the Analyst's job), so this stage is honestly skipped.
            out.append(StageStatus(name, "Skipped", None, 0, 0))
            continue
        if name == "Export" and not report.export.validated:
            out.append(StageStatus(name, "Failed", ss.get(key), items[name], 100))
            continue
        out.append(StageStatus(name, "Completed", ss.get(key) if key else None,
                               items[name], 100))
    return out


def running_timeline() -> list[StageStatus]:
    """Timeline shown while the (atomic) pipeline is executing: the pipeline is
    running; exact per-stage numbers arrive when it reports at completion."""
    return [StageStatus(name, "Running" if i == 0 else "Waiting", None, 0, 0)
            for i, (name, _key) in enumerate(STAGES)]


# --- model activity --------------------------------------------------------
def model_activity(container: Container, report: ExecutionReport) -> ModelActivity:
    b, q = report.benchmark, report.quality
    device = container.settings.gpu.device
    memory = _memory_influence_dict(container, report)
    return ModelActivity(
        detection={
            "model": _short(container.settings.models.detector), "device": device,
            "inference_ms": b.avg_inference_ms, "objects": report.detections,
            "avg_confidence": q.avg_confidence,
        },
        segmentation={
            "model": _short(container.settings.models.segmenter),
            "masks": q.masks, "avg_iou": "—", "inference_ms": b.avg_inference_ms,
        },
        verification={
            "model": "RuleBasedVerifier (deterministic)", "flagged": report.needs_review,
            "result": f"{report.verified_approved} approved / {report.rejected} rejected",
            "confidence": q.avg_confidence, "reason": "confidence + geometry checks",
        },
        quality={
            "recommendations": _recommendations(report),
            "warnings": _quality_warnings(report),
        },
        memory=memory,
    )


def _quality_warnings(report: ExecutionReport) -> list[str]:
    q = report.quality
    out = []
    if q.review_rate > 0.3:
        out.append(f"Human-review rate {q.review_rate:.0%}")
    if q.images_with_no_detection:
        out.append(f"{q.images_with_no_detection} image(s) with zero detections")
    if q.duplicate_detections:
        out.append(f"{q.duplicate_detections} duplicate detection(s)")
    return out or ["No quality red flags"]


def _memory_influence_dict(container: Container, report: ExecutionReport) -> dict:
    try:
        ctx = build_dataset_context(report.project_id, container.settings, container.images)
        fp = DatasetFingerprint(
            resolution_mp=ctx.resolution_summary.get("megapixels_max", -1.0),
            dataset_size=ctx.image_count,
        )
        guidance = container.memory.recall(fp)
    except Exception:
        return {"match": "unavailable", "similarity": None, "strategy": "—", "review_reduction": "—"}
    if not guidance.matches:
        return {"match": "No historical match", "similarity": None,
                "strategy": "—", "review_reduction": "—"}
    m = guidance.matches[0]
    return {
        "match": m.memory.project_id or m.memory.id,
        "similarity": round(m.score, 3),
        "strategy": f"detector={m.memory.planner_decisions.detector}, "
                    f"conf={m.memory.planner_decisions.confidence_threshold}",
        "review_reduction": f"prior review rate {m.memory.execution_metrics.review_rate:.0%}",
    }


# --- metrics ---------------------------------------------------------------
def metrics(report: ExecutionReport, *, elapsed_s: float, cpu: float, ram_mb: float) -> PipelineMetrics:
    b, q = report.benchmark, report.quality
    return PipelineMetrics(
        images_processed=report.imported, images_remaining=0,
        images_per_second=b.images_per_second, avg_latency_ms=b.avg_inference_ms,
        gpu_util=b.gpu_util_percent, gpu_mem_mb=b.peak_vram_mb,
        cpu_percent=cpu, ram_mb=ram_mb, elapsed_s=round(elapsed_s, 2),
        throughput_ips=b.batch_throughput_ips or b.images_per_second,
        export_count=report.export.annotations, failed=q.invalid_annotations,
        skipped=report.quarantined, duplicates_removed=report.duplicates_skipped,
    )


# --- console events (reconstructed from measured results) ------------------
def console_events(report: ExecutionReport) -> list[tuple[str, str]]:
    q = report.quality
    ev: list[tuple[str, str]] = [
        ("info", f"Dataset loaded: {report.imported} images"),
    ]
    if report.duplicates_skipped:
        ev.append(("info", f"Duplicate images removed: {report.duplicates_skipped}"))
    if report.quarantined:
        ev.append(("warning", f"Validation quarantined: {report.quarantined} image(s)"))
    ev += [
        ("info", "Detection completed: "
                 f"{report.detections} objects, avg confidence {q.avg_confidence}"),
        ("info", f"Segmentation completed: {q.masks} masks"),
        ("info", f"Verification: {report.verified_approved} approved, "
                 f"{report.needs_review} flagged, {report.rejected} rejected"),
    ]
    if report.needs_review:
        ev.append(("warning", f"Verification flagged {report.needs_review} for human review"))
    ev.append(("info", "Quality analysis completed"))
    ev.append(("info", "Engineering memory update: skipped (no analyst in this run)"))
    status = "info" if report.export.validated else "error"
    ev.append((status, f"Export completed: {report.export.format}, "
                       f"{report.export.annotations} annotations, "
                       f"validated={report.export.validated}"))
    return ev


# --- preview (post-run annotated overlays) ---------------------------------
def preview_items(container: Container, project_id: str, limit: int = 12) -> list[PreviewItem]:
    from vds.core.contracts import Box2D

    items: list[PreviewItem] = []
    for img in container.images.by_project(project_id)[:limit]:
        path = container.cas.path(img.sha256)
        boxes = []
        for a in container.annotations.by_image(img.id):
            g = a.geometry
            if isinstance(g, Box2D):
                boxes.append((g.x, g.y, g.w, g.h, a.label, round(a.confidence, 2)))
        items.append(PreviewItem(name=img.id[:12], width=img.width, height=img.height,
                                 path=str(path), boxes=boxes))
    return items


# --- summary ---------------------------------------------------------------
def pipeline_summary(container: Container, report: ExecutionReport) -> PipelineSummary:
    b, q = report.benchmark, report.quality
    imgs = max(1, report.imported)
    ss = b.stage_seconds
    proj = container.projects.get(report.project_id)
    mem = _memory_influence_dict(container, report)
    mem_txt = (f"{mem['match']} (similarity {mem['similarity']})"
               if mem.get("similarity") else "No historical match")
    return PipelineSummary(
        dataset=proj.name if proj else report.project_id,
        execution_time_s=b.total_seconds, total_images=report.imported + report.duplicates_skipped,
        processed=report.imported, successful=report.verified_approved,
        failed=q.invalid_annotations, skipped=report.quarantined,
        duplicates=report.duplicates_skipped,
        avg_detection_ms=round(1000 * ss.get("detection", 0.0) / imgs, 2),
        avg_segmentation_ms=round(1000 * ss.get("segmentation", 0.0) / imgs, 2),
        avg_verification_ms=round(1000 * ss.get("verification", 0.0) / imgs, 2),
        avg_review_rate=q.review_rate,
        planner_strategy=f"detector={_short(container.settings.models.detector)}, "
                         "deterministic ExecutionPlanner",
        memory_influence=mem_txt,
        analyst_summary=_analyst_summary(container, report),
        export_stats=f"{report.export.format}: {report.export.images} images, "
                     f"{report.export.annotations} annotations, validated={report.export.validated}",
    )


def _analyst_summary(container: Container, report: ExecutionReport) -> str:
    """Real Analyst executive summary via the existing agent (deterministic under
    the default provider). Never fails the caller."""
    try:
        result = LLMAnalyst(container.llm_client).analyze(AnalystContext(execution=report))
        return result.report.executive_summary
    except Exception as exc:  # analyst must never break the summary
        return f"Analyst unavailable ({type(exc).__name__})"
