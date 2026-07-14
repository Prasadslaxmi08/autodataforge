"""Video import view-model (Phase 17.5) — plain data for the Import Workspace.

No Qt. It reuses the EXISTING Planner Agent for pre-extraction estimates: it builds a
synthetic DatasetContext from the video metadata + the chosen extraction strategy and
calls `container.planner_agent.plan(...)` exactly as the normal flow would after import.
Estimates the planner does not provide (review rate / duplicate rate under the
deterministic fallback) come from Engineering-Memory recall or are marked
'Unavailable' — never fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass

from vds.agents.planner_agent import DatasetContext
from vds.container import Container
from vds.gui.planner_view import _short_detector
from vds.memory import DatasetFingerprint
from vds.video import ExtractionConfig, VideoInfo, engine

NA = "Unavailable"


@dataclass
class VideoPlan:
    estimated_dataset_size: int
    estimated_disk_mb: float
    expected_processing_time: str
    expected_review_rate: str
    expected_duplicate_pct: str
    recommended_detector: str
    recommended_batch_size: int
    recommended_tiling: str
    recommended_segmentation: str
    estimated_runtime: str
    expected_export_size: str
    source: str  # ai | deterministic
    note: str


def estimate_frames(info: VideoInfo, config: ExtractionConfig) -> int:
    """Upper-bound frame count for a strategy (before dedup). Scene-change is only
    known after extraction, so it is reported as the candidate count."""
    return len(engine.frame_indices(config, info.total_frames, info.fps))


def _resolution_summary(info: VideoInfo, count: int) -> dict:
    w, h = info.width or 0, info.height or 0
    return {"count": count, "distinct": 1, "min": [w, h], "max": [w, h],
            "common": [w, h], "megapixels_max": info.megapixels}


def _duplicate_estimate(container: Container, info: VideoInfo, count: int) -> str:
    fp = DatasetFingerprint(resolution_mp=info.megapixels or -1.0, dataset_size=count)
    matches = container.memory.recall(fp).matches
    ratios = [m.memory.dataset_fingerprint.duplicate_ratio for m in matches
              if m.memory.dataset_fingerprint.duplicate_ratio >= 0.0]
    if ratios:
        return f"~{sum(ratios) / len(ratios):.0%} (from similar past datasets)"
    return f"{NA} (measured during extraction)"


def build_video_plan(container: Container, info: VideoInfo, config: ExtractionConfig) -> VideoPlan:
    count = estimate_frames(info, config)
    settings = container.settings
    ctx = DatasetContext(
        project_id="video-preview",
        image_count=count,
        resolution_summary=_resolution_summary(info, count),
        file_types=["png"],
        classes=["object"],
        available_detectors=["builtin"],
        available_segmenters=["builtin"],
        gpu_device=settings.gpu.device,
        vram_budget_mb=settings.gpu.vram_budget_mb,
        export_format=settings.export.default_format,
        review_budget_hours=settings.runtime.review_budget_hours,
    )
    result = container.planner_agent.plan(ctx)
    pp = result.processing_plan
    ai = result.ai_plan  # populated only when a real LLM planned (else deterministic)

    review = f"{ai.expected_review_percent:.0f}%" if ai else NA
    tiling = ("Yes" if ai.tiling_required else "No") if ai else NA
    seg = "Yes" if pp.segmenter and pp.segmenter != "none" else "No"
    runtime = f"{pp.estimated_seconds:.1f} s"
    disk = engine.estimate_disk_mb(info, count)

    return VideoPlan(
        estimated_dataset_size=count,
        estimated_disk_mb=disk,
        expected_processing_time=runtime,
        expected_review_rate=review,
        expected_duplicate_pct=_duplicate_estimate(container, info, count),
        recommended_detector=_short_detector(pp.detector),
        recommended_batch_size=pp.batch_size,
        recommended_tiling=tiling,
        recommended_segmentation=seg,
        estimated_runtime=runtime,
        expected_export_size=f"~{disk} MB",
        source=result.source,
        note=result.memory_note or "Planner recommendations — you may override them.",
    )
