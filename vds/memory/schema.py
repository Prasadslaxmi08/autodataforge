"""Engineering Memory schema (Phase 10).

Structured, versioned, explainable records of *engineering knowledge* — never raw
images, never hallucinated data. Every field here is populated from measured
pipeline outputs or from Analyst recommendations that already passed evidence
validation (see `builder.py`). No embeddings, no free-form blobs: each record is a
typed row that can be diffed, queried, and explained deterministically.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field

Validation = Literal["validated", "provisional", "rejected"]


def _round(x: float, n: int = 4) -> float:
    return round(float(x), n)


class DatasetFingerprint(BaseModel):
    """The measurable identity of a dataset. Pre-run fields (resolution, size,
    classes, scene_type) are known before execution and drive Planner recall;
    post-run fields (density, ratios, quality) are filled from results. A field
    left at its `None`/`-1` sentinel is treated as "unknown" and skipped by the
    similarity engine, so a pre-run query matches on the subset it knows."""

    # pre-run (available to the Planner before a job)
    resolution_mp: float = -1.0  # largest image megapixels
    dataset_size: int = -1  # image count
    class_distribution: dict[str, int] = Field(default_factory=dict)
    scene_type: str = "unknown"  # categorical tag, e.g. "aerial", "street"
    environment: dict[str, str] = Field(default_factory=dict)  # source, sensor, etc.
    # post-run (filled from measured outputs)
    scene_density: float = -1.0  # mean objects per image (crowdedness)
    object_density: float = -1.0  # detections per megapixel
    duplicate_ratio: float = -1.0
    small_object_ratio: float = -1.0
    image_quality: float = -1.0  # 0..1 proxy (1 - quarantine rate)
    avg_confidence: float = -1.0

    def hash(self) -> str:
        """Stable fingerprint-family id: same dataset shape -> same hash. Rounds
        floats so trivial numeric noise doesn't fork the family."""
        payload = {
            "resolution_mp": _round(self.resolution_mp, 1),
            "dataset_size": self.dataset_size,
            "classes": sorted(self.class_distribution.items()),
            "scene_type": self.scene_type,
            "scene_density": _round(self.scene_density, 1),
            "small_object_ratio": _round(self.small_object_ratio, 2),
        }
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class PlannerDecisions(BaseModel):
    detector: str
    segmentation_enabled: bool
    confidence_threshold: float
    batch_size: int
    worker_count: int = 1
    tiling: bool = False
    export_strategy: str = "coco"


class ExecutionMetrics(BaseModel):
    throughput_ips: float
    runtime_seconds: float
    gpu_util_percent: float | None = None
    cpu_percent: float = 0.0
    peak_ram_mb: float = 0.0
    review_rate: float = 0.0
    approval_rate: float = 0.0
    rejection_rate: float = 0.0
    invalid_annotations: int = 0
    empty_masks: int = 0
    export_format: str = ""
    export_validated: bool = False


class AnalystConclusions(BaseModel):
    root_causes: list[str] = Field(default_factory=list)
    bottlenecks: list[str] = Field(default_factory=list)
    improvement_opportunities: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class MemoryRecommendation(BaseModel):
    """A reusable engineering recommendation — only stored when it was evidence-backed
    (validated) by the Analyst. `supporting_metrics` are the evidence keys it cited."""

    action: str
    target: str
    reason: str
    expected_impact: str
    confidence: float
    supporting_metrics: list[str] = Field(default_factory=list)


class VerificationOutcomes(BaseModel):
    common_semantic_failures: dict[str, int] = Field(default_factory=dict)
    frequently_corrected_labels: dict[str, int] = Field(default_factory=dict)
    bbox_issues: int = 0
    segmentation_issues: int = 0
    false_positives: int = 0  # measured proxy: auto-rejected annotations
    false_negatives: int = 0  # measured proxy: images with zero detections


class BenchmarkSummary(BaseModel):
    throughput_ips: float
    review_rate: float
    approval_rate: float
    avg_confidence: float
    quality_score: float  # composite 0..1 (approval - rejection, clamped)


class EngineeringMemory(BaseModel):
    """One complete engineering-knowledge record for a single pipeline execution."""

    id: str
    created_at: str  # ISO-8601; stamped by the caller (keeps the module deterministic)
    project_id: str
    source: str  # who produced it, e.g. "analyst.ai" / "pipeline.deterministic"
    version: int = 1

    dataset_fingerprint: DatasetFingerprint
    planner_decisions: PlannerDecisions
    execution_metrics: ExecutionMetrics
    analyst_conclusions: AnalystConclusions
    verification_outcomes: VerificationOutcomes
    benchmark_summary: BenchmarkSummary
    engineering_recommendations: list[MemoryRecommendation] = Field(default_factory=list)

    validation_status: Validation = "provisional"
    confidence: float = 0.0

    def content_hash(self) -> str:
        """Identifies an *identical* record (for duplicate suppression). Excludes
        id/created_at/version so re-recording the same run is caught as a dup."""
        blob = self.model_dump(exclude={"id", "created_at", "version"})
        return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]
