"""Domain contracts — the stable, versioned vocabulary shared across modules
(System Design §2.1).

Every inter-module boundary object is defined here as a Pydantic model. Modules
depend on these types, never on each other's internals. Geometry is a tagged
union extensible to 3D (amendment 6) without touching existing modules.

Bootstrap scope: the shapes are defined; behaviour lives in the services that
produce and consume them (Phase 1+).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# --- identifiers ------------------------------------------------------------
# Opaque string ids (UUID/hash strings). Kept as aliases so signatures read
# intentionally without a NewType ceremony that pydantic would ignore anyway.
ProjectId = str
ImageId = str
AnnotationId = str
JobId = str
SnapshotId = str
Sha256 = str


# --- provenance -------------------------------------------------------------
class Provenance(BaseModel):
    """Who/what produced a fact, and why. Attached to every annotation (FR-3)."""

    source: str  # agent or engine name, e.g. "engine.labeling", "agent.verifier"
    model: str | None = None  # adapter/model id, e.g. "grounding_dino"
    model_version: str | None = None
    prompt: str | None = None
    plan_version: int | None = None


# --- geometry (tagged union) ------------------------------------------------
class Box2D(BaseModel):
    kind: Literal["box2d"] = "box2d"
    x: float
    y: float
    w: float
    h: float


class Mask(BaseModel):
    kind: Literal["mask"] = "mask"
    rle: str  # run-length-encoded mask; stored in CAS for large masks
    height: int
    width: int


class Polygon(BaseModel):
    kind: Literal["polygon"] = "polygon"
    points: list[tuple[float, float]]


class ImageLabel(BaseModel):
    kind: Literal["image_label"] = "image_label"  # whole-image classification


Geometry = Annotated[
    Box2D | Mask | Polygon | ImageLabel,
    Field(discriminator="kind"),
]


class Detection(BaseModel):
    """A detector's structured output: a box, a label, and a confidence."""

    box: Box2D
    label: str
    confidence: float


# --- core records -----------------------------------------------------------
class Project(BaseModel):
    id: ProjectId
    name: str
    brief: str
    phase: str  # ProjectPhase value; stored as str to keep core import-light


class ImageRecord(BaseModel):
    id: ImageId
    project_id: ProjectId
    sha256: Sha256
    width: int
    height: int
    state: str  # ImageState value
    quarantine_reason: str | None = None


class Annotation(BaseModel):
    id: AnnotationId
    image_id: ImageId
    label: str
    geometry: Geometry
    confidence: float
    state: str  # AnnotationState value
    provenance: Provenance
    # ponytail: mask stored inline for MVP (test masks are tiny). Move to CAS by
    # sha reference when production masks get large.
    mask: Mask | None = None


# --- agent I/O contracts ----------------------------------------------------
class ClassSpec(BaseModel):
    name: str
    definition: str
    synonyms: list[str] = Field(default_factory=list)
    edge_cases: list[str] = Field(default_factory=list)
    strategy: dict[str, Any] = Field(default_factory=dict)  # prompts/thresholds/tiling


class LabelingPlan(BaseModel):
    """The approved contract every downstream engine reads. Versioned; the
    Planner revises it in response to feedback."""

    project_id: ProjectId
    version: int
    classes: list[ClassSpec]
    open_questions: list[str] = Field(default_factory=list)
    approved: bool = False


VerdictLabel = Literal["correct", "wrong_class", "bad_geometry", "hallucination"]


class Verdict(BaseModel):
    annotation_id: AnnotationId
    verdict: VerdictLabel
    confidence: float
    rationale: str


class TriageScore(BaseModel):
    annotation_id: AnnotationId
    score: float  # higher = more worth human review
    reasons: list[str] = Field(default_factory=list)


class FeedbackSummary(BaseModel):
    project_id: ProjectId
    findings: list[str]
    recommended_plan_changes: list[str] = Field(default_factory=list)


class QualityMetrics(BaseModel):
    project_id: ProjectId
    class_balance: dict[str, int]
    verification_pass_rate: float
    mixed_label_clusters: int
    split_leakage: int
    anomalies: list[str] = Field(default_factory=list)


class QualityReport(BaseModel):
    project_id: ProjectId
    metrics: QualityMetrics
    shippable: bool
    recommendations: list[str] = Field(default_factory=list)


class SnapshotManifest(BaseModel):
    id: SnapshotId
    project_id: ProjectId
    plan_version: int
    image_hashes: list[Sha256]
    annotation_set_hash: Sha256
    created_at: str  # ISO-8601; stamped by the caller, not core


# --- Phase 1 execution planning & reporting ---------------------------------
class ProcessingPlan(BaseModel):
    """The deterministic execution plan the Phase-1 Planner produces; every later
    stage follows it (System Design §2.9, Phase-1 scope). Distinct from the
    semantic LabelingPlan, which the LLM Planner produces in Phase 2."""

    project_id: ProjectId
    version: int
    image_count: int
    classes: list[str]  # detection prompts/labels; MVP default ["object"]
    detector: str  # configured adapter name
    segmenter: str
    confidence_threshold: float
    batch_size: int
    num_batches: int
    gpu_budget_mb: int
    export_format: str
    estimated_seconds: float


class StageTiming(BaseModel):
    stage: str
    seconds: float
    items: int


class BenchmarkResult(BaseModel):
    """Everything measured during one pipeline execution (performance + resources)."""

    project_id: ProjectId
    images_processed: int
    total_seconds: float
    images_per_second: float
    avg_inference_ms: float
    stage_seconds: dict[str, float]  # detector/segmentation/verification/export/...
    num_batches: int = 0
    batch_throughput_ips: float = 0.0  # images/sec over the labeling stages
    peak_ram_mb: float = 0.0
    peak_vram_mb: float | None = None
    cpu_percent: float = 0.0
    gpu_util_percent: float | None = None
    created_at: str = ""


class ExportReport(BaseModel):
    format: str
    images: int
    annotations: int
    categories: list[str]
    output_path: str
    validated: bool
    validation_error: str | None = None


class DatasetQualityReport(BaseModel):
    """Deterministic dataset-quality metrics (Phase-5 baseline measurement).

    Pure counting/statistics over the produced annotations — no interpretation.
    The Analyst *interpreting* these numbers is Phase 3; this is the raw layer it
    will read.
    """

    project_id: ProjectId
    images: int
    detections: int
    masks: int
    approval_rate: float
    review_rate: float
    rejection_rate: float
    invalid_annotations: int
    duplicate_detections: int
    empty_masks: int
    annotation_density: float  # detections per image
    avg_confidence: float
    images_with_no_detection: int


class ErrorCategory(BaseModel):
    name: str
    count: int
    description: str
    heuristic: bool = True  # True => a proxy measured without ground truth


class ErrorAnalysis(BaseModel):
    """Deterministic, ground-truth-free failure categorization (Phase-5).

    Every category is a heuristic proxy: with no reference labels in the
    baseline, these measure *suspicious* cases, not confirmed errors. Categories
    needing ground truth (occlusion, lighting, true misses) are listed in
    `unmeasurable` rather than guessed at.
    """

    project_id: ProjectId
    total_annotations: int
    categories: list[ErrorCategory]
    unmeasurable: list[str] = Field(default_factory=list)


class ExecutionReport(BaseModel):
    """The human-facing summary of one end-to-end run."""

    project_id: ProjectId
    source: str
    imported: int
    duplicates_skipped: int
    quarantined: int
    detections: int
    verified_approved: int
    needs_review: int
    rejected: int
    export: ExportReport
    benchmark: BenchmarkResult
    quality: DatasetQualityReport
    errors: ErrorAnalysis


class StageKPIs(BaseModel):
    """Flat KPI record for one pipeline generation, used by the comparison
    framework (Phase-5). Future stages (planner, analyst, feedback, production)
    register their own record; any two are diffable."""

    stage: str  # deterministic | planner | analyst | feedback | production
    label: str
    created_at: str
    images_per_second: float
    approval_rate: float
    review_rate: float
    rejection_rate: float
    avg_confidence: float
    annotation_density: float
    peak_ram_mb: float
    invalid_annotations: int
    empty_masks: int
