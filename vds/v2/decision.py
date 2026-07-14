"""DecisionAgent — the optimization layer between Planner and Execution (V2-23).

The PlannerAgent recommends execution parameters from the user's *intent* (goal
text). The DecisionAgent refines those recommendations using *real dataset metadata
and history* the Planner never saw (file types, counts, resolution, existing classes,
historical review rates, previous exports), producing a ``DecisionReport`` and an
**enriched** ``ExecutionPlan``.

It never changes the user's intent (same steps, same task, same classes) — it only
tunes execution parameters and estimates. It never executes, never touches the
backend, and never mutates the Planner's plan (it enriches a deep copy). It does not
re-derive the Planner's intent logic; it starts from the plan's baseline and adjusts.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from vds.v2.planner import ExecutionPlan, FrameStrategy, ReviewLevel

_STRIDE = {
    FrameStrategy.EVERY_FRAME: 1, FrameStrategy.EVERY_2: 2, FrameStrategy.EVERY_5: 5,
    FrameStrategy.EVERY_10: 10, FrameStrategy.SCENE_CHANGE: 10, FrameStrategy.ADAPTIVE: 5,
    FrameStrategy.NONE: 1,
}
_THERMAL_TOKENS = {"thermal", "infrared", "ir", "lwir", "mwir", "eo/ir"}
_SEC_PER_IMAGE = {"cpu": 0.05, "gpu": 0.02}  # calibrated ETA per image


class DecisionArea(StrEnum):
    FRAME_SAMPLING = "frame_sampling"
    DETECTION_CONFIDENCE = "detection_confidence"
    IOU_THRESHOLD = "iou_threshold"
    SEGMENTATION = "segmentation"
    EXPORT_FORMAT = "export_format"
    REVIEW_LEVEL = "review_level"
    BATCH_SIZE = "batch_size"
    COMPUTE = "compute"
    DUPLICATE_REMOVAL = "duplicate_removal"
    EXPECTED_RUNTIME = "expected_runtime"
    ANNOTATION_COUNT = "annotation_count"


class DatasetMetadata(BaseModel):
    """Real facts about the data — the input the Planner lacked. All optional."""

    file_types: list[str] = Field(default_factory=list)
    image_count: int = 0
    video_duration_seconds: float | None = None
    fps: float | None = None
    resolution: str | None = None  # "low" | "medium" | "high" | "1920x1080"
    existing_classes: list[str] = Field(default_factory=list)
    historical_stats: dict = Field(default_factory=dict)  # avg_objects_per_image, avg_review_rate, false_positive_rate
    previous_exports: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    """One optimization decision. Every field the brief mandates is present."""

    area: str
    value: str
    reason: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)  # shown as a % in the UI
    alternative: str = ""
    impact: str = ""
    tradeoffs: str = ""


class DecisionReport(BaseModel):
    plan_id: str
    decisions: list[Decision] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    estimated_runtime_seconds: float = 0.0
    estimated_export_size_mb: float = 0.0
    expected_annotation_count: int = 0
    recommended_review: str = ReviewLevel.MEDIUM.value
    overall_confidence: float = 0.0

    def get(self, area: str) -> Decision | None:
        return next((d for d in self.decisions if d.area == area), None)


def _res_class(resolution: str | None) -> str:
    if not resolution:
        return "medium"
    r = resolution.lower()
    if r in ("low", "medium", "high"):
        return r
    if "x" in r:
        try:
            w, h = (int(p) for p in r.split("x")[:2])
        except ValueError:
            return "medium"
        m = max(w, h)
        return "high" if m >= 1600 else "low" if m <= 640 else "medium"
    return "medium"


class DecisionAgent:
    def decide(self, plan: ExecutionPlan, metadata: DatasetMetadata) -> tuple[ExecutionPlan, DecisionReport]:
        meta = metadata
        thermal = bool({t.lower() for t in meta.file_types} & _THERMAL_TOKENS)
        res = _res_class(meta.resolution)
        decisions: list[Decision] = []

        video = plan.frame_strategy != FrameStrategy.NONE or bool(meta.video_duration_seconds)
        frame = plan.frame_strategy
        if video:
            frame, fdec = self._frame(plan, meta)
            decisions.append(fdec)

        count = self._expected_count(plan, meta, frame, video)
        device = "gpu" if count > 500 else "cpu"

        conf, d = self._confidence(plan, meta, thermal)
        decisions.append(d)
        iou, d = self._iou(plan, meta)
        decisions.append(d)
        seg, d = self._segmentation(plan)
        decisions.append(d)
        fmts, d = self._export_format(plan, meta, seg)
        decisions.append(d)
        review, d = self._review(plan, meta, thermal, seg)
        decisions.append(d)
        batch, d = self._batch(res, device)
        decisions.append(d)
        decisions.append(self._compute(count, device))
        dedup, d = self._dedup(count, video)
        decisions.append(d)

        runtime = round(count * _SEC_PER_IMAGE[device], 2)
        decisions.append(self._runtime(runtime, device, count))
        anno = self._annotation_count(count, meta)
        decisions.append(self._annotation(anno, count))

        warnings, suggestions = self._notes(meta, thermal, count, seg, device, fmts)
        enriched = self._enrich(plan, conf, iou, seg, frame, fmts, review, batch, device,
                                count, runtime, dedup, warnings, suggestions)
        overall = round(sum(x.confidence for x in decisions) / len(decisions), 3) if decisions else 0.0
        report = DecisionReport(
            plan_id=plan.id, decisions=decisions, warnings=warnings, suggestions=suggestions,
            estimated_runtime_seconds=runtime, estimated_export_size_mb=round(count * 0.2, 1),
            expected_annotation_count=anno, recommended_review=review, overall_confidence=overall)
        return enriched, report

    # --- per-area decisions -------------------------------------------
    def _frame(self, plan: ExecutionPlan, meta: DatasetMetadata) -> tuple[FrameStrategy, Decision]:
        dur = meta.video_duration_seconds or 0
        density = meta.historical_stats.get("avg_objects_per_image", 0)
        if density > 8 or (0 < dur < 30):
            strat, reason = FrameStrategy.EVERY_2, "Dense or short footage — sample frequently to catch objects."
        elif dur > 300:
            strat, reason = FrameStrategy.EVERY_10, "Long footage — sample sparsely to bound annotation cost."
        else:
            strat, reason = FrameStrategy.EVERY_5, "Typical footage — balanced sampling."
        return strat, Decision(area=DecisionArea.FRAME_SAMPLING, value=strat.value, reason=reason,
                               confidence=0.7, alternative=FrameStrategy.ADAPTIVE.value,
                               impact="Sets dataset size and annotation cost.",
                               tradeoffs="denser = more coverage, more labeling")

    def _confidence(self, plan: ExecutionPlan, meta: DatasetMetadata, thermal: bool) -> tuple[float, Decision]:
        base = plan.recommended_confidence or 0.30
        fpr = meta.historical_stats.get("false_positive_rate", 0)
        if thermal:
            val, reason, conf = 0.20, "Thermal imagery has low contrast; a lower threshold recovers detections.", 0.92
            impact, trade, alt = "More detections, more false positives.", "recall up / precision down", "0.30"
        elif fpr > 0.3:
            val, reason, conf = 0.45, "Prior runs showed many false positives; raise the threshold.", 0.8
            impact, trade, alt = "Fewer false positives, fewer detections.", "precision up / recall down", f"{base:.2f}"
        else:
            val, reason, conf = base, "Planner threshold suits this dataset.", 0.6
            impact, trade, alt = "Balanced precision/recall.", "—", f"{base + 0.10:.2f}"
        return val, Decision(area=DecisionArea.DETECTION_CONFIDENCE, value=f"{val:.2f}", reason=reason,
                             confidence=conf, alternative=alt, impact=impact, tradeoffs=trade)

    def _iou(self, plan: ExecutionPlan, meta: DatasetMetadata) -> tuple[float, Decision]:
        base = plan.recommended_iou or 0.45
        density = meta.historical_stats.get("avg_objects_per_image", 0)
        if density > 8:
            val, reason, conf = 0.60, "Crowded scenes: keep overlapping boxes.", 0.75
        else:
            val, reason, conf = base, "Default overlap suits sparse scenes.", 0.6
        alt = "0.45" if val != 0.45 else "0.60"
        return val, Decision(area=DecisionArea.IOU_THRESHOLD, value=f"{val:.2f}", reason=reason,
                             confidence=conf, alternative=alt, impact="Controls NMS suppression.",
                             tradeoffs="higher = more overlapping boxes kept")

    def _segmentation(self, plan: ExecutionPlan) -> tuple[bool, Decision]:
        seg = plan.recommended_segmentation
        return seg, Decision(
            area=DecisionArea.SEGMENTATION, value=str(seg).lower(),
            reason="Masks required by the plan." if seg else "Boxes suffice; segmentation off saves time.",
            confidence=0.8 if seg else 0.7, alternative="disable" if seg else "enable",
            impact="Higher annotation cost." if seg else "Faster; no masks.",
            tradeoffs="masks vs speed")

    def _export_format(self, plan: ExecutionPlan, meta: DatasetMetadata, seg: bool) -> tuple[list[str], Decision]:
        if meta.previous_exports:
            fmts = list(dict.fromkeys(f.lower() for f in meta.previous_exports))
            reason = "Match previous exports for consistency."
        elif seg:
            fmts, reason = ["coco", "yolo"], "COCO carries masks; YOLO carries boxes."
        else:
            fmts, reason = [self._plan_export(plan)], "Planner default format."
        return fmts, Decision(area=DecisionArea.EXPORT_FORMAT, value=",".join(fmts), reason=reason,
                              confidence=0.7, alternative="coco" if fmts != ["coco"] else "yolo",
                              impact="Downstream tool compatibility.", tradeoffs="more formats = more disk")

    def _review(self, plan: ExecutionPlan, meta: DatasetMetadata, thermal: bool, seg: bool) -> tuple[str, Decision]:
        rate = meta.historical_stats.get("avg_review_rate", 0)
        level = plan.estimated_review
        if rate > 0.3 or thermal or seg:
            level = ReviewLevel.HIGH
        elif rate and rate < 0.1:
            level = ReviewLevel.LOW
        reason = "Historical review rate / complexity" if rate else "Dataset complexity"
        return level.value, Decision(area=DecisionArea.REVIEW_LEVEL, value=level.value,
                                     reason=f"{reason} sets the review effort.", confidence=0.7,
                                     alternative=ReviewLevel.MEDIUM.value, impact="Human hours needed.",
                                     tradeoffs="thoroughness vs cost")

    def _batch(self, res: str, device: str) -> tuple[int, Decision]:
        batch = {"high": 4, "medium": 8, "low": 16}[res]
        if device == "cpu":
            batch = max(2, batch // 2)
        return batch, Decision(area=DecisionArea.BATCH_SIZE, value=str(batch),
                               reason=f"{res}-resolution on {device.upper()}.", confidence=0.65,
                               alternative=str(batch * 2), impact="Throughput vs memory.",
                               tradeoffs="larger = faster but more VRAM/RAM")

    def _compute(self, count: int, device: str) -> Decision:
        reason = "Large dataset — GPU cuts runtime." if device == "gpu" else "Small dataset — CPU is sufficient."
        return Decision(area=DecisionArea.COMPUTE, value=device.upper(), reason=reason, confidence=0.7,
                        alternative="CPU" if device == "gpu" else "GPU",
                        impact="Runtime and VRAM budget.", tradeoffs="speed vs 8 GB VRAM limit")

    def _dedup(self, count: int, video: bool) -> tuple[bool, Decision]:
        enabled = count > 2000 or video
        return enabled, Decision(
            area=DecisionArea.DUPLICATE_REMOVAL, value=str(enabled).lower(),
            reason="Video / large dataset has near-duplicate frames." if enabled else "Small image set; keep all.",
            confidence=0.7, alternative="disable" if enabled else "enable",
            impact="Fewer redundant images to review." if enabled else "Nothing removed.",
            tradeoffs="cost savings vs possible coverage loss")

    def _runtime(self, runtime: float, device: str, count: int) -> Decision:
        return Decision(area=DecisionArea.EXPECTED_RUNTIME, value=f"{runtime}s",
                        reason=f"{count} images at {_SEC_PER_IMAGE[device]}s/image on {device.upper()}.",
                        confidence=0.6, alternative="—", impact="Wall-clock estimate.", tradeoffs="—")

    def _annotation(self, anno: int, count: int) -> Decision:
        return Decision(area=DecisionArea.ANNOTATION_COUNT, value=str(anno),
                        reason=f"{count} images at the historical objects/image rate.",
                        confidence=0.55, alternative="—", impact="Review workload driver.", tradeoffs="—")

    # --- helpers -------------------------------------------------------
    @staticmethod
    def _plan_export(plan: ExecutionPlan) -> str:
        step = plan.get("export_dataset")
        return step.arguments.get("format", "coco") if step else "coco"

    @staticmethod
    def _expected_count(plan: ExecutionPlan, meta: DatasetMetadata, frame: FrameStrategy, video: bool) -> int:
        if video and meta.video_duration_seconds and meta.fps:
            return int(meta.video_duration_seconds * meta.fps / _STRIDE.get(frame, 5))
        return meta.image_count or plan.estimated_dataset_size or 0

    @staticmethod
    def _annotation_count(count: int, meta: DatasetMetadata) -> int:
        avg = meta.historical_stats.get("avg_objects_per_image", 3)
        return int(count * avg)

    @staticmethod
    def _notes(meta, thermal, count, seg, device, fmts) -> tuple[list[str], list[str]]:
        warnings, suggestions = [], []
        if thermal:
            warnings.append("Thermal imagery: lower object contrast may reduce recall.")
        if count == 0:
            warnings.append("No dataset metadata; estimates are unavailable.")
        if count > 5000:
            warnings.append("Large dataset — enable duplicate removal and prefer GPU.")
        if seg and device == "cpu":
            suggestions.append("Segmentation on CPU is slow; a GPU is recommended.")
        if len(fmts) > 1:
            suggestions.append(f"Export both formats: {', '.join(fmts)}.")
        return warnings, suggestions

    @staticmethod
    def _enrich(plan, conf, iou, seg, frame, fmts, review, batch, device, count, runtime, dedup,
                warnings, suggestions) -> ExecutionPlan:
        """Write decided parameters into a DEEP COPY — the Planner's plan is untouched."""
        p = plan.model_copy(deep=True)
        p.recommended_confidence = conf
        p.recommended_iou = iou
        p.recommended_segmentation = seg
        p.frame_strategy = frame
        p.estimated_dataset_size = count
        p.estimated_runtime_seconds = runtime
        p.estimated_review = ReviewLevel(review)
        p.warnings = list(dict.fromkeys([*p.warnings, *warnings, *suggestions]))
        for s in p.steps:
            if s.id == "run_detection":
                s.arguments.update({"confidence": conf, "iou": iou,
                                    "batch_size": batch, "device": device})
            elif s.id == "export_dataset":
                s.arguments["format"] = fmts[0]  # primary; extras noted in suggestions
            elif s.id == "extract_frames":
                s.arguments["frame_strategy"] = frame.value
            elif s.id in ("import_images", "import_video"):
                s.arguments["dedup"] = dedup
        return p

    # --- overrides (GUI: accept / reject / override) -------------------
    def apply_overrides(self, plan: ExecutionPlan, report: DecisionReport,
                        overrides: dict) -> tuple[ExecutionPlan, DecisionReport]:
        """Apply user overrides (area -> value) to the enriched plan + report. The
        user's value wins with full confidence; other decisions are untouched."""
        p = plan.model_copy(deep=True)
        r = report.model_copy(deep=True)
        for area, value in overrides.items():
            d = r.get(area)
            if d is not None:
                d.value, d.reason, d.confidence = str(value), "User override.", 1.0
            self._apply_param(p, area, value)
        r.overall_confidence = round(sum(x.confidence for x in r.decisions) / len(r.decisions), 3)
        return p, r

    @staticmethod
    def _apply_param(p: ExecutionPlan, area: str, value) -> None:
        if area == DecisionArea.DETECTION_CONFIDENCE:
            p.recommended_confidence = float(value)
        elif area == DecisionArea.IOU_THRESHOLD:
            p.recommended_iou = float(value)
        elif area == DecisionArea.SEGMENTATION:
            p.recommended_segmentation = str(value).lower() in ("true", "1", "yes", "enable")
        elif area == DecisionArea.FRAME_SAMPLING:
            p.frame_strategy = FrameStrategy(value)
        elif area == DecisionArea.REVIEW_LEVEL:
            p.estimated_review = ReviewLevel(value)
        for s in p.steps:
            if s.id == "run_detection" and area == DecisionArea.DETECTION_CONFIDENCE:
                s.arguments["confidence"] = float(value)
            elif s.id == "run_detection" and area == DecisionArea.IOU_THRESHOLD:
                s.arguments["iou"] = float(value)
            elif s.id == "run_detection" and area == DecisionArea.BATCH_SIZE:
                s.arguments["batch_size"] = int(value)
            elif s.id == "export_dataset" and area == DecisionArea.EXPORT_FORMAT:
                s.arguments["format"] = str(value).split(",")[0]
            elif s.id == "extract_frames" and area == DecisionArea.FRAME_SAMPLING:
                s.arguments["frame_strategy"] = str(value)


def decision_view(report: DecisionReport) -> dict:
    """GUI Decision Summary panel data (V2-23 §GUI). No Qt — a future phase renders it."""
    return {
        "recommendations": [d.model_dump() for d in report.decisions],
        "warnings": list(report.warnings),
        "suggestions": list(report.suggestions),
        "estimated_runtime_seconds": report.estimated_runtime_seconds,
        "estimated_export_size_mb": report.estimated_export_size_mb,
        "expected_annotation_count": report.expected_annotation_count,
        "review_level": report.recommended_review,
        "overall_confidence": report.overall_confidence,
        "reasoning": [{"area": d.area, "reason": d.reason, "confidence": d.confidence,
                       "alternative": d.alternative, "tradeoffs": d.tradeoffs} for d in report.decisions],
    }
