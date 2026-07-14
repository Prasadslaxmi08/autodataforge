"""BackendController — the single seam between the GUI and the platform (Phase 11).

The UI talks to the backend ONLY through this object. It owns a `Container` (the
existing composition root) and exposes plain-Python methods returning plain data —
no Qt here, no business logic reimplemented. Every method delegates to an existing
backend module (pipeline, repos, memory, settings). This keeps UI and backend
strictly separated: the controller could be driven by a CLI or a test just as well
as by the desktop app.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from vds.container import Container
from vds.core.contracts import (
    Annotation,
    Box2D,
    ExecutionReport,
    ExportReport,
    Mask,
    Provenance,
)
from vds.core.enums import AnnotationState
from vds.core.geometry import mask_is_empty

_REJECTED = (AnnotationState.REJECTED, AnnotationState.REJECTED_AUTO)


@dataclass
class DatasetSummary:
    """A row in the Dataset Manager — everything the list/table needs, precomputed
    off the UI thread."""

    project_id: str
    name: str
    phase: str
    image_count: int
    annotation_count: int
    approved: int = 0
    needs_review: int = 0
    rejected: int = 0
    thumbnails: list[str] = field(default_factory=list)  # CAS file paths


@dataclass
class ImageRef:
    """One image in a project — everything the editor/filmstrip needs up front."""

    image_id: str
    sha256: str
    path: str  # CAS file path (for QPixmap)
    width: int
    height: int


@dataclass
class EditableBox:
    """A box annotation in an editable, Qt-free shape. `id` is "" for a not-yet-saved
    box (an AI proposal or a freshly drawn one)."""

    id: str
    x: float
    y: float
    w: float
    h: float
    label: str
    confidence: float
    state: str
    has_mask: bool = False


class BackendController:
    def __init__(self, container: Container | None = None) -> None:
        self._c = container or Container()

    @property
    def container(self) -> Container:
        return self._c

    # --- environment / dashboard ---------------------------------------
    def current_model(self) -> str:
        return self._c.settings.llm.model

    def current_provider(self) -> str:
        return self._c.settings.llm.provider.split(":")[-1]

    def dashboard_snapshot(self) -> dict:
        """Everything the Dashboard shows, read once from the backend."""
        projects = self._c.projects.list()
        memories = self._c.memory.all()
        return {
            "recent_projects": [(p.id, p.name, p.phase) for p in projects[-8:]],
            "dataset_count": len(projects),
            "current_model": self.current_model(),
            "current_provider": self.current_provider(),
            "gpu_device": self._c.settings.gpu.device,
            "vram_budget_mb": self._c.settings.gpu.vram_budget_mb,
            "latest_benchmarks": self._latest_benchmarks(),
            "recent_memory": [
                (m.id, m.created_at, m.benchmark_summary.quality_score,
                 m.execution_metrics.review_rate)
                for m in memories[-5:]
            ],
            "memory_count": len(memories),
            "recent_analyst_report": self._recent_file("benchmarks/analyst_report_example.md"),
            "pipeline_status": "idle",
        }

    def _latest_benchmarks(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        bench = Path("benchmarks")
        if bench.exists():
            for f in sorted(bench.glob("*.json")):
                out.append((f.name, f"{f.stat().st_size} bytes"))
        return out[-6:]

    @staticmethod
    def _recent_file(path: str) -> str | None:
        p = Path(path)
        return p.name if p.exists() else None

    # --- datasets (Dataset Manager) ------------------------------------
    def list_datasets(self, *, thumbnails: int = 0) -> list[DatasetSummary]:
        out: list[DatasetSummary] = []
        for p in self._c.projects.list():
            images = self._c.images.by_project(p.id)
            anns = [a for img in images for a in self._c.annotations.by_image(img.id)]
            approved = sum(1 for a in anns if a.state == "auto_accepted")
            review = sum(1 for a in anns if a.state == "needs_review")
            rejected = sum(1 for a in anns if a.state in ("rejected", "rejected_auto"))
            thumbs = [str(self._c.cas.path(img.sha256)) for img in images[:thumbnails]]
            out.append(DatasetSummary(
                project_id=p.id, name=p.name, phase=p.phase, image_count=len(images),
                annotation_count=len(anns), approved=approved, needs_review=review,
                rejected=rejected, thumbnails=thumbs,
            ))
        return out

    def dataset_detail(self, project_id: str, *, thumbnails: int = 12) -> DatasetSummary | None:
        proj = self._c.projects.get(project_id)
        if proj is None:
            return None
        images = self._c.images.by_project(project_id)
        anns = [a for img in images for a in self._c.annotations.by_image(img.id)]
        return DatasetSummary(
            project_id=proj.id, name=proj.name, phase=proj.phase, image_count=len(images),
            annotation_count=len(anns),
            approved=sum(1 for a in anns if a.state == "auto_accepted"),
            needs_review=sum(1 for a in anns if a.state == "needs_review"),
            rejected=sum(1 for a in anns if a.state in ("rejected", "rejected_auto")),
            thumbnails=[str(self._c.cas.path(img.sha256)) for img in images[:thumbnails]],
        )

    def set_detector_config(
        self, model: str, conf: float, iou: float, imgsz: int, segment: bool
    ) -> None:
        """Set the active YOLO run parameters (model weights, confidence, IoU, image
        size, segmentation) before an import. No-op for the builtin detector — this
        only tunes the YOLO adapter once it is the selected detector."""
        from vds.models.adapters.yolo_config import YoloRuntimeConfig, set_config

        set_config(YoloRuntimeConfig(model=model, conf=conf, iou=iou,
                                     imgsz=imgsz, segment=segment))

    def import_dataset(
        self,
        source: str,
        name: str,
        *,
        export_format: str = "coco",
        progress: Callable[[int, str], None] | None = None,
    ) -> ExecutionReport:
        """Run the FULL existing pipeline (import → plan → label → verify → export)
        on a folder. Long-running: callers MUST run this off the UI thread. Progress
        is coarse (the pipeline is monolithic); milestones are reported around it."""
        def emit(pct: int, msg: str) -> None:
            if progress is not None:
                progress(pct, msg)

        emit(5, f"Importing '{name}' from {source}")
        report = self._c.pipeline.run(
            source, name=name, export_format=export_format,
            dest=str(Path("export") / name),
        )
        self._cache_report(report)  # GUI-side cache of the backend's own report
        emit(90, "Pipeline finished; collecting results")
        emit(100, f"Imported {report.imported} images, {report.detections} annotations")
        return report

    def _intelligence_dir(self) -> Path:
        # Sibling of the CAS root, so it's isolated per environment/test container.
        return Path(self._c.settings.storage.cas_root).parent / "intelligence"

    def _cache_report(self, report: ExecutionReport) -> None:
        """Persist the ExecutionReport the pipeline returned so the Intelligence
        workspace can feed it to the real Analyst later. Not a backend change — the
        GUI caches what the backend already produced."""
        d = self._intelligence_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{report.project_id}.json").write_text(report.model_dump_json(indent=2),
                                                     encoding="utf-8")

    def cached_report(self, project_id: str) -> ExecutionReport | None:
        path = self._intelligence_dir() / f"{project_id}.json"
        if not path.exists():
            return None
        return ExecutionReport.model_validate_json(path.read_text(encoding="utf-8"))

    def rename_dataset(self, project_id: str, name: str) -> None:
        self._c.projects.rename(project_id, name)

    def delete_dataset(self, project_id: str) -> None:
        self._c.projects.delete(project_id)

    def export_project(
        self, project_id: str, fmt: str = "coco", dest: str | None = None
    ) -> ExportReport:
        """Re-export an existing dataset via the existing ExportService — no pipeline
        run. Reuses the same service the pipeline uses at import time."""
        out = dest or str(Path("export") / project_id)
        return self._c.exporter.run(project_id, fmt, out)

    # --- annotation editor (Phase 19) ----------------------------------
    # Every method here reuses the existing repos/services. Editing persists through
    # the existing add()/set_state() only: the backend is frozen.
    def project_images(self, project_id: str) -> list[ImageRef]:
        return [
            ImageRef(img.id, img.sha256, str(self._c.cas.path(img.sha256)),
                     img.width, img.height)
            for img in self._c.images.by_project(project_id)
        ]

    def image_boxes(self, image_id: str) -> list[EditableBox]:
        """Editable (non-rejected) box annotations for one image."""
        out: list[EditableBox] = []
        for a in self._c.annotations.by_image(image_id):
            if a.state in _REJECTED or getattr(a.geometry, "kind", None) != "box2d":
                continue
            g = a.geometry
            has_mask = a.mask is not None and not mask_is_empty(a.mask.rle)
            out.append(EditableBox(a.id, g.x, g.y, g.w, g.h, a.label,
                                   a.confidence, a.state, has_mask))
        return out

    def box_mask(self, annotation_id: str) -> dict | None:
        """The RLE mask of a saved annotation (for the read-only overlay), or None."""
        a = self._c.annotations.get(annotation_id)
        if a is None or a.mask is None or mask_is_empty(a.mask.rle):
            return None
        return {"rle": a.mask.rle, "width": a.mask.width, "height": a.mask.height}

    def _make_annotation(self, image_id: str, op: dict, state: AnnotationState) -> Annotation:
        b = op["box"]
        mask = None
        if op.get("mask"):
            m = op["mask"]
            mask = Mask(rle=m["rle"], width=m["width"], height=m["height"])
        return Annotation(
            id=uuid.uuid4().hex, image_id=image_id, label=op.get("label", "object"),
            geometry=Box2D(x=b["x"], y=b["y"], w=b["w"], h=b["h"]),
            confidence=float(op.get("confidence", 1.0)), state=state,
            mask=mask, provenance=Provenance(source="human.editor"),
        )

    def save_edits(self, image_id: str, ops: list[dict]) -> int:
        """Persist a diff. create → add(ACCEPTED); delete → set_state(REJECTED);
        edit → reject the old row and add a corrected one (FIXED). No update/delete API
        is touched — the backend stays frozen."""
        for op in ops:
            kind = op["op"]
            if kind == "delete":
                self._c.annotations.set_state(op["id"], AnnotationState.REJECTED)
            elif kind == "create":
                self._c.annotations.add(self._make_annotation(image_id, op, AnnotationState.ACCEPTED))
            elif kind == "edit":
                self._c.annotations.set_state(op["id"], AnnotationState.REJECTED)
                self._c.annotations.add(self._make_annotation(image_id, op, AnnotationState.FIXED))
        return len(ops)

    def ai_annotate(self, image_id: str) -> list[EditableBox]:
        """Re-run the configured detector on one image; boxes are proposals (id="")
        the editor adds to the session, committed on Save."""
        from vds.models.protocols import Capability

        record = self._c.images.get(image_id)
        data = self._c.cas.get(record.sha256)
        detector = self._c.models.get(Capability.DETECTOR)
        dets = detector.detect([data], ["object"], {"confidence_threshold": 0.0})[0]
        return [EditableBox("", d.box.x, d.box.y, d.box.w, d.box.h, d.label,
                            d.confidence, "new") for d in dets]

    def resegment(self, image_id: str, box: dict) -> dict:
        """Regenerate a box's mask with the configured segmenter. Returns the RLE."""
        from vds.models.protocols import Capability

        record = self._c.images.get(image_id)
        data = self._c.cas.get(record.sha256)
        segmenter = self._c.models.get(Capability.SEGMENTER)
        mask = segmenter.segment(data, [Box2D(x=box["x"], y=box["y"], w=box["w"], h=box["h"])])
        return {"rle": mask.rle, "width": mask.width, "height": mask.height}

    # --- class manager -------------------------------------------------
    def project_classes(self, project_id: str) -> dict[str, int]:
        """Distinct class labels → count, over non-rejected annotations."""
        counts: dict[str, int] = {}
        for img in self._c.images.by_project(project_id):
            for a in self._c.annotations.by_image(img.id):
                if a.state in _REJECTED:
                    continue
                counts[a.label] = counts.get(a.label, 0) + 1
        return counts

    def rename_class(self, project_id: str, old: str, new: str) -> int:
        """Relabel every non-rejected `old` annotation to `new` (reject-old + add-new,
        preserving geometry and mask)."""
        changed = 0
        for img in self._c.images.by_project(project_id):
            for a in self._c.annotations.by_image(img.id):
                if a.label != old or a.state in _REJECTED:
                    continue
                self._c.annotations.set_state(a.id, AnnotationState.REJECTED)
                self._c.annotations.add(a.model_copy(update={
                    "id": uuid.uuid4().hex, "label": new,
                    "state": AnnotationState.FIXED,
                    "provenance": Provenance(source="human.editor"),
                }))
                changed += 1
        return changed

    def merge_classes(self, project_id: str, sources: list[str], target: str) -> int:
        return sum(self.rename_class(project_id, s, target) for s in sources if s != target)

    def delete_class(self, project_id: str, label: str) -> int:
        changed = 0
        for img in self._c.images.by_project(project_id):
            for a in self._c.annotations.by_image(img.id):
                if a.label == label and a.state not in _REJECTED:
                    self._c.annotations.set_state(a.id, AnnotationState.REJECTED)
                    changed += 1
        return changed

    # --- planner workspace ---------------------------------------------
    def plan_dataset(self, project_id: str, overrides=None):
        """Run the existing Planner Agent for a dataset and assemble the workspace
        view. Long-running (LLM call) — run off the UI thread. `overrides` is a
        PlanControls of engineer edits; passing None uses the Planner's own plan."""
        from vds.gui.planner_view import build_plan_view

        return build_plan_view(self._c, project_id, overrides)

    def detector_options(self) -> list[str]:
        from vds.gui.planner_view import DETECTOR_OPTIONS

        return DETECTOR_OPTIONS

    def export_options(self) -> list[str]:
        from vds.gui.planner_view import EXPORT_OPTIONS

        return EXPORT_OPTIONS

    # --- pipeline workspace --------------------------------------------
    def run_pipeline(
        self,
        source: str,
        name: str,
        *,
        export_format: str = "coco",
        progress: Callable[[int, str], None] | None = None,
    ) -> ExecutionReport:
        """Execute the existing pipeline (unchanged) on a folder. Long-running —
        run off the UI thread. Alias of the Dataset Manager import path, named for
        the monitoring workspace."""
        return self.import_dataset(source, name, export_format=export_format, progress=progress)

    def stage_timeline(self, report: ExecutionReport):
        from vds.gui import pipeline_view

        return pipeline_view.stage_timeline(report)

    def running_timeline(self):
        from vds.gui import pipeline_view

        return pipeline_view.running_timeline()

    def model_activity(self, report: ExecutionReport):
        from vds.gui import pipeline_view

        return pipeline_view.model_activity(self._c, report)

    def pipeline_metrics(self, report: ExecutionReport, *, elapsed_s: float, cpu: float, ram_mb: float):
        from vds.gui import pipeline_view

        return pipeline_view.metrics(report, elapsed_s=elapsed_s, cpu=cpu, ram_mb=ram_mb)

    def console_events(self, report: ExecutionReport):
        from vds.gui import pipeline_view

        return pipeline_view.console_events(report)

    def pipeline_preview(self, project_id: str, limit: int = 12):
        from vds.gui import pipeline_view

        return pipeline_view.preview_items(self._c, project_id, limit)

    def pipeline_summary(self, report: ExecutionReport):
        from vds.gui import pipeline_view

        return pipeline_view.pipeline_summary(self._c, report)

    def source_images(self, source: str, limit: int = 24) -> list[str]:
        """Original image file paths from a source folder (live preview before a run)."""
        root = Path(source)
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        return [str(p) for p in sorted(root.rglob("*"))
                if p.is_file() and p.suffix.lower() in exts][:limit]

    def report_markdown(self, report: ExecutionReport | None) -> str:
        """Render the run report via the existing reporting infrastructure.

        ponytail: review-/export-only plans have no pipeline ExecutionReport
        (only import runs produce one), so report is None here — return a note
        instead of dereferencing .benchmark and crashing."""
        from vds.reporting import build_report

        if report is None:
            return "# Run Report\n\nNo pipeline run report for this task (review/export only)."
        return build_report(report)

    def save_report_file(self, report: ExecutionReport) -> str:
        from vds.reporting import save_report

        return str(save_report(report))

    def export_dir(self, name: str) -> str:
        return str(Path("export") / name)

    # --- verification workspace ----------------------------------------
    def object_verdicts(self, project_id: str):
        """Reproduce (deterministically, via the existing verifier) the verdict for
        every annotation in a dataset. Long-ish — run off the UI thread."""
        from vds.gui import verification_view

        return verification_view.object_verdicts(self._c, project_id)

    def object_evidence(self, verdict):
        from vds.gui import verification_view

        return verification_view.evidence_for(self._c, verdict)

    def verification_stats(self, verdicts):
        from vds.gui import verification_view

        return verification_view.verification_stats(self._c, verdicts)

    def verification_history(self, project_id: str):
        from vds.gui import verification_view

        return verification_view.historical_comparison(self._c, project_id)

    def verification_timeline(self, verdict):
        from vds.gui import verification_view

        return verification_view.timeline_for(verdict)

    def apply_review(self, annotation_id: str, action: str) -> tuple[bool, str]:
        from vds.gui import verification_view

        return verification_view.apply_review(self._c, annotation_id, action)

    def verification_report_markdown(self, project_id: str) -> str:
        from vds.gui import verification_view

        return verification_view.verification_report(self._c, project_id)

    # --- dataset intelligence workspace --------------------------------
    def analyze_dataset(self, project_id: str, created_at: str):
        """Run the existing AI Dataset Analyst over the cached ExecutionReport and
        assemble the intelligence view. Returns None if no run is cached for this
        dataset. Long-running (LLM call) — run off the UI thread."""
        from vds.gui.intelligence_view import build_intelligence

        return build_intelligence(self._c, project_id, created_at)

    def intelligence_markdown(self, intel, section: str = "all") -> str:
        """Export a section of the intelligence as Markdown, from measured/validated
        content only. 'engineering' reuses the Analyst report rendering."""
        from vds.gui.intelligence_export import to_markdown

        return to_markdown(intel, section)

    # --- video dataset import (Phase 17.5) -----------------------------
    def probe_video(self, path: str):
        """Read video metadata (no extraction). Fast — but may shell to ffprobe, so
        run it off the UI thread for real videos."""
        from vds.video import probe

        return probe(path)

    def video_frame_estimate(self, info, config) -> int:
        from vds.gui.video_import_view import estimate_frames

        return estimate_frames(info, config)

    def plan_video(self, info, config):
        """Existing Planner Agent pre-analysis over the estimated extracted dataset.
        Long-running (LLM call) — run off the UI thread."""
        from vds.gui.video_import_view import build_video_plan

        return build_video_plan(self._c, info, config)

    def video_thumbnail(self, path: str, dest_dir: str) -> str | None:
        """Save the first frame as a PNG thumbnail and return its path (or None)."""
        from vds.video import open_source

        src = open_source(path)
        try:
            for _idx, _ts, img in src.iter_frames([0]):
                out = Path(dest_dir) / "video_thumb.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                img.save(out, format="PNG")
                return str(out)
        finally:
            src.close()
        return None

    def import_video_dataset(
        self,
        video_path: str,
        name: str,
        config,
        *,
        dedup: bool = True,
        export_format: str = "coco",
        cancel: Callable[[], bool] | None = None,
        progress: Callable[[int, str], None] | None = None,
    ):
        """Extract a video into a standard image folder, then run the EXISTING
        pipeline on it. Returns (ExecutionReport, ExtractionStats, VideoInfo). The
        video's per-frame metadata manifest is preserved alongside the dataset so it
        stays accessible after import. Long-running — run off the UI thread."""
        from vds.video import import_video

        def emit(pct: int, msg: str) -> None:
            if progress is not None:
                progress(pct, msg)

        frames_dir = self._video_frames_dir(name)
        emit(3, f"Extracting frames from {Path(video_path).name}")
        vres = import_video(video_path, config, frames_dir, dedup=dedup,
                            cancel=cancel, progress=lambda p, m: emit(min(60, p), m))
        if vres.stats.cancelled:
            return None, vres.stats, vres.info
        emit(62, f"Extracted {vres.stats.unique_frames} frames; running the pipeline")
        report = self._c.pipeline.run(
            str(frames_dir), name=name, export_format=export_format,
            dest=str(Path("export") / name), dedup=dedup)
        self._cache_report(report)
        self._save_video_manifest(report.project_id, vres)
        emit(100, f"Imported {report.imported} images from video")
        return report, vres.stats, vres.info

    def _video_frames_dir(self, name: str) -> Path:
        d = Path(self._c.settings.storage.cas_root).parent / "video_frames" / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _video_dir(self) -> Path:
        return Path(self._c.settings.storage.cas_root).parent / "video"

    def _save_video_manifest(self, project_id: str, vres) -> None:
        import json

        d = self._video_dir()
        d.mkdir(parents=True, exist_ok=True)
        payload = {"video": vres.info.as_dict(),
                   "stats": vres.stats.__dict__, "frames": vres.manifest}
        (d / f"{project_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def video_manifest(self, project_id: str) -> dict | None:
        import json

        path = self._video_dir() / f"{project_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def extraction_strategies(self) -> list[str]:
        from vds.video import STRATEGIES

        return list(STRATEGIES)

    # --- knowledge center ----------------------------------------------
    def knowledge_records(self):
        from vds.gui import knowledge_view

        return knowledge_view.list_records(self._c)

    def search_knowledge(self, query: str, field: str = "Keyword"):
        from vds.gui import knowledge_view

        return knowledge_view.search_records(self._c, query, field)

    def knowledge_search_fields(self):
        from vds.gui import knowledge_view

        return knowledge_view.search_fields()

    def knowledge_cards(self):
        from vds.gui import knowledge_view

        return knowledge_view.knowledge_cards(self._c)

    def knowledge_timeline(self):
        from vds.gui import knowledge_view

        return knowledge_view.timeline(self._c)

    def compare_knowledge(self, ids: list[str]):
        from vds.gui import knowledge_view

        return knowledge_view.compare_records(self._c, ids)

    def lessons_learned(self):
        from vds.gui import knowledge_view

        return knowledge_view.lessons_learned(self._c)

    def knowledge_filter_options(self):
        from vds.gui import knowledge_view

        return knowledge_view.filter_options(self._c)

    def knowledge_markdown(self, section: str, ids: list[str] | None = None) -> str:
        """Export a Knowledge Center section as Markdown. 'knowledge_report' and
        'engineering_summary' reuse the existing Engineering Memory reports."""
        from vds.gui.knowledge_export import to_markdown

        return to_markdown(self._c, section, ids)

    # --- operations & performance center -------------------------------
    def ops_snapshot(self, running_jobs: int = 0) -> dict:
        from vds.gui import operations_view

        return operations_view.live_snapshot(running_jobs)

    def ops_overview(self, live: dict):
        from vds.gui import operations_view

        return operations_view.executive_overview(self._c, live)

    def ops_system(self, live: dict):
        from vds.gui import operations_view

        return operations_view.system_performance(self._c, live)

    def ops_benchmarks(self):
        from vds.gui import operations_view

        return operations_view.benchmark_runs(self._c)

    def ops_compare(self, ids: list[str]):
        from vds.gui import operations_view

        return operations_view.compare_runs(self._c, ids)

    def ops_trends(self):
        from vds.gui import operations_view

        return operations_view.historical_trends(self._c)

    def ops_health(self, live: dict):
        from vds.gui import operations_view

        return operations_view.platform_health(self._c, live)

    def ops_filter_options(self):
        from vds.gui import operations_view

        return operations_view.filter_options(self._c)

    def ops_markdown(self, section: str, live: dict | None = None) -> str:
        from vds.gui.operations_export import to_markdown

        return to_markdown(self._c, section, live)

    def image_path(self, image_id: str) -> str | None:
        rec = self._c.images.get(image_id)
        return str(self._c.cas.path(rec.sha256)) if rec else None

    def image_meta(self, image_id: str) -> dict:
        rec = self._c.images.get(image_id)
        if rec is None:
            return {"name": image_id, "resolution": "unavailable"}
        return {"name": rec.id[:12], "resolution": f"{rec.width}×{rec.height}",
                "width": rec.width, "height": rec.height}
