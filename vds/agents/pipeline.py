"""Phase-1 end-to-end pipeline (the concrete Orchestrator for this phase).

Runs Import -> Plan -> Detect -> Segment -> Verify -> Export synchronously,
advancing the project through its phases and applying the annotation state
machine per verdict. The async job/worker execution of these same stages is
Phase 2 wiring; the MVP proves the sequence end to end.
"""

from __future__ import annotations

import uuid

from vds.agents.orchestrator import guard_phase
from vds.agents.planner import ExecutionPlanner
from vds.agents.verifier import APPROVED, REJECTED, RuleBasedVerifier
from vds.benchmark import BENCHMARKS_DIR, BenchmarkCollector
from vds.comparison import ComparisonRegistry
from vds.core.contracts import ExecutionReport, Project
from vds.core.enums import AnnotationState, ProjectPhase, assert_transition
from vds.engine.labeler import LabelingEngine
from vds.export.service import ExportService
from vds.ingest.service import ImportService
from vds.logging import bind, get_logger
from vds.quality.metrics import QualityAnalyzer
from vds.reporting import REPORTS_DIR, save_report, to_kpis
from vds.store.cas import Cas
from vds.store.sqlite import AnnotationRepo, ImageRepo, ProjectRepo

log = get_logger(__name__)


class Phase1Pipeline:
    def __init__(
        self,
        *,
        projects: ProjectRepo,
        images: ImageRepo,
        annotations: AnnotationRepo,
        cas: Cas,
        importer: ImportService,
        planner: ExecutionPlanner,
        engine: LabelingEngine,
        verifier: RuleBasedVerifier,
        exporter: ExportService,
        analyzer: QualityAnalyzer,
        registry: ComparisonRegistry | None = None,
        reports_dir=REPORTS_DIR,
        benchmarks_dir=BENCHMARKS_DIR,
    ) -> None:
        self._projects = projects
        self._images = images
        self._annotations = annotations
        self._cas = cas
        self._importer = importer
        self._planner = planner
        self._engine = engine
        self._verifier = verifier
        self._exporter = exporter
        self._analyzer = analyzer
        self._registry = registry
        self._reports_dir = reports_dir
        self._benchmarks_dir = benchmarks_dir

    def run(
        self,
        source: str,
        *,
        name: str = "phase1",
        brief: str = "",
        export_format: str = "coco",
        dest: str = "export",
        dedup: bool = True,
    ) -> ExecutionReport:
        project_id = uuid.uuid4().hex
        bind(project_id=project_id)
        bench = BenchmarkCollector(project_id)

        project = Project(
            id=project_id, name=name, brief=brief, phase=ProjectPhase.CREATED
        )
        self._projects.add(project)
        phase = ProjectPhase.CREATED

        # --- Import ---
        phase = self._advance(project_id, phase, ProjectPhase.INGESTING)
        with bench.stage("ingest"):
            ingest = self._importer.import_folder(project_id, source, dedup=dedup)

        # --- Plan ---
        phase = self._advance(project_id, phase, ProjectPhase.PLANNING)
        plan = self._planner.plan(project_id, ingest.imported)
        phase = self._advance(project_id, phase, ProjectPhase.PLAN_AWAITING_APPROVAL)
        phase = self._advance(project_id, phase, ProjectPhase.ACTIVE)  # auto-approve

        # --- Detect + Segment ---
        self._engine.label(plan, ingest.image_ids, bench)

        # --- Verify ---
        approved = needs_review = rejected = detections = 0
        with bench.stage("verification"):
            for image_id in ingest.image_ids:
                data = self._cas.get(self._images.get(image_id).sha256)
                for ann in self._annotations.by_image(image_id):
                    detections += 1
                    verdict = self._verifier.verify(data, ann)
                    target = self._apply_verdict(ann.id, verdict.verdict)
                    if target == AnnotationState.AUTO_ACCEPTED:
                        approved += 1
                    elif target == AnnotationState.NEEDS_REVIEW:
                        needs_review += 1
                    else:
                        rejected += 1

        # --- Export ---
        phase = self._advance(project_id, phase, ProjectPhase.AUDITING)
        with bench.stage("export"):
            export = self._exporter.run(project_id, export_format, dest)
        self._advance(project_id, phase, ProjectPhase.SNAPSHOT_READY)

        benchmark = bench.finish(
            images_processed=ingest.imported, num_batches=plan.num_batches
        )
        BenchmarkCollector.save(benchmark, self._benchmarks_dir)

        # --- Measurement layer (Phase-5): quality + errors + report + registry ---
        quality = self._analyzer.quality(project_id)
        errors = self._analyzer.errors(project_id)
        report = ExecutionReport(
            project_id=project_id,
            source=source,
            imported=ingest.imported,
            duplicates_skipped=ingest.duplicates_skipped,
            quarantined=ingest.quarantined,
            detections=detections,
            verified_approved=approved,
            needs_review=needs_review,
            rejected=rejected,
            export=export,
            benchmark=benchmark,
            quality=quality,
            errors=errors,
        )
        save_report(report, self._reports_dir)
        if self._registry is not None:
            self._registry.register(to_kpis(report, stage="deterministic"))
        return report

    def _advance(
        self, project_id: str, current: ProjectPhase, target: ProjectPhase
    ) -> ProjectPhase:
        guard_phase(current, target)
        self._projects.set_phase(project_id, target)
        return target

    def _apply_verdict(self, annotation_id: str, verdict: str) -> AnnotationState:
        """Map a verifier verdict onto the annotation state machine (auditable)."""
        if verdict == REJECTED:
            target = AnnotationState.REJECTED_AUTO
            assert_transition(AnnotationState.LABELED, target)
        elif verdict == APPROVED:
            assert_transition(AnnotationState.LABELED, AnnotationState.VERIFIED)
            target = AnnotationState.AUTO_ACCEPTED
            assert_transition(AnnotationState.VERIFIED, target)
        else:  # NEEDS_REVIEW
            assert_transition(AnnotationState.LABELED, AnnotationState.VERIFIED)
            target = AnnotationState.NEEDS_REVIEW
            assert_transition(AnnotationState.VERIFIED, target)
        self._annotations.set_state(annotation_id, target)
        return target
