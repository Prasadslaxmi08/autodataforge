"""Dependency wiring (System Design §2, §5).

A single composition root. No DI framework — the object graph is small and
explicit constructor wiring is the lazier, more debuggable choice. Each process
(API, worker, CLI) builds one Container at startup and reads its dependencies
from it.

Bootstrap scope: wires the pieces that exist — Settings, logging, GpuManager,
ModelRegistry. Repositories and services attach here in Phase 1 as they are
implemented.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

from vds.agents.analyst_agent import LLMAnalyst
from vds.agents.llm import LLMClient
from vds.agents.pipeline import Phase1Pipeline
from vds.agents.planner import ExecutionPlanner
from vds.agents.planner_agent import LLMPlanner
from vds.agents.providers.base import LLMProvider, load_provider
from vds.agents.verifier import RuleBasedVerifier
from vds.agents.vlm_verifier import LLMVerifier
from vds.comparison import ComparisonRegistry
from vds.config.settings import Settings, get_settings
from vds.engine.labeler import LabelingEngine
from vds.export.service import ExportService
from vds.ingest.service import ImportService
from vds.logging import configure, get_logger
from vds.memory import MEMORY_PATH, EngineeringMemoryService
from vds.models.gpu import GpuManager
from vds.models.registry import ModelRegistry
from vds.quality.metrics import QualityAnalyzer
from vds.store.cas import LocalCas
from vds.store.sqlite import (
    AnnotationRepo,
    Database,
    ImageRepo,
    ProjectRepo,
    SnapshotRepo,
)

log = get_logger(__name__)


class Container:
    """Composition root. Each dependency is a cached_property that depends only
    on those below it in the layer stack (System Design §2)."""

    def __init__(
        self,
        settings: Settings | None = None,
        db_path: str | None = None,
        artifacts_dir: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._db_path = db_path or "data/vds.db"
        self._artifacts_dir = artifacts_dir  # None => default benchmarks/ locations
        configure(level=self.settings.log_level, json=self.settings.log_json)
        log.info("container.init", environment=self.settings.environment)

    # --- L1 ---
    @cached_property
    def gpu(self) -> GpuManager:
        return GpuManager(vram_budget_mb=self.settings.gpu.vram_budget_mb)

    @cached_property
    def models(self) -> ModelRegistry:
        return ModelRegistry(selection=self.settings.models, gpu=self.gpu)

    @cached_property
    def cas(self) -> LocalCas:
        return LocalCas(self.settings.storage.cas_root)

    @cached_property
    def db(self) -> Database:
        return Database(self._db_path)

    @cached_property
    def projects(self) -> ProjectRepo:
        return ProjectRepo(self.db)

    @cached_property
    def images(self) -> ImageRepo:
        return ImageRepo(self.db)

    @cached_property
    def annotations(self) -> AnnotationRepo:
        return AnnotationRepo(self.db)

    @cached_property
    def snapshots(self) -> SnapshotRepo:
        return SnapshotRepo(self.db)

    # --- L2 services ---
    @cached_property
    def importer(self) -> ImportService:
        return ImportService(self.cas, self.images)

    @cached_property
    def engine(self) -> LabelingEngine:
        return LabelingEngine(self.models, self.cas, self.images, self.annotations)

    @cached_property
    def exporter(self) -> ExportService:
        return ExportService(self.images, self.annotations, self.cas)

    # --- Agent framework ---
    @cached_property
    def llm_provider(self) -> LLMProvider:
        return load_provider(self.settings.llm.provider, self.settings.llm)

    @cached_property
    def llm_client(self) -> LLMClient:
        return LLMClient(self.llm_provider, self.settings.llm)

    # --- L3 agents ---
    @cached_property
    def planner(self) -> ExecutionPlanner:
        return ExecutionPlanner(self.settings)

    @cached_property
    def memory(self) -> EngineeringMemoryService:
        path = (self._artifacts_dir / "engineering_memory.json"
                if self._artifacts_dir is not None else MEMORY_PATH)
        return EngineeringMemoryService(path)

    @cached_property
    def planner_agent(self) -> LLMPlanner:
        return LLMPlanner(self.llm_client, self.planner, self.memory)

    @cached_property
    def analyst_agent(self) -> LLMAnalyst:
        return LLMAnalyst(self.llm_client)

    @cached_property
    def verifier(self) -> RuleBasedVerifier:
        return RuleBasedVerifier()

    @cached_property
    def vlm_verifier(self) -> LLMVerifier:
        """Multimodal semantic verifier. Falls back to `verifier` when the
        configured provider can't do vision (e.g. the default Echo)."""
        return LLMVerifier(self.llm_client, self.verifier)

    @cached_property
    def analyzer(self) -> QualityAnalyzer:
        return QualityAnalyzer(self.images, self.annotations)

    @cached_property
    def registry(self) -> ComparisonRegistry:
        if self._artifacts_dir is not None:
            return ComparisonRegistry(self._artifacts_dir / "registry.json")
        return ComparisonRegistry()

    @cached_property
    def pipeline(self) -> Phase1Pipeline:
        from vds.benchmark import BENCHMARKS_DIR
        from vds.reporting import REPORTS_DIR

        if self._artifacts_dir:
            reports_dir = self._artifacts_dir / "reports"
            benchmarks_dir = self._artifacts_dir / "raw"
        else:
            reports_dir, benchmarks_dir = REPORTS_DIR, BENCHMARKS_DIR
        return Phase1Pipeline(
            projects=self.projects,
            images=self.images,
            annotations=self.annotations,
            cas=self.cas,
            importer=self.importer,
            planner=self.planner,
            engine=self.engine,
            verifier=self.verifier,
            exporter=self.exporter,
            analyzer=self.analyzer,
            registry=self.registry,
            reports_dir=reports_dir,
            benchmarks_dir=benchmarks_dir,
        )


def build_container(settings: Settings | None = None, db_path: str | None = None) -> Container:
    return Container(settings, db_path)
