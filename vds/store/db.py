"""Persistence repositories (System Design §2.2).

Repositories are the typed interface between the domain (contract objects) and
SQL. Modules above L1 speak contract types; only these classes know the schema.

Bootstrap scope: the repository *protocols* are defined so services can be typed
and wired against them now. The SQLAlchemy-backed implementations (schema,
Alembic migrations, pgvector queries) land in Phase 1 — they are persistence
business logic, deliberately out of the bootstrap.
"""

from __future__ import annotations

from typing import Protocol

from vds.core.contracts import (
    Annotation,
    ImageRecord,
    Project,
    ProjectId,
    SnapshotManifest,
)


class ProjectRepo(Protocol):
    def add(self, project: Project) -> None: ...
    def get(self, project_id: ProjectId) -> Project | None: ...
    def list(self) -> list[Project]: ...
    def set_phase(self, project_id: ProjectId, phase: str) -> None: ...


class ImageRepo(Protocol):
    def add(self, image: ImageRecord) -> None: ...
    def get(self, image_id: str) -> ImageRecord | None: ...
    def by_project(self, project_id: ProjectId) -> list[ImageRecord]: ...


class AnnotationRepo(Protocol):
    def add(self, annotation: Annotation) -> None: ...
    def by_image(self, image_id: str) -> list[Annotation]: ...
    def set_state(self, annotation_id: str, state: str) -> None: ...


class AgentLogRepo(Protocol):
    """Persists every agent LLM/VLM call (FR-7): prompt, output, model, latency, cost."""

    def record(self, entry: dict) -> None: ...
    def by_project(self, project_id: ProjectId) -> list[dict]: ...


class SnapshotRepo(Protocol):
    def add(self, manifest: SnapshotManifest) -> None: ...
    def get(self, snapshot_id: str) -> SnapshotManifest | None: ...


class EventRepo(Protocol):
    """Append-only pipeline events, relayed to the UI over WebSocket (§4)."""

    def append(self, project_id: ProjectId, kind: str, payload: dict) -> None: ...
    def since(self, project_id: ProjectId, after_id: int) -> list[dict]: ...
