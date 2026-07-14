"""L2 service — dataset intake (deterministic).

Responsibility: walk a source -> integrity check -> CAS write -> EXIF strip ->
perceptual-hash dedup -> blur/corrupt filter -> ImageRecord rows. Runs as a job.

Bootstrap scope: the service interface. Implementation (Pillow/imagehash logic)
is Phase 1 business logic.
"""

from __future__ import annotations

from typing import Protocol

from vds.core.contracts import JobId, ProjectId


class IngestService(Protocol):
    def start(self, project_id: ProjectId, source: str) -> JobId:
        """Enqueue ingestion of `source` into the project's corpus."""
        ...
