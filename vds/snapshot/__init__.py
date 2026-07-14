"""L2 service — dataset versioning (deterministic; native manifests, amendment 2).

Responsibility: immutable SnapshotManifest = frozen set of (image sha,
annotation-set hash, plan version, split) + lineage. create / diff / verify.
Same manifest -> byte-identical export (FR-5).

Bootstrap scope: the service interface. Manifest logic is Phase 1.
"""

from __future__ import annotations

from typing import Protocol

from vds.core.contracts import ProjectId, SnapshotId, SnapshotManifest


class SnapshotService(Protocol):
    def create(self, project_id: ProjectId) -> SnapshotManifest: ...
    def diff(self, a: SnapshotId, b: SnapshotId) -> dict: ...
    def verify(self, snapshot_id: SnapshotId) -> bool: ...
