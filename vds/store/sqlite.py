"""SQLite-backed repositories (Phase 1).

ponytail: stdlib sqlite3 is the Phase-1 store — single-node, local-first, zero
external services, fully testable in CI. It implements the same repository
Protocols from `store.db`, so swapping to Postgres/SQLAlchemy for pgvector and
concurrent workers in Phase 2 changes only this file (the "database is the seam"
principle from the System Design). Contract objects serialize to JSON columns.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from vds.core.contracts import (
    Annotation,
    ImageRecord,
    Project,
    ProjectId,
    SnapshotManifest,
)
from vds.logging import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY, name TEXT, brief TEXT, phase TEXT
);
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY, project_id TEXT, sha256 TEXT,
    width INTEGER, height INTEGER, state TEXT, quarantine_reason TEXT,
    UNIQUE(project_id, sha256)
);
CREATE TABLE IF NOT EXISTS annotations (
    id TEXT PRIMARY KEY, image_id TEXT, state TEXT, data TEXT
);
CREATE TABLE IF NOT EXISTS snapshots (
    id TEXT PRIMARY KEY, project_id TEXT, data TEXT
);
CREATE TABLE IF NOT EXISTS agent_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, data TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, kind TEXT, payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_images_project ON images(project_id);
CREATE INDEX IF NOT EXISTS idx_ann_image ON annotations(image_id);
"""


class Database:
    """Owns the connection and schema. One per process; sqlite handles the rest."""

    def __init__(self, path: str | Path = "data/vds.db") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        log.info("db.ready", path=str(path))


class ProjectRepo:
    def __init__(self, db: Database) -> None:
        self._c = db.conn

    def add(self, project: Project) -> None:
        self._c.execute(
            "INSERT INTO projects VALUES (?,?,?,?)",
            (project.id, project.name, project.brief, project.phase),
        )
        self._c.commit()

    def get(self, project_id: ProjectId) -> Project | None:
        row = self._c.execute(
            "SELECT * FROM projects WHERE id=?", (project_id,)
        ).fetchone()
        return Project(**row) if row else None

    def list(self) -> list[Project]:
        rows = self._c.execute("SELECT * FROM projects").fetchall()
        return [Project(**r) for r in rows]

    def set_phase(self, project_id: ProjectId, phase: str) -> None:
        self._c.execute("UPDATE projects SET phase=? WHERE id=?", (phase, project_id))
        self._c.commit()

    def rename(self, project_id: ProjectId, name: str) -> None:
        self._c.execute("UPDATE projects SET name=? WHERE id=?", (name, project_id))
        self._c.commit()

    def delete(self, project_id: ProjectId) -> None:
        """Remove a project and its rows (images, annotations). CAS blobs are
        content-addressed and shared, so they are intentionally left in place."""
        # ponytail: leave shared CAS blobs; a GC sweep can reclaim orphans later.
        img_ids = [
            r["id"]
            for r in self._c.execute(
                "SELECT id FROM images WHERE project_id=?", (project_id,)
            ).fetchall()
        ]
        self._c.executemany(
            "DELETE FROM annotations WHERE image_id=?", [(i,) for i in img_ids]
        )
        self._c.execute("DELETE FROM images WHERE project_id=?", (project_id,))
        self._c.execute("DELETE FROM projects WHERE id=?", (project_id,))
        self._c.commit()


class ImageRepo:
    def __init__(self, db: Database) -> None:
        self._c = db.conn

    def add(self, image: ImageRecord) -> bool:
        """Returns False if a same-content image already exists (dedup)."""
        try:
            self._c.execute(
                "INSERT INTO images VALUES (?,?,?,?,?,?,?)",
                (
                    image.id,
                    image.project_id,
                    image.sha256,
                    image.width,
                    image.height,
                    image.state,
                    image.quarantine_reason,
                ),
            )
            self._c.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # duplicate (project_id, sha256)

    def get(self, image_id: str) -> ImageRecord | None:
        row = self._c.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
        return ImageRecord(**row) if row else None

    def by_project(self, project_id: ProjectId) -> list[ImageRecord]:
        rows = self._c.execute(
            "SELECT * FROM images WHERE project_id=? ORDER BY id", (project_id,)
        ).fetchall()
        return [ImageRecord(**r) for r in rows]


class AnnotationRepo:
    def __init__(self, db: Database) -> None:
        self._c = db.conn

    def add(self, annotation: Annotation) -> None:
        self._c.execute(
            "INSERT INTO annotations VALUES (?,?,?,?)",
            (annotation.id, annotation.image_id, annotation.state, annotation.model_dump_json()),
        )
        self._c.commit()

    def by_image(self, image_id: str) -> list[Annotation]:
        rows = self._c.execute(
            "SELECT data FROM annotations WHERE image_id=?", (image_id,)
        ).fetchall()
        return [Annotation(**json.loads(r["data"])) for r in rows]

    def get(self, annotation_id: str) -> Annotation | None:
        row = self._c.execute(
            "SELECT data FROM annotations WHERE id=?", (annotation_id,)
        ).fetchone()
        return Annotation(**json.loads(row["data"])) if row else None

    def set_state(self, annotation_id: str, state: str) -> None:
        row = self._c.execute(
            "SELECT data FROM annotations WHERE id=?", (annotation_id,)
        ).fetchone()
        if row is None:
            return
        data = json.loads(row["data"])
        data["state"] = state
        self._c.execute(
            "UPDATE annotations SET state=?, data=? WHERE id=?",
            (state, json.dumps(data), annotation_id),
        )
        self._c.commit()


class SnapshotRepo:
    def __init__(self, db: Database) -> None:
        self._c = db.conn

    def add(self, manifest: SnapshotManifest) -> None:
        self._c.execute(
            "INSERT INTO snapshots VALUES (?,?,?)",
            (manifest.id, manifest.project_id, manifest.model_dump_json()),
        )
        self._c.commit()

    def get(self, snapshot_id: str) -> SnapshotManifest | None:
        row = self._c.execute(
            "SELECT data FROM snapshots WHERE id=?", (snapshot_id,)
        ).fetchone()
        return SnapshotManifest(**json.loads(row["data"])) if row else None
