"""Engineering Memory storage (Phase 10).

Append-only, versioned, deterministic. One JSON file, one record per row — the
same "a table in a file" choice the comparison registry already makes (no vector
DB, per the phase rules). Complete history is retained: re-recording a dataset
family appends a new version; nothing is overwritten. Corrupt input never crashes
a caller — a bad file is quarantined and treated as empty; a bad row is skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from vds.logging import get_logger
from vds.memory.schema import EngineeringMemory

log = get_logger(__name__)

MEMORY_PATH = Path("data/engineering_memory.json")


class MemoryStore:
    def __init__(self, path: Path = MEMORY_PATH) -> None:
        self._path = path

    # --- read ---
    def all(self) -> list[EngineeringMemory]:
        """Every record, oldest first. Tolerant of corruption (phase: 'corrupted
        memory'): an unparseable file is quarantined and treated as empty; an
        invalid row is skipped, not fatal."""
        if not self._path.exists():
            return []
        try:
            rows = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError("memory file is not a JSON array")
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            self._quarantine(exc)
            return []
        out: list[EngineeringMemory] = []
        for row in rows:
            try:
                out.append(EngineeringMemory.model_validate(row))
            except ValidationError as exc:
                log.warning("memory.skip_invalid_row", error=str(exc))
        out.sort(key=lambda m: (m.created_at, m.version))
        return out

    def get(self, memory_id: str) -> EngineeringMemory | None:
        return next((m for m in self.all() if m.id == memory_id), None)

    def family(self, fingerprint_hash: str) -> list[EngineeringMemory]:
        """All versions recorded for one dataset fingerprint family, oldest first."""
        return [m for m in self.all() if m.dataset_fingerprint.hash() == fingerprint_hash]

    # --- write (append-only) ---
    def add(self, memory: EngineeringMemory) -> EngineeringMemory:
        """Append a record. Duplicate suppression: an identical record (same
        content hash) is not stored twice — the existing one is returned. Otherwise
        the record's version is set to the next in its fingerprint family and it is
        appended (history preserved, never overwritten)."""
        existing = self.all()
        dup = next((m for m in existing if m.content_hash() == memory.content_hash()), None)
        if dup is not None:
            log.info("memory.duplicate_ignored", id=dup.id)
            return dup

        fam = [m for m in existing if m.dataset_fingerprint.hash() == memory.dataset_fingerprint.hash()]
        memory.version = (max((m.version for m in fam), default=0)) + 1
        existing.append(memory)
        self._persist(existing)
        log.info("memory.added", id=memory.id, version=memory.version,
                 fingerprint=memory.dataset_fingerprint.hash())
        return memory

    def _persist(self, records: list[EngineeringMemory]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps([m.model_dump() for m in records], indent=2), encoding="utf-8")
        tmp.replace(self._path)  # atomic-ish: a crash mid-write can't corrupt the live file

    def _quarantine(self, exc: Exception) -> None:
        bad = self._path.with_suffix(".corrupt")
        try:
            self._path.replace(bad)
            log.warning("memory.quarantined_corrupt_file", moved_to=str(bad), error=str(exc))
        except OSError:
            log.warning("memory.corrupt_file_unreadable", error=str(exc))
