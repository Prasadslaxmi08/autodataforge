"""Dataset import service (Phase 1).

Walks a folder, validates image formats, strips EXIF (by re-encoding to PNG),
stores each unique image in the CAS, deduplicates (exact by content hash, near
by average-hash), and writes ImageRecord metadata to the database.
"""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from vds.core.contracts import ImageRecord, ProjectId
from vds.core.enums import ImageState
from vds.logging import get_logger
from vds.store.cas import Cas
from vds.store.sqlite import ImageRepo

log = get_logger(__name__)

_SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_AHASH_HAMMING_DUP = 4  # <= this many differing bits => treat as near-duplicate


@dataclass
class IngestResult:
    imported: int = 0
    duplicates_skipped: int = 0
    quarantined: int = 0
    image_ids: list[str] = field(default_factory=list)


def _average_hash(img: Image.Image) -> int:
    """64-bit aHash: is each 8x8 pixel above the mean?"""
    small = np.asarray(img.convert("L").resize((8, 8)), dtype=np.float64)
    bits = (small > small.mean()).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


class ImportService:
    def __init__(self, cas: Cas, images: ImageRepo) -> None:
        self._cas = cas
        self._images = images

    def import_folder(self, project_id: ProjectId, source: str, *, dedup: bool = True) -> IngestResult:
        root = Path(source)
        if not root.exists():
            raise FileNotFoundError(f"import source not found: {source}")

        result = IngestResult()
        seen_hashes: list[int] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _SUPPORTED:
                continue
            try:
                img = Image.open(path)
                img.load()
                img = img.convert("RGB")
            except Exception as exc:  # corrupt / undecodable -> quarantine the item
                result.quarantined += 1
                log.warning("ingest.quarantine", file=str(path), reason=str(exc))
                continue

            ahash = _average_hash(img)
            if dedup and any(bin(ahash ^ h).count("1") <= _AHASH_HAMMING_DUP for h in seen_hashes):
                result.duplicates_skipped += 1
                continue

            # EXIF-stripped, normalized bytes (re-encoded PNG).
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            sha = self._cas.put(buf.getvalue())

            record = ImageRecord(
                id=uuid.uuid4().hex,
                project_id=project_id,
                sha256=sha,
                width=img.width,
                height=img.height,
                state=ImageState.INGESTED,
            )
            if not self._images.add(record):  # exact-content duplicate in DB
                result.duplicates_skipped += 1
                continue

            seen_hashes.append(ahash)
            result.imported += 1
            result.image_ids.append(record.id)

        log.info(
            "ingest.done",
            project_id=project_id,
            imported=result.imported,
            duplicates=result.duplicates_skipped,
            quarantined=result.quarantined,
        )
        return result
