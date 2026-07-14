"""Content-addressed store (System Design §2.2).

Write-once, atomic blob storage keyed by sha256. Content-addressing gives dedup
and immutability for free. This 4-method protocol is the *only* place in the
system that touches storage paths — it is the single seam swapped for S3 in
cloud deployment (§10). Nothing above L1 constructs a path.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Protocol

from vds.core.contracts import Sha256
from vds.core.errors import IntegrityError


class Cas(Protocol):
    def put(self, data: bytes) -> Sha256: ...
    def get(self, sha: Sha256) -> bytes: ...
    def path(self, sha: Sha256) -> Path: ...
    def exists(self, sha: Sha256) -> bool: ...


def sha256_of(data: bytes) -> Sha256:
    return hashlib.sha256(data).hexdigest()


class LocalCas:
    """Filesystem CAS: blobs at `<root>/ab/cd/<sha>` (sharded by hash prefix)."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def path(self, sha: Sha256) -> Path:
        return self._root / sha[:2] / sha[2:4] / sha

    def exists(self, sha: Sha256) -> bool:
        return self.path(sha).exists()

    def put(self, data: bytes) -> Sha256:
        sha = sha256_of(data)
        dest = self.path(sha)
        if dest.exists():  # write-once: identical content already stored
            return sha
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: write to a temp file in the same dir, then rename.
        fd, tmp = tempfile.mkstemp(dir=dest.parent)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, dest)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return sha

    def get(self, sha: Sha256) -> bytes:
        data = self.path(sha).read_bytes()
        if sha256_of(data) != sha:
            raise IntegrityError(f"CAS hash mismatch for {sha}")
        return data
