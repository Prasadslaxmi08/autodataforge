"""L1 — persistence. `db` (repositories) and `cas` (content-addressed blobs).
The database is the only shared state in the system (System Design §1)."""

from vds.store.cas import Cas, LocalCas, sha256_of

__all__ = ["Cas", "LocalCas", "sha256_of"]
