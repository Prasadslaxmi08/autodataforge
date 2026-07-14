"""L0 — domain contracts, state enums, and the error taxonomy.

`core` is imported everywhere and depends on nothing but pydantic. It is the
shared vocabulary that keeps modules decoupled.
"""

from vds.core import contracts, enums, errors

__all__ = ["contracts", "enums", "errors"]
