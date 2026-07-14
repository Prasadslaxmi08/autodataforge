"""L4 — Python SDK (System Design §6).

The SDK *is* the service layer: the CLI and any user notebook script the pipeline
through the same in-process functions, so there is no second API to design. An
HTTP-backed twin with the identical surface can wrap this later.

Bootstrap scope: the client holds the container; pipeline methods attach in
Phase 1 as services are implemented.
"""

from __future__ import annotations

from vds.container import Container, build_container


class Client:
    def __init__(self, container: Container | None = None) -> None:
        self._container = container or build_container()

    @property
    def container(self) -> Container:
        return self._container

    # Phase 1: ingest(), plan(), label(), review(), snapshot(), export() —
    # thin pass-throughs to the wired services.


__all__ = ["Client"]
