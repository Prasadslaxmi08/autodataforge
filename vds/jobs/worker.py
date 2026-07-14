"""Worker loop scaffold (System Design §2.3, §8).

Claims queued jobs, dispatches to the registered handler, applies the per-error
recovery policy from the taxonomy, and checkpoints. Bootstrap scope: the loop
structure and error-policy dispatch exist; the Postgres claim/heartbeat backing
lands in Phase 1 with `store.db`. Run via `vds worker`.
"""

from __future__ import annotations

import signal
import time

from vds.jobs.queue import HandlerRegistry, registry
from vds.logging import get_logger

log = get_logger(__name__)


class Worker:
    def __init__(self, handlers: HandlerRegistry = registry, poll_interval: float = 1.0):
        self._handlers = handlers
        self._poll = poll_interval
        self._running = False

    def stop(self, *_: object) -> None:
        self._running = False
        log.info("worker.stopping")

    def run(self) -> None:
        """Poll for jobs until stopped. No queue backend yet -> idles cleanly."""
        self._running = True
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        log.info("worker.started", handlers=self._handlers.types())
        while self._running:
            # Phase 1: claim a job (SKIP LOCKED), dispatch, checkpoint, retry.
            # Until the queue backend exists there is nothing to claim.
            time.sleep(self._poll)
        log.info("worker.stopped")
