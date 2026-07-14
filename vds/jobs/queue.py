"""Job queue framework (System Design §2.3, amendment 1).

The queue is Postgres, not Redis: GPU/corpus work must be async, durable, and
resumable, and keeping jobs in the same transactional store as annotations means
a crash can never strand a job the DB thinks is done. Handlers process work in
batches and checkpoint after each, so resume never repeats completed batches
(NFR-4).

Bootstrap scope: the handler-registry and the queue *interface* are real and
importable. The `SELECT ... FOR UPDATE SKIP LOCKED` claim loop and heartbeat
reclaim are Postgres-backed and land with `store.db` in Phase 1.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from vds.core.contracts import JobId
from vds.logging import get_logger

log = get_logger(__name__)

# A handler receives the job payload and a checkpoint callback it invokes after
# each completed batch. It is expected to be idempotent w.r.t. the checkpoint.
JobHandler = Callable[[dict[str, Any], "Checkpointer"], None]


class Checkpointer(Protocol):
    def save(self, checkpoint: dict[str, Any]) -> None: ...


class Queue(Protocol):
    def enqueue(self, job_type: str, payload: dict[str, Any]) -> JobId: ...
    def cancel(self, job_id: JobId) -> None: ...
    def status(self, job_id: JobId) -> str: ...


class HandlerRegistry:
    """Maps a job type string to its handler. Populated by `@job_handler`."""

    def __init__(self) -> None:
        self._handlers: dict[str, JobHandler] = {}

    def register(self, job_type: str, handler: JobHandler) -> None:
        if job_type in self._handlers:
            raise ValueError(f"duplicate job handler for {job_type!r}")
        self._handlers[job_type] = handler
        log.info("jobs.register_handler", job_type=job_type)

    def get(self, job_type: str) -> JobHandler:
        if job_type not in self._handlers:
            raise KeyError(f"no handler registered for job type {job_type!r}")
        return self._handlers[job_type]

    def types(self) -> list[str]:
        return sorted(self._handlers)


# Process-wide registry. Services decorate their handlers at import time.
registry = HandlerRegistry()


def job_handler(job_type: str) -> Callable[[JobHandler], JobHandler]:
    def decorator(fn: JobHandler) -> JobHandler:
        registry.register(job_type, fn)
        return fn

    return decorator
