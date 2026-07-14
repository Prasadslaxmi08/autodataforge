"""Logging framework (System Design §8).

structlog -> JSON on stdout (container-native). Every line carries the bound
context (project_id, job_id, trace_id) so a single grep reconstructs a run.
Call `configure()` once at process start; use `get_logger(__name__)` elsewhere.
"""

from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def configure(level: str = "INFO", json: bool = True) -> None:
    """Configure structlog + stdlib logging. Idempotent."""
    global _configured
    if _configured:
        return

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    renderer = (
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def bind(**kwargs: object) -> None:
    """Bind context vars onto every subsequent log line in this task."""
    structlog.contextvars.bind_contextvars(**kwargs)
