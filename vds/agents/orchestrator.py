"""Orchestrator (System Design §2.9, §7) — the typed state machine.

Deterministic code that advances project phases, enqueues jobs, and routes
escalations (low yield -> Planner; feedback -> Analyst -> Planner). Only its
*nodes* call agents; the control flow itself is explicit and auditable — which is
the whole point of not using an internal event bus (§4).

Bootstrap scope: the interface plus the phase-transition guard, which is real
today (it delegates to `core.enums.assert_transition`).
"""

from __future__ import annotations

from typing import Protocol

from vds.core.contracts import ProjectId
from vds.core.enums import ProjectPhase, assert_transition
from vds.logging import get_logger

log = get_logger(__name__)


class Orchestrator(Protocol):
    def advance(self, project_id: ProjectId, trigger: str) -> None:
        """Evaluate the project's state and take the next pipeline action."""
        ...


def guard_phase(current: ProjectPhase, target: ProjectPhase) -> None:
    """Refuse an illegal project-phase transition loudly (raises)."""
    assert_transition(current, target)
    log.info("orchestrator.transition", **{"from": current.value, "to": target.value})
