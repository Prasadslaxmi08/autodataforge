"""State enums and their legal-transition tables (System Design §7).

The transition tables are the single source of truth for the pipeline's state
machine. The orchestrator (Phase 3) consults `assert_transition` before moving
any row; an illegal transition is a bug, raised loudly, never silently allowed.
"""

from __future__ import annotations

from enum import StrEnum

from vds.core.errors import IllegalTransitionError


class ImageState(StrEnum):
    REGISTERED = "registered"
    INGESTED = "ingested"
    QUARANTINED = "quarantined"
    EMBEDDED = "embedded"
    LABELED = "labeled"


class AnnotationState(StrEnum):
    LABELED = "labeled"
    REJECTED_AUTO = "rejected_auto"
    VERIFIED = "verified"
    AUTO_ACCEPTED = "auto_accepted"
    NEEDS_REVIEW = "needs_review"
    ACCEPTED = "accepted"
    FIXED = "fixed"
    REJECTED = "rejected"


class JobState(StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD = "dead"
    CANCELLED = "cancelled"


class ProjectPhase(StrEnum):
    CREATED = "created"
    INGESTING = "ingesting"
    PLANNING = "planning"
    PLAN_AWAITING_APPROVAL = "plan_awaiting_approval"
    ACTIVE = "active"
    AUDITING = "auditing"
    SNAPSHOT_READY = "snapshot_ready"


# Legal transitions. A frozenset of allowed target states per source state.
# Terminal states map to an empty set.
_IMAGE_TRANSITIONS: dict[ImageState, frozenset[ImageState]] = {
    ImageState.REGISTERED: frozenset({ImageState.INGESTED, ImageState.QUARANTINED}),
    ImageState.INGESTED: frozenset({ImageState.EMBEDDED, ImageState.QUARANTINED}),
    ImageState.EMBEDDED: frozenset({ImageState.LABELED}),
    ImageState.LABELED: frozenset(),
    ImageState.QUARANTINED: frozenset(),
}

_ANNOTATION_TRANSITIONS: dict[AnnotationState, frozenset[AnnotationState]] = {
    AnnotationState.LABELED: frozenset(
        {AnnotationState.REJECTED_AUTO, AnnotationState.VERIFIED}
    ),
    AnnotationState.VERIFIED: frozenset(
        {AnnotationState.AUTO_ACCEPTED, AnnotationState.NEEDS_REVIEW}
    ),
    # Audit may re-open an auto-accepted annotation for review.
    AnnotationState.AUTO_ACCEPTED: frozenset({AnnotationState.NEEDS_REVIEW}),
    AnnotationState.NEEDS_REVIEW: frozenset(
        {AnnotationState.ACCEPTED, AnnotationState.FIXED, AnnotationState.REJECTED}
    ),
    AnnotationState.REJECTED_AUTO: frozenset(),
    AnnotationState.ACCEPTED: frozenset(),
    AnnotationState.FIXED: frozenset(),
    AnnotationState.REJECTED: frozenset(),
}

_JOB_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.QUEUED: frozenset({JobState.CLAIMED, JobState.CANCELLED}),
    JobState.CLAIMED: frozenset({JobState.RUNNING, JobState.QUEUED, JobState.CANCELLED}),
    JobState.RUNNING: frozenset(
        {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}
    ),
    # A retryable failure returns to the queue; exhausted retries go DEAD.
    JobState.FAILED: frozenset({JobState.QUEUED, JobState.DEAD}),
    JobState.SUCCEEDED: frozenset(),
    JobState.DEAD: frozenset(),
    JobState.CANCELLED: frozenset(),
}

_PROJECT_TRANSITIONS: dict[ProjectPhase, frozenset[ProjectPhase]] = {
    ProjectPhase.CREATED: frozenset({ProjectPhase.INGESTING}),
    ProjectPhase.INGESTING: frozenset({ProjectPhase.PLANNING}),
    ProjectPhase.PLANNING: frozenset({ProjectPhase.PLAN_AWAITING_APPROVAL}),
    ProjectPhase.PLAN_AWAITING_APPROVAL: frozenset(
        {ProjectPhase.ACTIVE, ProjectPhase.PLANNING}
    ),
    # ACTIVE re-enters itself on plan revision / added data (System Design §7).
    ProjectPhase.ACTIVE: frozenset({ProjectPhase.AUDITING, ProjectPhase.PLANNING}),
    ProjectPhase.AUDITING: frozenset(
        {ProjectPhase.SNAPSHOT_READY, ProjectPhase.ACTIVE}
    ),
    ProjectPhase.SNAPSHOT_READY: frozenset({ProjectPhase.ACTIVE}),
}

_TRANSITIONS: dict[type, dict] = {
    ImageState: _IMAGE_TRANSITIONS,
    AnnotationState: _ANNOTATION_TRANSITIONS,
    JobState: _JOB_TRANSITIONS,
    ProjectPhase: _PROJECT_TRANSITIONS,
}


def is_legal_transition(current, target) -> bool:
    """Return True if `current -> target` is an allowed transition."""
    table = _TRANSITIONS[type(current)]
    return target in table[current]


def assert_transition(current, target) -> None:
    """Raise IllegalTransitionError unless `current -> target` is allowed."""
    if not is_legal_transition(current, target):
        raise IllegalTransitionError(
            f"{type(current).__name__}: {current} -> {target} is not permitted"
        )
