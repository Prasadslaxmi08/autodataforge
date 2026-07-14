"""Error taxonomy (System Design §8).

Each class carries exactly one recovery policy, decided by the layer that
catches it (the job boundary). The class hierarchy *is* the policy table:
catch the base you want to handle and the policy follows.

    VDSError
    ├── TransientError    retry w/ backoff, then DEAD
    ├── ResourceError     halve batch + retry, floor -> quarantine
    ├── DataError         quarantine the item, continue the batch
    ├── AgentOutputError  bounded re-prompt, then dead-letter + safe fallback
    ├── IntegrityError    fail loud, halt the stage (this is corruption)
    │   └── IllegalTransitionError
    ├── ExportError       fail the export, delete partial output
    └── ConfigError       fail at startup, never mid-run
"""

from __future__ import annotations


class VDSError(Exception):
    """Base for every error the platform raises deliberately."""


class TransientError(VDSError):
    """A momentary failure worth retrying (model crash, LLM 5xx/timeout)."""


class ResourceError(VDSError):
    """A resource ceiling was hit (GPU OOM). Recovered by shrinking the batch."""


class DataError(VDSError):
    """A single input item is bad (corrupt image, EXIF bomb). Quarantine it."""


class AgentOutputError(VDSError):
    """An agent returned schema-invalid or nonsensical output."""


class IntegrityError(VDSError):
    """A consistency invariant was violated (hash mismatch, bad manifest).

    Not weather — corruption. Halt the affected stage rather than continue.
    """


class IllegalTransitionError(IntegrityError):
    """An attempt to move a state machine along a disallowed edge."""


class ExportError(VDSError):
    """An export failed validation. Partial output must be discarded."""


class ConfigError(VDSError):
    """Configuration is invalid. Raised at startup, never during a run."""
