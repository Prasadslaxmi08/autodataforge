"""Goal — the single thing the GUI submits (V2-20 §GOALS).

Version 1 asked the user to drive a workflow (Import -> Annotate -> Review ->
Export). Version 2 asks only for a *goal* in natural language; the system decides
how. Intent parsing is a future phase — today a Goal carries the raw text plus any
structured params the UI already knows (a source path, a project name).
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class Goal(BaseModel):
    """A user objective, e.g. 'Create a vehicle detection dataset from this video.'

    Serializable: it is stored verbatim on the SessionState so a run is fully
    reconstructable from its persisted state.
    """

    id: str
    text: str
    params: dict = Field(default_factory=dict)


def new_goal(text: str, **params: object) -> Goal:
    """Mint a Goal with a fresh id. Known structured hints (source, name, ...)
    go in ``params``; the planner reads them, falling back to the text."""
    return Goal(id=uuid.uuid4().hex, text=text, params=dict(params))
