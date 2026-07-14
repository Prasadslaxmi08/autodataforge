"""Goal parser (V2-21 §GOAL PARSER) — deterministic, keyword-based intent.

Turns a free-text Goal into a ``ParsedGoal``: task type, input modality, domain
hints (thermal / drone), target classes, and export format. No LLM — a senior
engineer reads the words and classifies. Same text -> same parse.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from vds.v2.goal import Goal
from vds.v2.planner import TaskType

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Small, extensible keyword tables. Order matters only for MIXED detection.
_SEG_WORDS = ("segmentation", "segment", "mask", "instance seg")
_DET_WORDS = ("detection", "detect", "bounding box", "bbox")
_CLS_WORDS = ("classification", "classify", "label images")
_REVIEW_WORDS = ("improve", "review", "quality", "clean up", "fix annotations", "re-annotate")
_EXPORT_WORDS = ("export", "convert to", "to coco", "to yolo", "to voc")
_VIDEO_WORDS = ("video", "footage", "clip", "mp4", "recording")
_EXISTING_WORDS = ("existing", "this dataset", "current dataset", "improve this")
_THERMAL_WORDS = ("thermal", "infrared", "ir ", "lwir", "eo/ir", "eo ir")
_DRONE_WORDS = ("drone", "aerial", "uav", "overhead")

# Known object nouns we can pull out as target classes; falls back to the noun
# sitting before "detection"/"dataset".
_CLASS_WORDS = (
    "vehicle", "car", "truck", "person", "people", "pedestrian", "face",
    "animal", "aircraft", "ship", "boat", "building",
)  # note: "drone" is a platform hint, not a target class
_EXPORT_FORMATS = ("coco", "yolo", "voc")


class ParsedGoal(BaseModel):
    goal_id: str
    text: str
    task_type: TaskType
    modality: str  # "video" | "images" | "existing" | "unknown"
    thermal: bool = False
    drone: bool = False
    export_format: str | None = None
    target_classes: list[str] = Field(default_factory=list)
    source: str | None = None


def _has(text: str, words: tuple[str, ...]) -> bool:
    return any(w in text for w in words)


def _classify(text: str) -> TaskType:
    hits = {
        TaskType.SEGMENTATION: _has(text, _SEG_WORDS),
        TaskType.DETECTION: _has(text, _DET_WORDS),
        TaskType.CLASSIFICATION: _has(text, _CLS_WORDS),
        TaskType.REVIEW: _has(text, _REVIEW_WORDS),
        TaskType.EXPORT: _has(text, _EXPORT_WORDS),
    }
    active = [t for t, on in hits.items() if on]
    if not active:
        return TaskType.UNKNOWN
    # Export/review are terminal intents on their own; multiple *build* intents = MIXED.
    build = [t for t in active if t in (TaskType.DETECTION, TaskType.SEGMENTATION, TaskType.CLASSIFICATION)]
    if len(build) > 1:
        return TaskType.MIXED
    if len(active) == 1:
        return active[0]
    # e.g. "detection" + "export" -> the build intent wins (export is a trailing step).
    return build[0] if build else active[0]


def _target_classes(text: str) -> list[str]:
    found = [w for w in _CLASS_WORDS if re.search(rf"\b{re.escape(w)}\b", text)]
    if found:
        # normalise plurals we listed explicitly
        return sorted({"person" if w in ("people", "pedestrian") else w for w in found})
    # fallback: the word before "detection" or "dataset"
    m = re.search(r"\b(\w+)\s+(?:detection|segmentation|dataset)\b", text)
    return [m.group(1)] if m and m.group(1) not in ("this", "the", "a", "an") else []


def _source_and_modality(text: str, params: dict) -> tuple[str | None, str]:
    source = params.get("source") or params.get("path")
    if isinstance(source, str) and Path(source).suffix.lower() in _VIDEO_EXTS:
        return source, "video"
    if _has(text, _EXISTING_WORDS) or params.get("project") or params.get("dataset"):
        return source, "existing"
    if _has(text, _VIDEO_WORDS):
        return source, "video"
    if source:
        return source, "images"
    return None, "unknown"


class GoalParser:
    def parse(self, goal: Goal) -> ParsedGoal:
        text = goal.text.lower()
        source, modality = _source_and_modality(text, goal.params)
        fmt = next((f for f in _EXPORT_FORMATS if f in text), None)
        return ParsedGoal(
            goal_id=goal.id,
            text=goal.text,
            task_type=_classify(text),
            modality=modality,
            thermal=_has(text, _THERMAL_WORDS),
            drone=_has(text, _DRONE_WORDS),
            export_format=fmt,
            target_classes=_target_classes(text),
            source=source,
        )
