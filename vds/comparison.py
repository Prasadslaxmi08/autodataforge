"""Comparison framework (Phase-5).

A registry of StageKPIs so future pipeline generations — Planner, Analyst, the
human-feedback loop, the final production pipeline — can each drop a labelled KPI
record and be diffed against the deterministic baseline. It exists now, with only
the baseline registered, so every future improvement is measured, not asserted.

Storage is a single JSON file (one row per stage); the pipeline shape is small
and single-node, so a table in a file is the right amount of machinery.
"""

from __future__ import annotations

import json
from pathlib import Path

from vds.core.contracts import StageKPIs

REGISTRY_PATH = Path("benchmarks/registry.json")

# The ordered pipeline generations the framework anticipates (System Design
# roadmap). Comparisons are rendered in this order.
STAGE_ORDER = ["deterministic", "planner", "analyst", "feedback", "production"]


class ComparisonRegistry:
    def __init__(self, path: Path = REGISTRY_PATH) -> None:
        self._path = path

    def load(self) -> list[StageKPIs]:
        if not self._path.exists():
            return []
        return [
            StageKPIs(**row)
            for row in json.loads(self._path.read_text(encoding="utf-8"))
        ]

    def register(self, kpis: StageKPIs) -> None:
        """Insert or replace the record for `kpis.stage` (latest wins)."""
        rows = {k.stage: k for k in self.load()}
        rows[kpis.stage] = kpis
        ordered = sorted(
            rows.values(),
            key=lambda k: STAGE_ORDER.index(k.stage) if k.stage in STAGE_ORDER else 99,
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([k.model_dump() for k in ordered], indent=2), encoding="utf-8"
        )

    def compare(self, base_stage: str, candidate_stage: str) -> dict[str, dict]:
        """Per-KPI {base, candidate, delta} between two registered stages."""
        rows = {k.stage: k for k in self.load()}
        if base_stage not in rows or candidate_stage not in rows:
            raise KeyError("both stages must be registered before comparing")
        base, cand = rows[base_stage], rows[candidate_stage]
        kpis = [
            "images_per_second", "approval_rate", "review_rate", "rejection_rate",
            "avg_confidence", "annotation_density", "peak_ram_mb",
            "invalid_annotations", "empty_masks",
        ]
        out = {}
        for name in kpis:
            b, c = getattr(base, name), getattr(cand, name)
            out[name] = {"base": b, "candidate": c, "delta": round(c - b, 4)}
        return out

    def render_table(self) -> str:
        """Markdown KPI-by-stage table across everything registered."""
        rows = self.load()
        if not rows:
            return "_no stages registered yet_"
        kpis = [
            "images_per_second", "approval_rate", "review_rate", "rejection_rate",
            "avg_confidence", "annotation_density", "peak_ram_mb",
        ]
        header = "| KPI | " + " | ".join(r.stage for r in rows) + " |"
        sep = "|" + "---|" * (len(rows) + 1)
        lines = [header, sep]
        for name in kpis:
            cells = " | ".join(str(getattr(r, name)) for r in rows)
            lines.append(f"| {name} | {cells} |")
        return "\n".join(lines)
