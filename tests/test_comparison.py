"""Unit tests for the comparison framework and report generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from vds.comparison import ComparisonRegistry
from vds.core.contracts import StageKPIs


def _kpis(stage: str, ips: float, approval: float) -> StageKPIs:
    return StageKPIs(
        stage=stage, label=f"{stage}", created_at="2026-01-01T00:00:00Z",
        images_per_second=ips, approval_rate=approval, review_rate=0.2,
        rejection_rate=0.1, avg_confidence=0.8, annotation_density=2.0,
        peak_ram_mb=50.0, invalid_annotations=0, empty_masks=0,
    )


def test_register_and_load(tmp_path: Path):
    reg = ComparisonRegistry(tmp_path / "registry.json")
    reg.register(_kpis("deterministic", 60.0, 0.7))
    rows = reg.load()
    assert len(rows) == 1 and rows[0].stage == "deterministic"


def test_register_replaces_same_stage(tmp_path: Path):
    reg = ComparisonRegistry(tmp_path / "registry.json")
    reg.register(_kpis("deterministic", 60.0, 0.7))
    reg.register(_kpis("deterministic", 90.0, 0.75))  # newer run wins
    rows = reg.load()
    assert len(rows) == 1 and rows[0].images_per_second == 90.0


def test_compare_deltas(tmp_path: Path):
    reg = ComparisonRegistry(tmp_path / "registry.json")
    reg.register(_kpis("deterministic", 60.0, 0.70))
    reg.register(_kpis("analyst", 55.0, 0.85))
    cmp = reg.compare("deterministic", "analyst")
    assert cmp["approval_rate"]["delta"] == pytest.approx(0.15)
    assert cmp["images_per_second"]["delta"] == pytest.approx(-5.0)


def test_compare_requires_both(tmp_path: Path):
    reg = ComparisonRegistry(tmp_path / "registry.json")
    reg.register(_kpis("deterministic", 60.0, 0.70))
    with pytest.raises(KeyError):
        reg.compare("deterministic", "planner")


def test_render_table_orders_stages(tmp_path: Path):
    reg = ComparisonRegistry(tmp_path / "registry.json")
    reg.register(_kpis("analyst", 55.0, 0.85))
    reg.register(_kpis("deterministic", 60.0, 0.70))
    table = reg.render_table()
    # deterministic must precede analyst regardless of insertion order.
    assert table.index("deterministic") < table.index("analyst")
