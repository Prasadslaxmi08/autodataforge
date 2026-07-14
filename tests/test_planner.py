"""Unit test for the deterministic execution planner."""

from __future__ import annotations

from vds.agents.planner import ExecutionPlanner
from vds.config.settings import Settings


def test_plan_counts_and_batches():
    settings = Settings()  # batch_size default 16
    plan = ExecutionPlanner(settings).plan("p1", image_count=40)
    assert plan.image_count == 40
    assert plan.batch_size == 16
    assert plan.num_batches == 3  # ceil(40/16)
    assert plan.classes == ["object"]
    assert plan.export_format == settings.export.default_format
    assert plan.estimated_seconds > 0


def test_plan_empty_dataset():
    plan = ExecutionPlanner(Settings()).plan("p1", image_count=0)
    assert plan.num_batches == 0
    assert plan.estimated_seconds == 0
