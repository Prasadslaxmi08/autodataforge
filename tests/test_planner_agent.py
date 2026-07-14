"""Planner Agent tests: AI path, schema validation, and every fallback route."""

from __future__ import annotations

import json
from pathlib import Path

from vds.agents.llm import LLMClient
from vds.agents.planner import ExecutionPlanner
from vds.agents.planner_agent import (
    DatasetContext,
    LLMPlanner,
    build_dataset_context,
)
from vds.agents.providers.echo import EchoProvider
from vds.config.settings import LLMSettings, Settings
from vds.container import Container

FAST = LLMSettings(model="m", retry_backoff_seconds=0.0, max_retries=2)

VALID_PLAN = {
    "detector": "builtin", "segmenter": "builtin", "run_segmentation": True,
    "confidence_threshold": 0.4, "tiling_required": False, "batch_size": 32,
    "worker_count": 4, "expected_processing_seconds": 12.5, "expected_gpu_mb": 2048,
    "expected_review_percent": 15.0, "expected_annotation_density": 3.2,
    "export_format": "coco", "execution_order": ["detect", "segment", "verify", "export"],
    "rationale": [{"decision": "batch_size", "confidence": 0.8, "justification": "medium set"}],
    "summary": "balanced plan",
}


def _ctx(**over) -> DatasetContext:
    base = dict(
        project_id="p1", image_count=50, resolution_summary={"count": 50},
        file_types=["png"], classes=["object"],
        available_detectors=["builtin"], available_segmenters=["builtin"],
        gpu_device="cpu", vram_budget_mb=8192, export_format="coco",
        review_budget_hours=8.0,
    )
    base.update(over)
    return DatasetContext(**base)


def _planner(provider) -> LLMPlanner:
    return LLMPlanner(LLMClient(provider, FAST), ExecutionPlanner(Settings()))


# --- AI path ---
def test_ai_plan_adopted():
    result = _planner(EchoProvider(reply=json.dumps(VALID_PLAN))).plan(_ctx())
    assert result.source == "ai"
    assert result.fallback_reason is None
    assert result.processing_plan.batch_size == 32
    assert result.processing_plan.confidence_threshold == 0.4
    assert result.processing_plan.num_batches == 2  # ceil(50/32)
    assert result.ai_plan.rationale[0].decision == "batch_size"
    assert result.latency_ms >= 0


def test_metrics_and_cost_captured():
    plan = dict(VALID_PLAN)
    provider = EchoProvider(reply=json.dumps(plan))
    result = LLMPlanner(
        LLMClient(provider, LLMSettings(model="claude-sonnet-5", retry_backoff_seconds=0.0)),
        ExecutionPlanner(Settings()),
    ).plan(_ctx())
    assert result.completion_tokens > 0
    assert result.estimated_cost_usd is not None  # sonnet is priced


# --- fallback routes ---
def test_fallback_on_invalid_json():
    result = _planner(EchoProvider(reply="not json")).plan(_ctx())
    assert result.source == "deterministic"
    assert result.fallback_reason is not None
    assert result.processing_plan.image_count == 50  # deterministic still planned


def test_fallback_on_schema_violation():
    bad = dict(VALID_PLAN, confidence_threshold=2.0)  # > 1.0, violates schema
    result = _planner(EchoProvider(reply=json.dumps(bad))).plan(_ctx())
    assert result.source == "deterministic"


def test_fallback_on_invalid_choice():
    bad = dict(VALID_PLAN, detector="grounding_dino")  # not in available list
    result = _planner(EchoProvider(reply=json.dumps(bad))).plan(_ctx())
    assert result.source == "deterministic"
    assert "not available" in result.fallback_reason


def test_fallback_on_provider_failure():
    result = _planner(EchoProvider(fail_times=99)).plan(_ctx())
    assert result.source == "deterministic"


def test_fallback_never_raises_on_unexpected_error():
    class Boom(EchoProvider):
        def complete(self, request):
            raise ValueError("unexpected non-VDS error")

    result = _planner(Boom()).plan(_ctx())  # must not propagate
    assert result.source == "deterministic"


def test_provider_not_configured_falls_back():
    from vds.agents.providers.anthropic import AnthropicProvider

    planner = LLMPlanner(
        LLMClient(AnthropicProvider(LLMSettings(api_key=None)), FAST),
        ExecutionPlanner(Settings()),
    )
    assert planner.plan(_ctx()).source == "deterministic"


# --- differentiation (plumbing): distinct LLM plans -> distinct ProcessingPlans ---
def test_different_plans_produce_different_processing_plans():
    dense = dict(VALID_PLAN, batch_size=8, confidence_threshold=0.6, tiling_required=True)
    sparse = dict(VALID_PLAN, batch_size=64, confidence_threshold=0.25)
    r_dense = _planner(EchoProvider(reply=json.dumps(dense))).plan(_ctx()).processing_plan
    r_sparse = _planner(EchoProvider(reply=json.dumps(sparse))).plan(_ctx()).processing_plan
    assert r_dense.batch_size != r_sparse.batch_size
    assert r_dense.confidence_threshold > r_sparse.confidence_threshold


# --- integration: default echo provider always falls back safely ---
def test_container_planner_falls_back_with_echo(container: Container, dataset_dir: Path):
    container.importer.import_folder("p1", str(dataset_dir))
    ctx = build_dataset_context("p1", container.settings, container.images)
    assert ctx.image_count == 3
    result = container.planner_agent.plan(ctx)
    assert result.source == "deterministic"  # echo can't produce a valid plan
    assert result.processing_plan.image_count == 3
