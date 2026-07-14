"""Planner Agent evaluation (Phase 7).

Runs the Planner over seven dataset profiles and reports the plan it produces,
plus fallback rate and latency. Two modes:

  default : uses the configured provider (Echo by default) -> demonstrates the
            safe-fallback path (100% deterministic with no real LLM).
  --sim   : uses a SIMULATED senior-CV-engineer provider (a heuristic stand-in
            for a real LLM, since this environment has no API key) so the
            differentiation across profiles can be shown end to end.

Writes benchmarks/planner_eval.md. Run: python scripts/planner_eval.py [--sim]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from vds.agents.llm import LLMClient
from vds.agents.messages import CompletionResponse, Usage
from vds.agents.planner import ExecutionPlanner
from vds.agents.planner_agent import DatasetContext, LLMPlanner
from vds.agents.providers.base import BaseProvider
from vds.agents.providers.echo import EchoProvider
from vds.config.settings import LLMSettings, Settings

# (name, image_count, resolution [w,h], density hint, note)
PROFILES = [
    ("small", 10, [128, 128], 2, "quick smoke set"),
    ("large", 5000, [128, 128], 3, "throughput matters"),
    ("drone", 400, [3840, 2160], 6, "high-res, small objects from altitude"),
    ("surveillance", 800, [1920, 1080], 4, "fixed-camera, medium objects"),
    ("mixed_res", 300, [2048, 1536], 3, "varied resolutions"),
    ("dense", 200, [1280, 720], 14, "crowded scenes"),
    ("sparse", 200, [1280, 720], 1, "few objects per frame"),
]


def make_context(name, count, res, density) -> DatasetContext:
    return DatasetContext(
        project_id=name,
        image_count=count,
        resolution_summary={"count": count, "common": res, "max": res,
                            "megapixels_max": round(res[0] * res[1] / 1e6, 2)},
        file_types=["png"],
        classes=["object"],
        available_detectors=["builtin"],
        available_segmenters=["builtin"],
        gpu_device="cuda",
        vram_budget_mb=8192,
        export_format="coco",
        review_budget_hours=8.0,
        user_preferences={"density_hint": density},
    )


class SimulatedCVEngineerProvider(BaseProvider):
    """A heuristic stand-in for a real LLM planner — NOT a model. It reads the
    dataset context and emits an adapted PlannerPlan so the eval can demonstrate
    differentiation without API credentials. Replace by pointing config at a real
    provider."""

    name = "simulated"

    def complete(self, request) -> CompletionResponse:
        ctx = json.loads(request.messages[-1].content.split("\n\n", 1)[1])
        res = ctx["resolution_summary"]
        mp = res.get("megapixels_max", 1.0)
        density = ctx["user_preferences"].get("density_hint", 3)
        count = ctx["image_count"]

        tiling = mp > 4.0  # high-res -> tile
        batch = 8 if mp > 4 else (64 if count > 1000 else 24)
        threshold = 0.55 if density > 8 else (0.3 if density < 2 else 0.4)
        review = min(60.0, 5.0 + density * 3.0 + (10.0 if tiling else 0.0))
        plan = {
            "detector": "builtin", "segmenter": "builtin", "run_segmentation": density < 10,
            "confidence_threshold": threshold, "tiling_required": tiling,
            "batch_size": batch, "worker_count": 8 if count > 1000 else 4,
            "expected_processing_seconds": round(count * (0.08 if tiling else 0.04), 1),
            "expected_gpu_mb": 4096 if tiling else 2048,
            "expected_review_percent": round(review, 1),
            "expected_annotation_density": float(density),
            "export_format": "coco",
            "execution_order": ["detect", "segment", "verify", "export"],
            "rationale": [
                {"decision": "tiling", "confidence": 0.9,
                 "justification": f"{mp}MP images {'need' if tiling else 'do not need'} tiling"},
                {"decision": "confidence_threshold", "confidence": 0.8,
                 "justification": f"density {density} sets the precision/recall balance"},
            ],
            "summary": f"{count} imgs, {mp}MP, density {density}",
        }
        return CompletionResponse(
            text=json.dumps(plan), model="simulated-cv-engineer", provider=self.name,
            usage=Usage(prompt_tokens=200, completion_tokens=180),
        )


def main() -> None:
    sim = "--sim" in sys.argv
    config = LLMSettings(model="simulated-cv-engineer" if sim else "echo-model",
                         retry_backoff_seconds=0.0)
    provider = SimulatedCVEngineerProvider(config) if sim else EchoProvider(config)
    planner = LLMPlanner(LLMClient(provider, config), ExecutionPlanner(Settings()))

    rows, fallbacks = [], 0
    for name, count, res, density, _note in PROFILES:
        r = planner.plan(make_context(name, count, res, density))
        if r.source == "deterministic":
            fallbacks += 1
        p = r.processing_plan
        ai = r.ai_plan
        rows.append(
            f"| {name} | {r.source} | {p.batch_size} | {p.confidence_threshold} | "
            f"{ai.tiling_required if ai else '-'} | "
            f"{ai.expected_review_percent if ai else '-'} | "
            f"{ai.run_segmentation if ai else '-'} | {r.latency_ms} |"
        )

    md = [
        "# Planner Agent Evaluation",
        "",
        f"Mode: **{'simulated LLM' if sim else 'configured provider (Echo)'}**  ·  "
        f"fallback rate: **{fallbacks}/{len(PROFILES)}**",
        "",
        "Simulated mode uses a heuristic stand-in for a real LLM (no API key in "
        "this environment); it demonstrates that the Planner adopts differentiated "
        "plans. Echo mode demonstrates the safe deterministic fallback." if sim else
        "Echo cannot produce a valid plan, so every profile falls back to the "
        "deterministic planner — the safety path working as designed.",
        "",
        "| Profile | source | batch | conf | tiling | review% | segment | latency ms |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
    ]
    out = Path("benchmarks") / "planner_eval.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
