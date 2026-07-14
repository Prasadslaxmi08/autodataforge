"""AI Dataset Analyst evaluation (Phase 8).

Runs a real annotation job, then the Analyst, and writes:
  benchmarks/analyst_report_example.md  — the engineering report
  benchmarks/analyst_metrics.json        — latency/tokens/cost/coverage/fallback

Two modes:
  default : configured provider (Echo) -> deterministic report (safe fallback).
  --sim   : a SIMULATED senior-CV-scientist provider (heuristic stand-in for a
            real LLM; no API key here) that reasons over the evidence pack and
            cites real evidence keys, so the AI path and evidence enforcement can
            be shown end to end.

Run: python scripts/analyst_eval.py [--sim]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from vds.agents.analyst_agent import AnalystContext, LLMAnalyst, build_evidence
from vds.agents.llm import LLMClient
from vds.agents.messages import CompletionResponse, Usage
from vds.agents.providers.base import BaseProvider
from vds.agents.providers.echo import EchoProvider
from vds.config.settings import LLMSettings, Settings, StorageSettings
from vds.container import Container


class SimulatedCVScientistProvider(BaseProvider):
    """Heuristic stand-in for a real LLM Analyst — NOT a model. Reads the evidence
    pack from the prompt and emits a valid AnalystReport that cites only real
    evidence keys. Replace by pointing config at a real provider."""

    name = "simulated"

    def complete(self, request) -> CompletionResponse:
        pack = request.messages[-1].content
        keys = {tok[1:-1] for tok in pack.split() if tok.startswith("[") and tok.endswith("]")}

        def rec(action, target, reason, impact, conf, metrics, trade):
            cites = [m for m in metrics if m in keys]
            return {"action": action, "target": target, "reason": reason,
                    "expected_impact": impact, "confidence": conf,
                    "supporting_metrics": cites, "trade_offs": trade}

        recs, planner_recs = [], []
        if "uncalibrated_confidence" in keys:
            recs.append(rec("Introduce a learned/VLM verifier to calibrate confidence.",
                            "verification", "100% approval with 0% review is not real quality.",
                            "Trustworthy approvals and meaningful review routing.", 0.85,
                            ["approval_rate", "review_rate", "uncalibrated_confidence"], "Adds model cost."))
        if "bottleneck" in keys:
            recs.append(rec("Batch or sample the dominant stage.", "pipeline",
                            "One stage dominates runtime.", "Lower total runtime.", 0.7,
                            ["bottleneck"], "Sampling may miss cases."))
        if "small_object_dominance" in keys or "high_resolution" in keys:
            planner_recs.append(rec("Enable tiling and a higher-accuracy detector.", "planner",
                                    "Small/high-res objects are error-prone.", "Higher recall.", 0.7,
                                    ["small_object_dominance", "high_resolution"], "More compute."))
        if "sparse_scenes" in keys:
            planner_recs.append(rec("Skip segmentation on sparse scenes.", "planner",
                                    "Few objects per image.", "Faster runs.", 0.5,
                                    ["sparse_scenes", "annotation_density"], "No masks."))
        report = {
            "executive_summary": "Run reviewed against measured evidence.",
            "pipeline_performance": "Throughput and bottleneck assessed from stage timings.",
            "dataset_characteristics": "Density and object-size profile derived from metrics.",
            "detection_analysis": "Detection counts and confidence assessed.",
            "segmentation_analysis": "Mask coverage assessed.",
            "verification_analysis": "Approval/review/reject distribution assessed.",
            "resource_utilization": "RAM and GPU utilization within budget.",
            "strengths": ["Export validated", "Stable, deterministic pipeline"],
            "weaknesses": ["Confidence uncalibrated" if "uncalibrated_confidence" in keys
                           else "None material"],
            "root_cause_analysis": "Geometric confidence heuristic drives the approval pattern."
            if "uncalibrated_confidence" in keys else "No dominant root cause.",
            "recommendations": recs,
            "planner_recommendations": planner_recs,
            "expected_improvement": "Calibrated confidence should convert vacuous 100% approval "
                                    "into a real, reducible review rate.",
            "confidence": 0.8,
            "next_actions": [r["action"] for r in (recs + planner_recs)[:3]],
        }
        return CompletionResponse(text=json.dumps(report), model="simulated-cv-scientist",
                                  provider=self.name, usage=Usage(prompt_tokens=600, completion_tokens=350))


def render_report(report) -> str:
    def recs(rs):
        return "\n".join(
            f"- **{r.action}** ({r.target}, conf {r.confidence})\n"
            f"  - reason: {r.reason}\n  - impact: {r.expected_impact}\n"
            f"  - evidence: {', '.join(r.supporting_metrics)}\n  - trade-off: {r.trade_offs}"
            for r in rs) or "- none"
    return "\n".join([
        f"## Executive Summary\n{report.executive_summary}",
        f"## Pipeline Performance\n{report.pipeline_performance}",
        f"## Dataset Characteristics\n{report.dataset_characteristics}",
        f"## Detection Analysis\n{report.detection_analysis}",
        f"## Segmentation Analysis\n{report.segmentation_analysis}",
        f"## Verification Analysis\n{report.verification_analysis}",
        f"## Resource Utilization\n{report.resource_utilization}",
        "## Strengths\n- " + "\n- ".join(report.strengths),
        "## Weaknesses\n- " + "\n- ".join(report.weaknesses),
        f"## Root Cause Analysis\n{report.root_cause_analysis}",
        "## Engineering Recommendations\n" + recs(report.recommendations),
        "## Planner Recommendations\n" + recs(report.planner_recommendations),
        f"## Expected Improvement\n{report.expected_improvement}",
        f"## Confidence\n{report.confidence}",
        "## Next Actions\n- " + "\n- ".join(report.next_actions),
    ])


def main() -> None:
    sim = "--sim" in sys.argv
    provider = SimulatedCVScientistProvider(LLMSettings()) if sim else EchoProvider(LLMSettings())
    settings = Settings(storage=StorageSettings(cas_root=Path("benchmarks/_analyst_cas")))
    container = Container(settings=settings, db_path="benchmarks/analyst_eval.db",
                          artifacts_dir=Path("benchmarks/_analyst_art"))
    analyst = LLMAnalyst(LLMClient(provider, LLMSettings(model="simulated-cv-scientist"
                                                         if sim else "echo-model")))

    # Produce a real completed run to analyse.
    from PIL import Image, ImageDraw
    data = Path("benchmarks/_analyst_data")
    data.mkdir(parents=True, exist_ok=True)
    for i in range(12):
        im = Image.new("RGB", (128, 128), (240, 240, 240))
        d = ImageDraw.Draw(im)
        for j in range(2):
            x = 10 + i * 3 + j * 40
            d.rectangle([x, 10 + j * 30, x + 16, 26 + j * 30], fill=(15, 15, 15))
        im.save(data / f"a_{i:02d}.png")
    execution = container.pipeline.run(str(data), name="analyst_demo", dest="benchmarks/_analyst_export")

    result = analyst.analyze(AnalystContext(execution=execution))
    evidence = build_evidence(AnalystContext(execution=execution))

    md = [
        f"# AI Dataset Analyst — Example Report ({'simulated LLM' if sim else 'deterministic fallback'})",
        "",
        f"source: **{result.source}**  ·  evidence coverage: **{result.evidence_coverage}**  ·  "
        f"recommendations: **{result.recommendation_count}**  ·  latency: {result.latency_ms} ms"
        + (f"  ·  fallback: {result.fallback_reason}" if result.fallback_reason else ""),
        "",
        "## Evidence Pack (deterministic — the facts the AI must cite)",
        "```",
        evidence.render(),
        "```",
        "",
        render_report(result.report),
    ]
    Path("benchmarks/analyst_report_example.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    metrics = {
        "mode": "simulated" if sim else "echo",
        "source": result.source,
        "analysis_latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "estimated_cost_usd": result.estimated_cost_usd,
        "recommendation_count": result.recommendation_count,
        "evidence_coverage": result.evidence_coverage,
        "unsupported_recommendations": result.unsupported_recommendations,
        "fallback": result.source == "deterministic",
        "structured_valid": result.structured_valid,
    }
    Path("benchmarks/analyst_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
