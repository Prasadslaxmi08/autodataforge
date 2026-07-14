"""VLM Verifier benchmark (Phase 9).

Compares the deterministic RuleBasedVerifier against the multimodal LLMVerifier on
a fixed scenario with KNOWN ground truth, and writes:
  benchmarks/vlm_verifier_metrics.json  — the seven required metrics for both
  benchmarks/vlm_verifier_report.md      — side-by-side, per-annotation

Modes:
  default : configured provider (Echo, text-only) -> LLMVerifier falls back to the
            deterministic verifier. Proves the fallback path (phase brief).
  --sim   : a SIMULATED near-oracle VLM provider (heuristic stand-in for a real
            vision model; no API key here) that emits valid structured verdicts so
            the VLM path, evidence enforcement, and the metric harness run end to
            end. Replace by pointing config at a real vision provider.

Run: python scripts/vlm_verifier_eval.py [--sim]
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from vds.agents.cost import estimate_cost
from vds.agents.llm import LLMClient
from vds.agents.messages import CompletionResponse, Usage
from vds.agents.providers.base import BaseProvider
from vds.agents.providers.echo import EchoProvider
from vds.agents.verifier import APPROVED, NEEDS_REVIEW, REJECTED, RuleBasedVerifier
from vds.agents.vlm_verifier import LLMVerifier
from vds.config.settings import LLMSettings
from vds.core.contracts import Annotation, Box2D, Mask, Provenance

# --- ground-truth scenario -------------------------------------------------
# Each entry: (confidence, box, truth_decision, note). truth_decision is what a
# perfect reviewer looking at the pixels should decide.
APPROVE, REJECT, REVIEW = "approved", "rejected", "needs_human_review"

SCENARIO = [
    (0.92, Box2D(x=10, y=10, w=30, h=30), APPROVE, "correct object, high conf"),
    (0.88, Box2D(x=60, y=10, w=30, h=30), REJECT, "hallucination — nothing there"),
    (0.22, Box2D(x=10, y=60, w=30, h=30), APPROVE, "correct object, low conf (det wrongly rejects)"),
    (0.55, Box2D(x=60, y=60, w=30, h=30), APPROVE, "correct object, mid conf (det sends to review)"),
    (0.80, Box2D(x=110, y=10, w=30, h=30), REJECT, "wrong class label"),
    (0.83, Box2D(x=110, y=60, w=40, h=40), REVIEW, "loose/misaligned box"),
    (0.90, Box2D(x=0, y=0, w=0, h=30), REJECT, "invalid geometry"),
    (0.70, Box2D(x=10, y=10, w=30, h=30), REJECT, "duplicate of #0"),
]


def build_annotations() -> list[Annotation]:
    anns = []
    for i, (conf, box, _truth, _note) in enumerate(SCENARIO):
        anns.append(Annotation(
            id=f"ann{i}", image_id="img", label="object", geometry=box,
            confidence=conf, state="labeled", provenance=Provenance(source="engine.labeling"),
            mask=Mask(rle=json.dumps([100, 50, 100]), height=20, width=20),
        ))
    return anns


def build_image() -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (160, 100), (235, 235, 235))
    d = ImageDraw.Draw(im)
    # Draw the real objects (everything the ground truth says exists).
    for _conf, b, truth, _note in SCENARIO:
        if truth in (APPROVE, REVIEW):
            d.rectangle([b.x, b.y, b.x + b.w, b.y + b.h], fill=(20, 20, 20))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


# --- simulated near-oracle VLM ---------------------------------------------
class SimulatedVLMProvider(BaseProvider):
    """Heuristic stand-in for a real vision model — NOT inference. It is handed the
    ground-truth decisions and emits a valid VLMSceneVerdict citing real evidence
    keys, so the VLM path and metric harness can be exercised without an API key.
    A real vision provider replaces this by config."""

    name = "simulated-vlm"

    def complete(self, request) -> CompletionResponse:
        prompt = request.messages[-1].content
        keys = {t[1:-1] for t in prompt.split() if t.startswith("[") and t.endswith("]")}

        def cite(*ks):
            return [k for k in ks if k in keys]

        verdicts, findings = [], []
        for i, (_conf, _box, truth, note) in enumerate(SCENARIO):
            label_ok = "wrong class" not in note
            identity_ok = "hallucination" not in note and "duplicate" not in note
            corrections = []
            if truth != APPROVE:
                corrections.append({
                    "issue": note,
                    "suggested_correction": "reject" if truth == REJECT else "route to human review",
                    "reasoning": f"Pixels and evidence indicate: {note}.",
                    "confidence": 0.8, "expected_impact": "Prevents a bad annotation from shipping.",
                    "cited_evidence": cite(f"a{i}_conf", f"a{i}_box", f"a{i}_boxvalid", f"a{i}_label"),
                })
            verdicts.append({
                "annotation_index": i, "decision": truth,
                "object_identity_correct": identity_ok, "label_correct": label_ok,
                "bbox_quality": "invalid" if "invalid" in note else ("loose" if "loose" in note else "good"),
                "segmentation_quality": "good",
                "confidence": 0.85, "reasoning": note,
                "cited_evidence": cite(f"a{i}_conf", f"a{i}_box"),
                "corrections": corrections,
            })
            if "duplicate" in note:
                findings.append({
                    "kind": "duplicate_annotation", "description": f"#{i} duplicates #0",
                    "annotation_indices": [0, i], "reasoning": "Boxes overlap heavily.",
                    "confidence": 0.9, "suggested_correction": "Drop the lower-confidence duplicate.",
                    "expected_impact": "Removes double-counting.",
                    "cited_evidence": cite("dup_0_7", f"a{i}_box"),
                })
        findings.append({
            "kind": "missing_annotation", "description": "No missed objects detected.",
            "annotation_indices": [], "reasoning": "Every visible object already has an annotation.",
            "confidence": 0.6, "suggested_correction": "None.", "expected_impact": "None.",
            "cited_evidence": cite("scene_count"),
        })
        out = {"annotation_verdicts": verdicts, "scene_findings": findings,
               "summary": "8 annotations reviewed against pixels + evidence."}
        return CompletionResponse(text=json.dumps(out), model="simulated-vlm",
                                  provider=self.name, usage=Usage(prompt_tokens=1400, completion_tokens=650))


# --- scoring ---------------------------------------------------------------
# Core VerdictLabel -> the 3-way decision. Both verifiers emit only these three
# routing labels (APPROVED="correct", NEEDS_REVIEW="bad_geometry", REJECTED="hallucination").
_DET_TO_DECISION = {APPROVED: APPROVE, NEEDS_REVIEW: REVIEW, REJECTED: REJECT}


def score(decisions: list[str]) -> dict:
    truth = [s[2] for s in SCENARIO]
    n = len(truth)
    correct = sum(d == t for d, t in zip(decisions, truth, strict=True))
    false_approvals = sum(d == APPROVE and t != APPROVE for d, t in zip(decisions, truth, strict=True))
    false_rejections = sum(d == REJECT and t == APPROVE for d, t in zip(decisions, truth, strict=True))
    reviews = sum(d == REVIEW for d in decisions)
    return {
        "approval_accuracy": round(correct / n, 3),
        "false_approvals": false_approvals,
        "false_rejections": false_rejections,
        "human_review_rate": round(reviews / n, 3),
        "decisions": decisions,
    }


def main() -> None:
    sim = "--sim" in sys.argv
    image, anns = build_image(), build_annotations()

    # Deterministic baseline.
    det = RuleBasedVerifier()
    det_decisions = [_DET_TO_DECISION[det.verify(image, a).verdict] for a in anns]
    det_metrics = score(det_decisions)

    # VLM (or its fallback).
    provider = SimulatedVLMProvider(LLMSettings()) if sim else EchoProvider(LLMSettings())
    model = "simulated-vlm" if sim else "echo-model"
    vlm = LLMVerifier(LLMClient(provider, LLMSettings(model=model)))
    result = vlm.verify_scene(image, anns)
    vlm_decisions = [_DET_TO_DECISION[v.verdict] for v in result.verdicts]
    vlm_metrics = score(vlm_decisions)

    review_reduction = round(det_metrics["human_review_rate"] - vlm_metrics["human_review_rate"], 3)
    est_cost = estimate_cost(model, Usage(prompt_tokens=result.prompt_tokens,
                                          completion_tokens=result.completion_tokens))

    metrics = {
        "mode": "simulated-vlm" if sim else "echo(fallback)",
        "vlm_source": result.source,
        "fallback_reason": result.fallback_reason,
        "deterministic": det_metrics,
        "vlm": vlm_metrics,
        "human_review_reduction": review_reduction,
        "evidence_coverage": result.evidence_coverage,
        "dropped_recommendations": result.dropped_recommendations,
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "estimated_cost_usd": est_cost,
    }
    Path("benchmarks").mkdir(exist_ok=True)
    Path("benchmarks/vlm_verifier_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Per-annotation markdown table.
    rows = ["| # | truth | deterministic | vlm | note |", "|---|---|---|---|---|"]
    for i, (s, dd, vd) in enumerate(zip(SCENARIO, det_decisions, vlm_decisions, strict=True)):
        rows.append(f"| {i} | {s[2]} | {dd} | {vd} | {s[3]} |")
    md = [
        f"# VLM Verifier Benchmark ({'simulated VLM' if sim else 'Echo → deterministic fallback'})",
        "",
        f"VLM source: **{result.source}**"
        + (f"  ·  fallback: {result.fallback_reason}" if result.fallback_reason else ""),
        "",
        "## Metrics",
        "| metric | deterministic | vlm |",
        "|---|---|---|",
        f"| approval accuracy | {det_metrics['approval_accuracy']} | {vlm_metrics['approval_accuracy']} |",
        f"| false approvals | {det_metrics['false_approvals']} | {vlm_metrics['false_approvals']} |",
        f"| false rejections | {det_metrics['false_rejections']} | {vlm_metrics['false_rejections']} |",
        f"| human-review rate | {det_metrics['human_review_rate']} | {vlm_metrics['human_review_rate']} |",
        "",
        f"human-review reduction: **{review_reduction}**  ·  latency: {result.latency_ms} ms  ·  "
        f"tokens: {result.prompt_tokens}+{result.completion_tokens}  ·  cost: {est_cost}  ·  "
        f"evidence coverage: {result.evidence_coverage}",
        "",
        "## Per-annotation",
        *rows,
    ]
    Path("benchmarks/vlm_verifier_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
