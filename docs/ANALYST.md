# AI Dataset Analyst (Phase 8)

The platform's second AI agent. It reviews a **completed** annotation run — using
structured metrics only, never raw images — like a Senior CV Scientist, and
produces an evidence-backed engineering report plus recommendations for both the
pipeline and the Planner. It uses the provider-agnostic `LLMClient` exclusively.

## The anti-hallucination design (the hard requirement)

"Every recommendation must be supported by measurable evidence; never hallucinate
statistics." This is enforced structurally, not by prompt alone:

1. **Deterministic evidence pack** — `build_evidence()` computes every fact from
   the run's metrics (throughput, bottleneck share, density, small-object ratio,
   approval/review/reject, resource use, historical deltas). Each fact has a
   `[key]`. The AI never sees a blank slate; it sees pre-computed facts.
2. **The LLM cites, it does not count.** The prompt gives the model the evidence
   pack and requires every recommendation's `supporting_metrics` to be `[key]`
   tokens that exist in the pack.
3. **Deterministic citation validator** — `_enforce_evidence()` drops any
   recommendation citing a key that is not in the pack and strips invented
   citations. A fabricated statistic or unsupported recommendation cannot survive.
   `evidence_coverage` reports the fraction that passed.

So the AI supplies interpretation, prioritization, and root-cause narrative; it
never supplies a number. That is the difference between an engineering assistant
and a plausible-sounding chatbot.

## Inputs

An `AnalystContext`: the run's `ExecutionReport` (benchmark + quality + errors +
export), the optional `PlannerResult`, an optional `DatasetContext` (for
resolution facts), and historical `StageKPIs` from the comparison registry. No
images.

## Report schema

`AnalystReport` (validated): executive summary, pipeline performance, dataset
characteristics, detection / segmentation / verification analysis, resource
utilization, strengths, weaknesses, root-cause analysis, `recommendations` and
`planner_recommendations` (each: action, target, reason, expected impact,
confidence, `supporting_metrics`, trade-offs), expected improvement, confidence,
next actions.

## Safety & metrics

Like the Planner, it never fails the caller: on invalid output, provider failure,
or missing credentials it returns the **deterministic report** built from the same
evidence pack (rule-based recommendations, always evidence-backed), and logs why.
Metrics captured per run: analysis latency, prompt/completion tokens, cost
estimate, recommendation count, evidence coverage, fallback flag, structured
validity (`benchmarks/analyst_metrics.json`).

## Deterministic reporting vs AI reasoning

Measured from `scripts/analyst_eval.py` on the same run:

| | Deterministic (echo fallback) | AI (simulated CV scientist) |
|---|---|---|
| source | deterministic | ai |
| recommendations | 5 (rule-based, one per matched pattern) | 3 (prioritized) |
| evidence coverage | 1.0 | 1.0 |
| root cause | single templated statement | connected narrative across facts |
| tokens / latency | 0 / ~1 ms | 950 / ~1 ms |

The deterministic report *states* every matched pattern; the AI report *ranks*
them, ties them into a root-cause story, adds expected-impact and trade-offs, and
frames Planner-specific advice — while both stay at 100% evidence coverage.

## Answers

**1. Insights deterministic logic could not produce.** Prioritization ("fix the
verifier first, it's the root cause, before chasing the bottleneck"), cross-fact
root-cause narratives (linking 100%-approval + 0%-review + high-confidence into
"uncalibrated, not perfect"), and expected-impact framing. Deterministic rules can
*detect* each pattern but cannot *weigh, connect, or narrate* them.

**2. Recommendations that genuinely need AI.** Trade-off-aware prioritization when
signals conflict (speed vs recall), Planner-directed advice phrased as reusable
knowledge, and expected-improvement estimates. Single-condition recommendations
(e.g. "duplicate-heavy → tighten dedup") do not need AI.

**3. Analyses that should stay deterministic.** The entire evidence pack — every
statistic, threshold check, and historical delta — and the citation validator.
Numbers and their verification must never be an LLM's job.

**4. How it improves future annotation quality.** Its Planner recommendations
become future Planner knowledge (tiling for high-res, skip-segmentation for
sparse, confidence-threshold changes), and its verification findings drive the
roadmap (the uncalibrated-confidence finding is exactly what motivates the real
VLM verifier). Evidence-backed, so improvements are measured, not guessed.

**5. Production-ready?** The evidence layer, enforcement, fallback, and metrics are
production-ready and safe (it cannot fabricate a stat or fail the caller). Full AI
value needs a real provider configured (with Echo it falls back to the
deterministic report). Ready to switch on with a provider + key.
