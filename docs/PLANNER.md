# Planner Agent (Phase 7)

The platform's first true AI agent. It analyses an imported dataset and produces
an **execution plan** — which models, thresholds, tiling, batching, review
expectations, and order — with a confidence and justification for each decision.
It reasons through the provider-agnostic `LLMClient` (never a hardcoded provider)
and every output is schema-validated before use.

## Prompt design

System prompt (`PLANNER_SYSTEM_PROMPT` in `vds/agents/planner_agent.py`) casts the
model as a **Senior Computer Vision Engineer** optimizing five explicit goals:
accuracy, speed, GPU utilization, human-review reduction, annotation quality. It
constrains choices to the *available* models, enforces the VRAM budget, ties
tiling to resolution and review-percent to difficulty, and demands **JSON only,
concise, with a confidence + one-line justification per major decision**. No
free-form reasoning.

## LabelingPlan schema

`PlannerPlan` (validated Pydantic, constrained fields so bad output fails fast):
`detector`, `segmenter`, `run_segmentation`, `confidence_threshold` (0–1),
`tiling_required`, `batch_size` (1–1024), `worker_count` (1–64),
`expected_processing_seconds`, `expected_gpu_mb`, `expected_review_percent`
(0–100), `expected_annotation_density`, `export_format`, `execution_order`,
`rationale: [{decision, confidence, justification}]`, `summary`. It reduces to the
pipeline's existing `ProcessingPlan`, so nothing downstream changes.

## Safety: never stops the pipeline

On invalid JSON, schema violation, an unavailable model choice, a provider
failure, or missing credentials, the Planner **falls back to the deterministic
`ExecutionPlanner`** and logs the reason. Structure comes from the schema (with
automatic validation-retry via `LLMClient.structured`); reasoning comes from the
LLM. With the default Echo provider (no real LLM) it falls back every time — the
correct safe default.

## Evaluation (`scripts/planner_eval.py`, `benchmarks/planner_eval.md`)

Seven profiles: small, large, drone, surveillance, mixed-res, dense, sparse.

- **Configured provider = Echo (default):** 7/7 fallback → identical deterministic
  plans (batch 16, conf 0.3). The safety path, proven.
- **Simulated senior-CV-engineer provider** (a heuristic stand-in for a real LLM,
  since this environment has no API key): 0/7 fallback, and the plans differ
  appropriately —

  | Profile | batch | conf | tiling | review% | segment |
  |---|---|---|---|---|---|
  | drone (4K, small obj) | 8 | 0.40 | **yes** | **33%** | yes |
  | dense (crowded) | 24 | **0.55** | no | **47%** | **no** |
  | large | **64** | 0.40 | no | 14% | yes |
  | sparse | 24 | **0.30** | no | **8%** | yes |

  High-resolution drone imagery gets tiling and a bigger review budget; crowded
  scenes raise the confidence bar and skip segmentation; large sets get bigger
  batches; sparse scenes need little review. The deterministic planner produces
  none of this variation.

## Metrics captured per plan
Latency, retry count, fallback flag, prompt/completion tokens, and cost estimate
(`vds/agents/cost.py` — priced for Claude/GPT models; `None` for local/echo).

## Answers

**What the Planner improved.** Decisions that depend on *judgement about the data*
now adapt to it: tiling for high-resolution/small-object imagery, confidence
thresholds by scene density, batch/worker sizing by dataset size, whether to run
segmentation at all, and an honest human-review estimate. The deterministic
planner used one fixed value for each.

**What stays deterministic.** The *fallback plan itself*, plus everything with a
correct answer: `num_batches` math, VRAM-budget enforcement, format validity, and
the whole downstream pipeline. The Planner advises; deterministic code guards.

**Production-ready?** The framework, fallback, validation, and metrics are
production-ready and safe (it cannot break the pipeline). It is **not yet
delivering AI value in production** because no real provider is configured here —
with Echo it always falls back. Ready to switch on the moment a provider + key are
set; until then it safely degrades to the validated deterministic baseline.
