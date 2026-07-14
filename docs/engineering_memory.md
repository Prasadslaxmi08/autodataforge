# Engineering Memory (Phase 10)

Long-term, **deterministic** operational memory that lets the platform learn from
previous annotation jobs and improve future planning — **without RAG, vector
databases, embeddings, or conversational memory**. It is a structured, versioned,
queryable, explainable record of engineering decisions and their measured
outcomes.

## Architecture (no redesign)

Layered on the existing pipeline; nothing was restructured.

| File | Responsibility |
|------|----------------|
| `vds/memory/schema.py` | Typed records: `EngineeringMemory`, `DatasetFingerprint`, `PlannerDecisions`, `ExecutionMetrics`, `AnalystConclusions`, `VerificationOutcomes`, `BenchmarkSummary`. |
| `vds/memory/store.py` | `MemoryStore` — append-only, versioned JSON file. Corruption-safe (quarantines bad files, skips bad rows), atomic writes, duplicate suppression. |
| `vds/memory/similarity.py` | `SimilarityEngine` — deterministic weighted feature scoring. No embeddings. Every match is explained. |
| `vds/memory/builder.py` | Builds a memory from **measured outputs + validated recommendations only** (the anti-hallucination gate). |
| `vds/memory/trends.py` | `TrendAnalyzer` — evolution, strategy performance, effective thresholds, engineering reports. |
| `vds/memory/service.py` | `EngineeringMemoryService` — the facade the Planner (`recall`) and Analyst (`record_execution`) use. |

Storage is one JSON file per environment — the same "a table in a file" choice the
comparison registry already makes. Single-node, small, git-diffable, auditable.

## Memory model

Every record carries: unique id, timestamp, dataset fingerprint, planner
decisions, execution metrics, analyst conclusions, verification outcomes,
benchmark summary, engineering recommendations, validation status, confidence,
version. **No raw images** — only engineering knowledge (enforced by test
`test_no_raw_images_ever_stored`).

Versioning: re-recording the same dataset *family* (same fingerprint hash) appends
a new version; the old one is retained (complete history, never overwritten). An
*identical* record (same content hash) is suppressed as a duplicate.

## Deterministic similarity

`similarity(query, candidate)` is a weighted average of per-feature agreement over
resolution, dataset scale, scene density, object density, small-object ratio,
duplicate ratio, average confidence, and scene type. Ratios score by absolute gap;
magnitudes by relative gap. **Features unknown in the query are skipped**, so the
Planner's pre-run query (which only knows resolution / scale / scene type) matches
on that subset. Ties break deterministically (score, then newest, then id). Each
match returns the features it matched on and their contributions.

## Planner integration

`LLMPlanner` optionally holds an `EngineeringMemoryService`. Before planning it
asks *"have we processed a similar dataset?"*, injects the ranked matches +
validated historical recommendations + historical review rates into the prompt,
and records the influence on `PlannerResult.memory_note` / `memory_matches`. If no
similar memory exists, it says so explicitly.

## Analyst integration

`LLMAnalyst.remember(...)` (→ `service.record_execution`) runs after a completed
execution. **Only validated recommendations** (those the Analyst's evidence check
kept) become reusable knowledge; unvalidated ones are dropped by the builder. No
automatic overwrite; full history maintained.

## Reports & trends

`service.trend_report()` and `service.engineering_report()` render: most successful
planner strategies, most effective confidence thresholds, most common
dataset/verification problems, and performance / review-rate / quality evolution.
See `benchmarks/memory_report.md` and `benchmarks/memory_metrics.json`
(`python scripts/memory_eval.py`).

---

## Required answers

**1. What knowledge is worth remembering?**
Measured engineering facts and their context: the dataset fingerprint, the planner
decisions that produced a run, the resulting throughput / review-rate / quality
metrics, verification failure patterns, and *validated* (evidence-backed) analyst
recommendations. In short — the inputs, the decisions, and the measured
consequences, so future decisions can be conditioned on real prior outcomes.

**2. What should never be remembered?**
Raw images or pixels; hallucinated or unvalidated statements; any analyst
recommendation that failed evidence validation; conversational history. Nothing
that wasn't measured or validated enters memory (enforced in `builder.py`).

**3. How does Engineering Memory improve future Planner decisions?**
The Planner recalls the most similar past datasets and sees which detector,
thresholds, tiling, and batching produced good quality and low review rates on
comparable data — plus validated recommendations from those runs. It starts from
evidence instead of from zero, and explains when prior experience shaped the plan.

**4. How does it reduce human review?**
Review rate is a stored, tracked metric. The Planner can prefer decision profiles
that historically yielded low review rates on similar datasets, and the trend
report surfaces which strategies drive review rate down — closing a feedback loop
that a from-scratch planner never had.

**5. How does this prepare the platform for future Vision RAG integration?**
It establishes the *structured knowledge substrate* first: typed, versioned,
validated, explainable engineering records with a clean retrieval interface
(`recall(fingerprint) → ranked, explained matches`). A future Vision RAG layer can
swap the deterministic similarity engine for embedding retrieval **behind the same
interface**, while keeping the anti-hallucination gate, validation status, and
audit trail — retrieval becomes richer without the memory becoming a black box.
