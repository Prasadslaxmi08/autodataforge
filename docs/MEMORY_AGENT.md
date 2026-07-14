# MemoryAgent (V2-24)

The experience layer. Before a job it **recalls** similar past projects and the
settings that worked; after a completed run it **records** what happened so the
next job can learn from it.

It **only remembers**. It never plans, never executes, never runs a model or a
tool — it retrieves and stores knowledge, and everything it returns is advice the
Planner/Decision agents (and the user) may ignore.

## Reuses the existing Engineering Memory — no duplicated storage

There is **no new database**. The MemoryAgent is a thin adapter over the Phase-10
Engineering Memory stack:

- `vds.memory.store.MemoryStore` — the same append-only, versioned JSON store
  (`data/engineering_memory.json`), with the same duplicate-suppression and
  corruption tolerance.
- `vds.memory.similarity.SimilarityEngine` — the same deterministic, explainable
  weighted-feature scorer. No embeddings, no vector DB.
- `vds.memory.schema.EngineeringMemory` — the same record schema.

V2 pipeline facts the shared schema has no column for (frame strategy, review
level, goal text, IoU, annotation count, success flag) ride in the fingerprint's
`environment` bag — which is serialized but **never scored** by similarity, so the
shared V1 schema is untouched and matching stays clean.

## Flow

```
new Goal ─▶ MemoryAgent.recall(goal, metadata?)
                 │  build query DatasetFingerprint from goal (+ metadata)
                 ▼
           SimilarityEngine.search(fingerprint, store.all())
                 │
                 └─▶ MemoryExperience
                       ├─ matches (ranked, each with a why)
                       ├─ similarity_score / confidence
                       ├─ successful_settings, recommendations
                       └─ warnings, lessons
   ... plan → decide → execute (unchanged) ...
completed run ─▶ MemoryAgent.record(goal, decision_report, execution_summary, plan?)
                 │  only if execution completed
                 ▼
           EngineeringMemory ─▶ MemoryStore.add  (dedup + version)
```

## Storage model

`record(...)` maps the V2 inputs onto the shared `EngineeringMemory`:

| Brief STORE field | Where it lands |
|---|---|
| Goal | `dataset_fingerprint.environment["goal"]` |
| Dataset Type / Domain | `dataset_fingerprint.scene_type` (e.g. `thermal_aerial`) + `environment` sensor/platform |
| Classes | `dataset_fingerprint.class_distribution` |
| Model Used | `planner_decisions.detector` |
| Confidence | `planner_decisions.confidence_threshold` |
| IoU | `environment["iou"]` |
| Frame Strategy | `environment["frame_strategy"]` |
| Segmentation | `planner_decisions.segmentation_enabled` |
| Review Level | `environment["review_level"]` |
| Export Format | `execution_metrics.export_format` (+ `planner_decisions.export_strategy`) |
| Execution Time | `execution_metrics.runtime_seconds` |
| Annotation Count | `environment["annotation_count"]` |
| Success | `environment["success"]` + `validation_status` |
| Warnings | `analyst_conclusions.bottlenecks` |

Only **completed** runs are stored (`record` returns `None` otherwise). `created_at`
is passed in by the caller (`DatasetEngineerAgent`) so the module stays
deterministic — same inputs, same record.

## Similarity

Unchanged from Phase 10. A query fingerprint is compared to every stored record by
a weighted average of per-feature agreement (resolution, dataset size, densities,
ratios) plus categorical `scene_type`. Unknown query features (sentinels) are
skipped, so a goal-only recall matches on the subset it knows — for a bare
"create thermal drone dataset", `scene_type` (`thermal_aerial`) does the work.
Every match carries `explain()` — the reasons it matched. Results are ranked
highest-first, ties broken newest-then-id (deterministic).

## Retrieval

`MemoryExperience` returns:

- **matches** — top-k similar `EngineeringMemory` records, each with score + reasons.
- **similarity_score** — the best match's score.
- **confidence** — mean similarity across matches.
- **successful_settings** — model, confidence, IoU, segmentation, frame strategy,
  review level, export format from the closest match.
- **recommendations** — the settings to reuse + validated engineering recommendations
  + a runtime estimate.
- **warnings** / **lessons** — bottlenecks and low-quality outcomes to avoid.

## GUI

`memory_view(experience)` returns the Memory Summary panel data (data only, no Qt —
same pattern as `decision_view`): `memory_summary`, `similar_projects`,
`lessons_learned`, `recommendations`, `previous_results`, `successful_settings`,
`confidence`, `similarity_score`.

## Example

```
User:  "Create thermal drone dataset"
        ▼
MemoryAgent.recall(...)
  Found 3 similar past project(s); closest mem_ab12… (similarity 1.0).
  Successful settings: model YOLO11s, confidence 0.20, frame every_2, review high.
  Recommendations: Use detection confidence 0.20; Estimated runtime ~18 min.
  Lessons: mem_cd34…: prior false-positive issue — raise confidence.
```

## Integration

`DatasetEngineerAgent` (the one component the GUI talks to) exposes:

- `recall_experience(goal, metadata?)` — before planning/decision.
- `record_experience(goal, decision_report, execution_summary, *, project_id, plan?, metadata?, export_summary?)`
  — after a completed run.
- `.memory` — the `MemoryAgent` for direct use / `memory_view`.

The PlannerAgent, DecisionAgent, and ExecutionAgent are **not modified** — memory is
a layer beside them, not a change to them.

## Architecture

```
GUI ─▶ DatasetEngineerAgent ─┬─ recall_experience ─▶ MemoryAgent.recall ─┐
                             └─ record_experience ─▶ MemoryAgent.record ─┤
                                                                         ▼
                              vds.memory  (MemoryStore · SimilarityEngine · EngineeringMemory)
                                                                         │
                                                          data/engineering_memory.json
```
