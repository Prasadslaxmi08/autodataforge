# DecisionAgent (V2-23)

The optimization layer that sits **between** the PlannerAgent and the ExecutionAgent.
The Planner recommends execution parameters from the user's *intent* (goal text). The
DecisionAgent refines them using *real dataset metadata and history* the Planner never
saw, then emits a `DecisionReport` and an **enriched** `ExecutionPlan` that the
ExecutionAgent runs unchanged.

It **only optimizes execution**. It never changes the user's intent (same steps, same
task, same classes), never runs models or tools, never touches the backend, and never
mutates the Planner's plan — it enriches a deep copy. It does **not** re-derive the
Planner's intent logic; it starts from the plan's baseline recommendations and adjusts
them from metadata.

## Decision flow

```
PlannerAgent ─▶ ExecutionPlan (intent-based recommendations)
                     │            + DatasetMetadata (file types, counts, resolution,
                     ▼              existing classes, historical stats, previous exports)
              DecisionAgent.decide(plan, metadata)
                     │
                     ├─▶ Decision per area (value + reason + confidence% + alternative + impact + tradeoffs)
                     ├─▶ enriched ExecutionPlan  (deep copy; params written into recommended_* + step args)
                     └─▶ DecisionReport          (decisions + warnings + suggestions + estimates)
                     │
   user Accept / Reject / Override ─▶ apply_overrides(plan, report, {area: value})
                     ▼
              approve ─▶ ExecutionAgent.execute(enriched plan)
```

## Decision areas

Frame Sampling · Detection Confidence · IoU Threshold · Segmentation · Export Format ·
Review Level · Batch Size · Compute (CPU vs GPU) · Duplicate Removal · Expected Runtime ·
Estimated Annotation Count.

Each is a `Decision` carrying **value, reason, confidence (0–1, shown as %), alternative,
impact, and tradeoffs** — no hidden reasoning.

## Recommendation model

Refinements are metadata-driven (baseline = the plan's existing recommendation):

| Area | Refined from |
|------|--------------|
| Detection Confidence | thermal file types → 0.20; high historical false-positive rate → 0.45; else plan baseline |
| Frame Sampling | video duration + fps + historical density → every 2 / 5 / 10 |
| IoU | historical objects-per-image (crowded → 0.60) |
| Export Format | previous exports (match them); else COCO+YOLO when segmenting; else plan default |
| Review Level | historical review rate + thermal/segmentation complexity |
| Batch Size | resolution + compute device |
| Compute | expected count (> 500 → GPU) |
| Duplicate Removal | video or large count (> 2000) |
| Expected Runtime | expected count × per-image time (device-adjusted) |
| Annotation Count | expected count × historical objects-per-image |

`DatasetMetadata` (input): `file_types`, `image_count`, `video_duration_seconds`, `fps`,
`resolution` (`"low"/"medium"/"high"` or `"1920x1080"`), `existing_classes`,
`historical_stats` (`avg_objects_per_image`, `avg_review_rate`, `false_positive_rate`),
`previous_exports`. All optional.

## Decision examples

```python
from vds.v2 import DecisionAgent, DatasetMetadata, PlannerAgent, new_goal

plan = PlannerAgent().create_plan(new_goal("detect cars from drone video", source="d.mp4"))
enriched, report = DecisionAgent().decide(
    plan, DatasetMetadata(video_duration_seconds=20, fps=30))       # short, dense
report.get("frame_sampling").value        # 'every_2'
enriched.estimated_dataset_size           # 20*30/2 = 300
report.expected_annotation_count          # 300 * historical objects/image
```

Thermal imagery lowers the confidence threshold:

```python
_, report = DecisionAgent().decide(plan, DatasetMetadata(image_count=800, file_types=["thermal"]))
report.get("detection_confidence").value        # '0.20'
report.get("detection_confidence").confidence   # 0.92
report.recommended_review                        # 'high'
```

User override (from the GUI Accept/Reject/Override actions):

```python
p2, r2 = DecisionAgent().apply_overrides(enriched, report, {"detection_confidence": 0.5})
p2.recommended_confidence                          # 0.5
p2.get("run_detection").arguments["confidence"]    # 0.5
r2.get("detection_confidence").reason              # 'User override.'
```

## How ExecutionAgent consumes recommendations

The DecisionAgent writes decided parameters into the enriched plan:

- `recommended_confidence` / `recommended_iou` / `recommended_segmentation` /
  `frame_strategy` / `estimated_*` on the plan, and
- the relevant **step arguments**: `run_detection` gets `confidence`, `iou`,
  `batch_size`, `device`; `export_dataset` gets `format`; `extract_frames` gets
  `frame_strategy`; import steps get `dedup`.

The ExecutionAgent (V2-22) runs the enriched plan exactly as it runs any plan — it reads
`step.arguments` when invoking tools, so the optimized parameters flow through with **no
ExecutionAgent change**.

## GUI integration (this phase)

Data surface only: `decision_view(report)` returns the Decision Summary panel data
(recommendations, warnings, suggestions, estimated runtime, expected annotation count,
review level, per-decision reasoning, overall confidence). The user's Accept / Reject /
Override maps to `apply_overrides`. No Qt widgets and no existing pages changed —
matching the V2-20…V2-22 defer-Qt decision. Master-agent entry:
`DatasetEngineerAgent.optimize_plan(plan, metadata)`.

## Rules

The DecisionAgent never runs models, executes tools, imports/exports data, modifies the
BackendController, or duplicates Planner logic. Decision layer only.
