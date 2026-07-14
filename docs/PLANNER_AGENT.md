# PlannerAgent (V2-21)

The first *intelligent* agent in the Version 2 stack. Given a user goal it produces
an explainable, serializable `ExecutionPlan`: what happens, in what order, what is
missing, what to recommend, and what needs human approval. It **only thinks** — no
imports, no detection, no export, no tool execution. Execution is a future phase.

It is fully deterministic and rule-based: **no LLM, no MCP**. Same goal + context →
same plan. It builds on V2-20 (registry, tools, `ExecutionPlan`) without modifying
it — the plan's steps reuse the V2-20 agent names and tasks so a future Execution
Agent can run them unchanged.

## Lifecycle

```
Goal ─▶ GoalParser ─▶ ParsedGoal ─▶ RecommendationEngine ─▶ RecommendationResult
                                                                   │
                         step selection (_select) ◀────────────────┘
                                   │
                                   ▼
                          ExecutionPlan (steps + reasoning + estimates)
                                   │
                          ValidationEngine.validate()  ── errors? ─▶ PlanValidationError
                                   │ ok
                                   ▼
                        PlanSessionStore: create / load / modify / approve / reject / export
```

## Planning algorithm

1. **Parse** (`goal_parser.py`) — keyword rules classify the `TaskType`
   (detection / segmentation / classification / review / export / mixed) and detect
   input modality (video / images / existing), thermal/drone hints, target classes,
   and export format.
2. **Recommend** (`recommendations.py`) — rule-based engine picks model, segmentation,
   confidence, IoU, frame strategy, dedup, dataset-size and runtime estimates, and a
   review level. Each decision is a `Recommendation` (reason / impact / confidence /
   alternative) — no hidden reasoning.
3. **Select steps** (`planner_agent._select`) — deterministic branch on task + modality
   chooses the pipeline: e.g. video-detection → Analyse → Import Video → Extract Frames
   → Run Detection → Quality Review → **Manual Review** → Export → Generate Report.
   Steps are linearly linked (`depends_on`).
4. **Validate** (`ValidationEngine`) — reject impossible plans; return meaningful errors.
5. The plan is returned; the caller can store, modify, approve/reject, and export it.

## ExecutionPlan schema

`ExecutionPlan` (in `vds/v2/planner.py`) was extended additively over V2-20 — every
new field defaults, so V2-20 template plans stay valid.

| Field | Meaning |
|-------|---------|
| `goal_id`, `goal_text`, `task_type` | the goal and its classification |
| `summary`, `current_state`, `reasoning` | human-readable narrative |
| `required_inputs` | `RequiredInput(name, provided, note)` — what's needed/missing |
| `steps` | ordered `PlanStep`s |
| `recommended_model`, `recommended_segmentation`, `recommended_confidence`, `recommended_iou` | model/settings recommendations |
| `frame_strategy` | `FrameStrategy` (none / every_2 / every_5 / every_10 / scene_change / adaptive) |
| `estimated_dataset_size`, `estimated_runtime_seconds`, `estimated_review` | estimates |
| `warnings`, `approvals_required` | risks and gates |
| `recommendations`, `alternatives` | `Recommendation` list + `Alternative` tradeoffs |
| `status` | `PlanStatus` (draft / approved / rejected) |

`PlanStep`: `id`, `name` (Title), `agent` (Responsible Agent), `task`, `arguments`,
`depends_on` (Dependencies), `status`, `requires_approval`, `description`,
`expected_output`, `reason`.

The plan is pure pydantic — `model_dump_json()` / `model_validate_json()` round-trip
it (including nested recommendations), which is what `PlanSessionStore.export` /
`restore` use.

## Recommendation engine

Rule highlights (all overridable later by the Detection Agent):

- **Model** — segmentation → `YOLO11-seg`; high-res / small objects → `YOLO11m`;
  otherwise `YOLO11s`. Always emits an `Alternative` with its tradeoff.
- **Confidence** — thermal → `0.20` (low contrast), else `0.30` (detector floor).
- **Frame strategy** — video only: dense → every 2; long/sparse → every 10; else every 5.
- **Dedup** — recommended once the estimated dataset exceeds ~2000 images.
- **Review level** — bumped by segmentation, thermal, or many classes.

## Validation rules

Rejected as impossible (`ValidationEngine`):

- Export step before any Import step.
- Quality Review without a Detection step (and no existing annotations).
- Segmentation without any image source.
- A video frame strategy set for non-video input.
- A step depending on an unknown step id.

## Examples

```python
from vds.v2 import PlannerAgent, PlanContext, new_goal, plan_view

agent = PlannerAgent()
plan = agent.create_plan(
    new_goal("Create a vehicle detection dataset from this highway video", source="hwy.mp4"),
    context=PlanContext(video_duration_seconds=120, fps=30, expected_density="high"),
)
plan.recommended_model      # 'YOLO11s'
plan.frame_strategy.value   # 'every_2'  (dense scene)
plan.approvals_required     # ['Manual Review']

edited = agent.modify(plan, segmentation=True, confidence=0.4)  # re-validated, back to DRAFT
view = plan_view(plan)      # dict for the GUI Plan Viewer
```

Session support:

```python
from vds.v2 import PlanSessionStore
store = PlanSessionStore()
store.create(plan)
store.approve(plan.id)              # PlanStatus.APPROVED
data = store.export(plan.id)        # JSON
restored = PlanSessionStore.restore(data)
```

## GUI integration (this phase)

Only the data surface: `plan_view(plan)` returns everything the Plan Viewer binds to
(goal, summary, timeline, recommendations, warnings, estimated runtime/review,
required approvals, alternatives). The GUI's single entry is
`DatasetEngineerAgent.generate_plan(goal, **context)` — plan-only, nothing executes.
No Qt widgets and no existing pages changed, matching the V2-20 defer-Qt decision.

## Rules (what the Planner never does)

Run detection or segmentation, modify projects, execute tools, import or export data.
The Planner only reasons; future phases execute the plan.
