"""PlannerAgent — the first intelligent agent (V2-21).

A deterministic, senior-engineer planner: given a goal it decides *what* happens,
*in what order*, *what is missing*, *what to recommend*, and *what needs approval* —
and emits a fully explainable, serializable ``ExecutionPlan``. It **only thinks**:
no imports, no detection, no export, no tool execution. Execution is a future phase.

Pieces: ``GoalParser`` (intent) + ``RecommendationEngine`` (settings) + this module's
``ValidationEngine`` (reject impossible plans) + ``PlanSessionStore`` (create/load/
modify/approve/reject/export). All rule-based — no LLM, no MCP.
"""

from __future__ import annotations

from vds.v2.goal import Goal
from vds.v2.goal_parser import GoalParser, ParsedGoal
from vds.v2.planner import (
    ExecutionPlan,
    FrameStrategy,
    PlanStatus,
    PlanStep,
    RequiredInput,
    TaskType,
)
from vds.v2.recommendations import PlanContext, RecommendationEngine, RecommendationResult

# Step catalog: kind -> (id, title, agent, task, description, expected_output, approval).
# Agent names and tasks match the V2-20 registry/tools so the Execution Agent (V2-22)
# can run these steps unchanged.
_STEPS: dict[str, tuple[str, str, str, str, str, str, bool]] = {
    "analyse": ("analyse_inputs", "Analyse Inputs", "PlannerAgent", "analyze_input",
                "Understand the goal, modality, and target classes.", "Parsed goal + input summary", False),
    "inspect": ("inspect_dataset", "Inspect Dataset", "DatasetAnalysisAgent", "inspect_dataset",
                "Profile the existing/incoming dataset.", "Dataset statistics", False),
    "import_images": ("import_images", "Import Images", "ImportAgent", "import_images",
                      "Ingest images (dedup + CAS).", "Imported image set", False),
    "import_video": ("import_video", "Import Video", "ImportAgent", "import_video",
                     "Register and read the source video.", "Registered video source", False),
    "extract": ("extract_frames", "Extract Frames", "ImportAgent", "extract_frames",
                "Sample frames per the recommended strategy.", "Extracted frame images", False),
    "detect": ("run_detection", "Run Detection", "DetectionAgent", "run_detection",
               "Detect objects with the recommended model.", "Bounding boxes", False),
    "segment": ("run_segmentation", "Run Segmentation", "SegmentationAgent", "run_segmentation",
                "Generate masks for detected boxes.", "Instance masks", False),
    "quality": ("quality_review", "Quality Review", "QualityAgent", "review_dataset",
                "Score annotation quality / verdicts.", "Quality verdicts", False),
    "manual": ("manual_review", "Manual Review", "ReviewAgent", "await_approval",
               "Human approval before export.", "Approval decision", True),
    "export": ("export_dataset", "Export Dataset", "ExportAgent", "export_dataset",
               "Export in the requested format.", "Exported dataset", False),
    "report": ("generate_report", "Generate Report", "ExportAgent", "generate_report",
               "Render the run report.", "Markdown report", False),
}

# Step ids that produce/carry images — used by the validator.
_IMAGE_PRODUCERS = {"import_images", "extract_frames", "inspect_dataset"}


class PlanValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


def _select(parse: ParsedGoal) -> list[str]:
    """Choose the step kinds for this goal. Deterministic branch on task + modality."""
    t = parse.task_type
    kinds = ["analyse"]
    if t == TaskType.EXPORT:
        if parse.modality == "existing":
            kinds.append("inspect")
        return [*kinds, "manual", "export", "report"]
    if t == TaskType.REVIEW:
        return [*kinds, "inspect", "quality", "manual", "report"]
    # build tasks: detection / segmentation / classification / mixed / unknown
    if parse.modality == "video":
        # import_video runs the whole self-contained pipeline (extract -> detect ->
        # segment -> verify -> export) and creates the project, so it needs no separate
        # extract/detect/segment steps. The rest still runs against that project.
        return [*kinds, "import_video", "quality", "manual", "export", "report"]
    if parse.modality == "existing":
        kinds.append("inspect")
    else:  # images or unknown source -> assume an image folder
        kinds.append("import_images")
    kinds.append("detect")
    if t in (TaskType.SEGMENTATION, TaskType.MIXED):
        kinds.append("segment")
    return [*kinds, "quality", "manual", "export", "report"]


def _relink(steps: list[PlanStep]) -> None:
    """Rewire a linear dependency chain from current step order (keeps plans valid
    after add/remove edits)."""
    prev: str | None = None
    for s in steps:
        s.depends_on = [prev] if prev else []
        prev = s.id


class PlannerAgent:
    def __init__(self) -> None:
        self._parser = GoalParser()
        self._engine = RecommendationEngine()
        self._validator = ValidationEngine()

    # --- create --------------------------------------------------------
    def create_plan(
        self,
        goal: Goal,
        *,
        context: PlanContext | None = None,
        project: str | None = None,
        dataset: str | None = None,
        preferences: dict | None = None,
    ) -> ExecutionPlan:
        # Fold optional project/dataset/prefs into the goal params the parser reads.
        params = dict(goal.params)
        if project:
            params["project"] = project
        if dataset:
            params["dataset"] = dataset
        parse = self._parser.parse(goal.model_copy(update={"params": params}))
        recs = self._engine.recommend(parse, context)

        export_fmt = parse.export_format or "coco"
        steps = self._build_steps(parse, recs, export_fmt)
        plan = ExecutionPlan(
            goal_id=goal.id,
            goal_text=goal.text,
            steps=steps,
            task_type=parse.task_type,
            summary=self._summary(parse, steps),
            current_state=self._current_state(parse),
            required_inputs=self._required_inputs(parse),
            recommended_model=recs.model,
            recommended_segmentation=recs.segmentation,
            recommended_confidence=recs.confidence,
            recommended_iou=recs.iou,
            frame_strategy=recs.frame_strategy,
            estimated_dataset_size=recs.estimated_dataset_size,
            estimated_runtime_seconds=recs.estimated_runtime_seconds,
            estimated_review=recs.review_level,
            warnings=self._warnings(parse, recs),
            approvals_required=[s.name for s in steps if s.requires_approval],
            recommendations=recs.recommendations,
            alternatives=recs.alternatives,
            reasoning=self._reasoning(parse, recs),
        )
        if preferences:
            plan = self.modify(plan, **self._prefs_to_overrides(preferences))

        errors = self._validator.validate(plan, parse)
        if errors:
            raise PlanValidationError(errors)
        return plan

    def _build_steps(self, parse, recs, export_fmt: str) -> list[PlanStep]:
        det_args = {"model": recs.model, "confidence": recs.confidence, "iou": recs.iou}
        steps: list[PlanStep] = []
        for kind in _select(parse):
            sid, title, agent, task, desc, out, approval = _STEPS[kind]
            args: dict = {}
            if kind == "extract":
                args = {"frame_strategy": recs.frame_strategy.value}
            elif kind == "detect":
                args = dict(det_args)
            elif kind == "export":
                args = {"format": export_fmt}
            steps.append(PlanStep(
                id=sid, name=title, agent=agent, task=task, arguments=args,
                requires_approval=approval, description=desc, expected_output=out,
                reason=self._step_reason(kind, parse, recs)))
        _relink(steps)
        return steps

    # --- validation & modification ------------------------------------
    def validate(self, plan: ExecutionPlan, parse: ParsedGoal | None = None) -> list[str]:
        return self._validator.validate(plan, parse)

    def modify(
        self,
        plan: ExecutionPlan,
        *,
        model: str | None = None,
        confidence: float | None = None,
        iou: float | None = None,
        frame_strategy: str | None = None,
        segmentation: bool | None = None,
        export_format: str | None = None,
    ) -> ExecutionPlan:
        """Apply user edits, re-link, re-validate. A modified plan reverts to DRAFT."""
        p = plan.model_copy(deep=True)
        if model is not None:
            p.recommended_model = model
        if confidence is not None:
            p.recommended_confidence = confidence
        if iou is not None:
            p.recommended_iou = iou
        if frame_strategy is not None:
            p.frame_strategy = FrameStrategy(frame_strategy)
        if export_format is not None:
            for s in p.steps:
                if s.id == "export_dataset":
                    s.arguments["format"] = export_format
        for s in p.steps:  # keep detect args in sync with edited settings
            if s.id == "run_detection":
                s.arguments.update({"model": p.recommended_model,
                                    "confidence": p.recommended_confidence, "iou": p.recommended_iou})
        if segmentation is not None and segmentation != p.recommended_segmentation:
            self._toggle_segmentation(p, segmentation)
        p.status = PlanStatus.DRAFT
        errors = self._validator.validate(p)
        if errors:
            raise PlanValidationError(errors)
        return p

    def _toggle_segmentation(self, p: ExecutionPlan, on: bool) -> None:
        p.recommended_segmentation = on
        if on:
            if p.get("run_segmentation") is None and p.get("run_detection") is not None:
                sid, title, agent, task, desc, out, approval = _STEPS["segment"]
                idx = next(i for i, s in enumerate(p.steps) if s.id == "run_detection")
                p.steps.insert(idx + 1, PlanStep(
                    id=sid, name=title, agent=agent, task=task, requires_approval=approval,
                    description=desc, expected_output=out, reason="User enabled segmentation."))
        else:
            p.steps = [s for s in p.steps if s.id != "run_segmentation"]
        _relink(p.steps)

    # --- prose helpers -------------------------------------------------
    @staticmethod
    def _prefs_to_overrides(prefs: dict) -> dict:
        keys = ("model", "confidence", "iou", "frame_strategy", "segmentation", "export_format")
        return {k: prefs[k] for k in keys if k in prefs}

    @staticmethod
    def _summary(parse: ParsedGoal, steps: list[PlanStep]) -> str:
        cls = ", ".join(parse.target_classes) or "object"
        return f"{parse.task_type.value.title()} plan for '{cls}' from {parse.modality} ({len(steps)} steps)."

    @staticmethod
    def _current_state(parse: ParsedGoal) -> str:
        return {
            "existing": "Existing dataset provided; will inspect before any change.",
            "video": "Video source identified; frames not yet extracted.",
            "images": "Image source identified; not yet imported.",
        }.get(parse.modality, "No input source identified yet.")

    @staticmethod
    def _required_inputs(parse: ParsedGoal) -> list[RequiredInput]:
        if parse.modality == "video":
            return [RequiredInput(name="video source", provided=bool(parse.source))]
        if parse.modality == "existing":
            return [RequiredInput(name="existing dataset", provided=True, note="from project/dataset")]
        if parse.modality == "images":
            return [RequiredInput(name="image folder", provided=bool(parse.source))]
        return [RequiredInput(name="input source", provided=False, note="modality unresolved")]

    @staticmethod
    def _warnings(parse: ParsedGoal, recs: RecommendationResult) -> list[str]:
        warns = list(recs.warnings)
        if parse.task_type == TaskType.UNKNOWN:
            warns.append("Goal intent is unclear; defaulted to a detection pipeline.")
        if parse.modality in ("video", "images", "unknown") and not parse.source:
            warns.append("No input source provided; the plan cannot run until one is supplied.")
        return warns

    @staticmethod
    def _reasoning(parse: ParsedGoal, recs: RecommendationResult) -> str:
        bits = [f"Classified as {parse.task_type.value}.",
                f"Recommended {recs.model} at conf {recs.confidence:.2f}, IoU {recs.iou:.2f}."]
        if recs.frame_strategy != FrameStrategy.NONE:
            bits.append(f"Frame strategy: {recs.frame_strategy.value}.")
        bits.append(f"Estimated {recs.estimated_dataset_size} images, "
                    f"{recs.review_level.value} human review.")
        return " ".join(bits)

    @staticmethod
    def _step_reason(kind: str, parse: ParsedGoal, recs: RecommendationResult) -> str:
        return {
            "extract": f"Sample frames ({recs.frame_strategy.value}) to bound annotation cost.",
            "detect": f"Detect '{', '.join(parse.target_classes) or 'objects'}' with {recs.model}.",
            "segment": "Goal requires masks.",
            "manual": "No dataset is exported without human approval.",
            "inspect": "Understand the current data before changing it.",
        }.get(kind, "")


class ValidationEngine:
    """Reject impossible plans (V2-21 §VALIDATION). Returns meaningful errors."""

    def validate(self, plan: ExecutionPlan, parse: ParsedGoal | None = None) -> list[str]:
        errors: list[str] = []
        ids = [s.id for s in plan.steps]
        index = {sid: i for i, sid in enumerate(ids)}

        # Export before Import.
        imports = [i for sid, i in index.items() if sid in ("import_images", "import_video", "extract")]
        if "export_dataset" in index and imports and index["export_dataset"] < min(imports):
            errors.append("Export step precedes Import.")

        # Review/Quality without Detection (unless annotations already exist).
        modality = parse.modality if parse else ("existing" if "inspect_dataset" in index else "")
        # 'video' plans run detection inside the single import_video step (which does the
        # whole pipeline), so they legitimately have no standalone run_detection step.
        if ("quality_review" in index and "run_detection" not in index
                and modality not in ("existing", "video")
                and "import_video" not in index):
            errors.append("Quality Review without a Detection step (no annotations to review).")

        # Segmentation without images.
        if "run_segmentation" in index and not (_IMAGE_PRODUCERS & set(ids)):
            errors.append("Segmentation without any image source.")

        # Video frame strategy on non-video input. A video plan carries the strategy on
        # either a standalone extract_frames step or the self-contained import_video step.
        if plan.frame_strategy != FrameStrategy.NONE and not (
                {"extract_frames", "import_video"} & set(ids)):
            errors.append("Video frame strategy set for a non-video input.")

        # Dependency integrity.
        for s in plan.steps:
            for dep in s.depends_on:
                if dep not in index:
                    errors.append(f"Step '{s.id}' depends on unknown step '{dep}'.")
        return errors


class PlanSessionStore:
    """In-memory plan sessions (V2-21 §SESSION SUPPORT). Serialization is the plan's
    own pydantic JSON, so restore is provider-free."""

    def __init__(self) -> None:
        self._plans: dict[str, ExecutionPlan] = {}

    def create(self, plan: ExecutionPlan) -> ExecutionPlan:
        self._plans[plan.id] = plan
        return plan

    def load(self, plan_id: str) -> ExecutionPlan:
        return self._plans[plan_id]

    def save(self, plan: ExecutionPlan) -> ExecutionPlan:
        self._plans[plan.id] = plan
        return plan

    def approve(self, plan_id: str) -> ExecutionPlan:
        self._plans[plan_id].status = PlanStatus.APPROVED
        return self._plans[plan_id]

    def reject(self, plan_id: str) -> ExecutionPlan:
        self._plans[plan_id].status = PlanStatus.REJECTED
        return self._plans[plan_id]

    def export(self, plan_id: str) -> str:
        return self._plans[plan_id].model_dump_json(indent=2)

    def list(self) -> list[str]:
        return list(self._plans)

    @staticmethod
    def restore(data: str) -> ExecutionPlan:
        return ExecutionPlan.model_validate_json(data)


def plan_view(plan: ExecutionPlan) -> dict:
    """GUI data surface (V2-21 §GUI): everything the Plan Viewer binds to. No Qt —
    a future phase renders this. Follows the V2-20 defer-Qt decision."""
    return {
        "goal": plan.goal_text,
        "summary": plan.summary,
        "task_type": plan.task_type.value,
        "current_state": plan.current_state,
        "status": plan.status.value,
        "required_inputs": [ri.model_dump() for ri in plan.required_inputs],
        "recommended": {
            "model": plan.recommended_model,
            "segmentation": plan.recommended_segmentation,
            "confidence": plan.recommended_confidence,
            "iou": plan.recommended_iou,
            "frame_strategy": plan.frame_strategy.value,
        },
        "timeline": [
            {"id": s.id, "title": s.name, "agent": s.agent, "status": s.status.value,
             "reason": s.reason, "expected_output": s.expected_output}
            for s in plan.steps
        ],
        "recommendations": [r.model_dump() for r in plan.recommendations],
        "alternatives": [a.model_dump() for a in plan.alternatives],
        "warnings": list(plan.warnings),
        "estimated_runtime_seconds": plan.estimated_runtime_seconds,
        "estimated_dataset_size": plan.estimated_dataset_size,
        "estimated_review": plan.estimated_review.value,
        "approvals_required": list(plan.approvals_required),
        "reasoning": plan.reasoning,
    }
