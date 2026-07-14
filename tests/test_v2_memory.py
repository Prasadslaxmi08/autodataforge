"""V2-24 MemoryAgent — recall + storage over the shared Engineering Memory.

Covers: storage (into the existing store), similarity search + ranking,
serialization, GUI rendering, and execution integration. The MemoryAgent never
plans and never executes — it only remembers.
"""

from __future__ import annotations

from vds.memory.store import MemoryStore
from vds.v2 import (
    DatasetMetadata,
    DecisionAgent,
    MemoryAgent,
    MemoryExperience,
    PlannerAgent,
    new_goal,
)
from vds.v2.execution import ExecutionSummary
from vds.v2.memory_agent import memory_view


def _agent(tmp_path):
    return MemoryAgent(store=MemoryStore(tmp_path / "mem.json"))


def _run(text, metadata, *, status="completed", failed=0, errors=None):
    """Plan -> decide -> a completed-looking ExecutionSummary, so record() has real
    V2 inputs to store."""
    plan = PlannerAgent().create_plan(new_goal(text, source="src/"))
    enriched, report = DecisionAgent().decide(plan, metadata)
    summary = ExecutionSummary(
        plan_id=enriched.id, status=status, total=len(enriched.steps),
        completed=len(enriched.steps), skipped=0, failed=failed, cancelled=0, retried=0,
        elapsed_seconds=123.0, errors=list(errors or []))
    return new_goal(text, source="src/"), enriched, report, summary


# --- storage ----------------------------------------------------------
def test_record_stores_into_shared_engineering_memory(tmp_path):
    agent = _agent(tmp_path)
    goal, plan, report, summary = _run("create thermal drone vehicle dataset",
                                       DatasetMetadata(image_count=800, resolution="high"))
    mem = agent.record(goal, report, summary, project_id="p1", created_at="2026-07-14T00:00:00Z", plan=plan)
    assert mem is not None and mem.source == "memory_agent.v2"
    # persisted and reloadable through the *existing* MemoryStore
    stored = MemoryStore(tmp_path / "mem.json").all()
    assert [m.id for m in stored] == [mem.id]
    # V2-only fields ride in the environment bag (schema untouched)
    env = mem.dataset_fingerprint.environment
    assert env["frame_strategy"] and env["review_level"] and env["success"] == "true"
    assert mem.dataset_fingerprint.scene_type == "thermal_aerial"


def test_incomplete_run_is_not_recorded(tmp_path):
    agent = _agent(tmp_path)
    goal, plan, report, summary = _run("detect cars", DatasetMetadata(image_count=100), status="cancelled")
    assert agent.record(goal, report, summary, project_id="p", created_at="t", plan=plan) is None


def test_duplicate_record_suppressed_by_shared_store(tmp_path):
    agent = _agent(tmp_path)
    goal, plan, report, summary = _run("detect cars", DatasetMetadata(image_count=100))
    a = agent.record(goal, report, summary, project_id="p", created_at="2026-07-14T00:00:00Z", plan=plan)
    b = agent.record(goal, report, summary, project_id="p", created_at="2026-07-14T09:99:99Z", plan=plan)
    assert a.id == b.id  # identical run -> store dedups on content hash
    assert len(MemoryStore(tmp_path / "mem.json").all()) == 1


# --- similarity search + ranking --------------------------------------
def test_recall_finds_and_ranks_similar_projects(tmp_path):
    agent = _agent(tmp_path)
    # two thermal-drone jobs (should match) + one rgb ground job (should not)
    for i, (text, meta) in enumerate([
        ("thermal drone vehicle dataset", DatasetMetadata(image_count=800, resolution="high")),
        ("thermal aerial person dataset", DatasetMetadata(image_count=780, resolution="high")),
        ("detect cars from street images", DatasetMetadata(image_count=50, resolution="low")),
    ]):
        g, p, r, s = _run(text, meta)
        agent.record(g, r, s, project_id=f"p{i}", created_at=f"2026-07-14T0{i}:00:00Z", plan=p)

    exp = agent.recall(new_goal("create thermal drone dataset", source="x/"),
                       DatasetMetadata(image_count=800, resolution="high"))
    assert exp.has_experience
    assert all(m.memory.dataset_fingerprint.scene_type == "thermal_aerial" for m in exp.matches)
    # ranked highest-first, and street/rgb job filtered out below the floor
    scores = [m.score for m in exp.matches]
    assert scores == sorted(scores, reverse=True)
    assert exp.similarity_score == scores[0]
    assert 0.0 < exp.confidence <= 1.0
    assert exp.recommendations and exp.successful_settings["model"]


def test_recall_no_experience_is_explicit(tmp_path):
    exp = _agent(tmp_path).recall(new_goal("segment satellites in space imagery"))
    assert not exp.has_experience
    assert exp.matches == [] and exp.similarity_score == 0.0
    assert "No similar" in exp.note


# --- serialization ----------------------------------------------------
def test_experience_roundtrip(tmp_path):
    agent = _agent(tmp_path)
    g, p, r, s = _run("thermal drone dataset", DatasetMetadata(image_count=800, resolution="high"))
    agent.record(g, r, s, project_id="p", created_at="2026-07-14T00:00:00Z", plan=p)
    exp = agent.recall(new_goal("thermal drone dataset"), DatasetMetadata(resolution="high"))
    restored = MemoryExperience.model_validate_json(exp.model_dump_json())
    assert restored.similarity_score == exp.similarity_score
    assert [m.memory.id for m in restored.matches] == [m.memory.id for m in exp.matches]


# --- GUI rendering ----------------------------------------------------
def test_memory_view_surface(tmp_path):
    agent = _agent(tmp_path)
    g, p, r, s = _run("thermal drone dataset", DatasetMetadata(image_count=800, resolution="high"))
    agent.record(g, r, s, project_id="p", created_at="2026-07-14T00:00:00Z", plan=p)
    v = memory_view(agent.recall(new_goal("thermal drone dataset"), DatasetMetadata(resolution="high")))
    for key in ("memory_summary", "similar_projects", "lessons_learned", "recommendations",
                "previous_results", "successful_settings", "confidence"):
        assert key in v
    assert v["similar_projects"] and v["similar_projects"][0]["goal"]
    assert v["previous_results"][0]["success"] == "true"


# --- execution integration (via the master agent) ---------------------
def test_dataset_engineer_recall_then_record(tmp_path, monkeypatch):
    class _FakeController:
        def __getattr__(self, name):
            return lambda *a, **k: None

    from vds.v2 import DatasetEngineerAgent

    eng = DatasetEngineerAgent(_FakeController())  # type: ignore[arg-type]
    # point the master agent's memory at a tmp store (don't touch the real file)
    eng.memory._store = MemoryStore(tmp_path / "mem.json")

    goal = new_goal("create thermal drone dataset", source="v.mp4")
    plan = eng.generate_plan(goal)
    enriched, report = eng.optimize_plan(plan, {"image_count": 800, "resolution": "high"})
    summary = ExecutionSummary(
        plan_id=enriched.id, status="completed", total=len(enriched.steps),
        completed=len(enriched.steps), skipped=0, failed=0, cancelled=0, retried=0, elapsed_seconds=90.0)

    mem = eng.record_experience(goal, report, summary, project_id="proj", plan=enriched,
                                metadata={"image_count": 800, "resolution": "high"})
    assert mem is not None
    exp = eng.recall_experience(new_goal("thermal drone dataset"), {"resolution": "high"})
    assert exp.has_experience and exp.matches[0].memory.id == mem.id
