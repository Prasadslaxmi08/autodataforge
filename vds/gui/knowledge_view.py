"""Knowledge Center view-model (Phase 16) — plain data over Engineering Memory.

No Qt here. It VISUALIZES the existing Engineering Memory: every value is read from a
stored `EngineeringMemory` record (measured pipeline outputs + validated Analyst
recommendations) or aggregated deterministically by the existing `TrendAnalyzer`. It
computes no new knowledge and fabricates nothing — when memory is empty, callers get
empty lists and the page says so explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vds.container import Container
from vds.memory.schema import DatasetFingerprint, EngineeringMemory
from vds.memory.trends import TrendAnalyzer

# --- dataclasses -----------------------------------------------------------


@dataclass
class KnowledgeRecord:
    """One stored engineering record — a Dataset-History row and a search hit."""

    id: str
    dataset: str
    project_id: str
    created_at: str
    version: int
    image_count: int
    scene_type: str
    detector: str
    planner_strategy: str
    review_rate: float
    runtime_seconds: float
    health: int  # 0..100 from benchmark quality_score
    status: str  # validation_status
    classes: list[str]
    recommendations: list[str]  # recommendation actions


@dataclass
class KnowledgeCard:
    title: str
    occurrences: int
    success_rate: float  # avg quality_score 0..1
    best_strategy: str
    expected_improvement: str
    supporting_datasets: list[str]
    confidence: float


@dataclass
class TimelineEvent:
    date: str
    kind: str
    detail: str


@dataclass
class ComparisonRow:
    metric: str
    values: list[str]
    trend: str  # "improved" | "regressed" | ""


@dataclass
class Comparison:
    datasets: list[str]
    rows: list[ComparisonRow]


@dataclass
class Lesson:
    problem: str
    root_cause: str
    solution: str
    evidence: list[str]
    occurrences: int
    expected_benefit: str
    confidence: float
    reference_datasets: list[str]


@dataclass
class FilterOptions:
    datasets: list[str] = field(default_factory=list)
    versions: list[int] = field(default_factory=list)
    scene_types: list[str] = field(default_factory=list)
    detectors: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)


# --- helpers ---------------------------------------------------------------
def _names(container: Container) -> dict[str, str]:
    return {p.id: p.name for p in container.projects.list()}


def _strategy(m: EngineeringMemory) -> str:
    d = m.planner_decisions
    return f"{d.detector}, conf={d.confidence_threshold}, seg={d.segmentation_enabled}, tiling={d.tiling}"


def _record(m: EngineeringMemory, names: dict[str, str]) -> KnowledgeRecord:
    d = m.planner_decisions
    return KnowledgeRecord(
        id=m.id, dataset=names.get(m.project_id, m.project_id[:8]), project_id=m.project_id,
        created_at=m.created_at, version=m.version, image_count=m.dataset_fingerprint.dataset_size,
        scene_type=m.dataset_fingerprint.scene_type, detector=d.detector,
        planner_strategy=_strategy(m), review_rate=m.execution_metrics.review_rate,
        runtime_seconds=m.execution_metrics.runtime_seconds,
        health=round(m.benchmark_summary.quality_score * 100), status=m.validation_status,
        classes=sorted(m.dataset_fingerprint.class_distribution),
        recommendations=[r.action for r in m.engineering_recommendations],
    )


def _sorted(memories: list[EngineeringMemory]) -> list[EngineeringMemory]:
    return sorted(memories, key=lambda m: (m.created_at, m.version))


# --- section 1/2: search + history -----------------------------------------
_SEARCH_FIELDS = ("Dataset Name", "Scene Type", "Object Class", "Resolution",
                  "Planner Strategy", "Detector", "Recommendation", "Keyword")


def list_records(container: Container) -> list[KnowledgeRecord]:
    """Dataset History — every processed dataset, newest first."""
    names = _names(container)
    return [_record(m, names) for m in reversed(_sorted(container.memory.all()))]


def _matches(m: EngineeringMemory, names: dict[str, str], field: str, q: str) -> bool:
    fp = m.dataset_fingerprint
    if field == "Dataset Name":
        return q in names.get(m.project_id, m.project_id).lower() or q in m.project_id.lower()
    if field == "Scene Type":
        return q in fp.scene_type.lower()
    if field == "Object Class":
        return any(q in c.lower() for c in fp.class_distribution)
    if field == "Resolution":
        return q in f"{fp.resolution_mp}"
    if field == "Planner Strategy":
        return q in _strategy(m).lower()
    if field == "Detector":
        return q in m.planner_decisions.detector.lower()
    if field == "Recommendation":
        return any(q in (r.action + r.reason + r.expected_impact).lower()
                   for r in m.engineering_recommendations)
    # Keyword: full-text over the serialized record
    return q in m.model_dump_json().lower()


def search_records(container: Container, query: str, field: str = "Keyword") -> list[KnowledgeRecord]:
    q = query.strip().lower()
    names = _names(container)
    if not q:
        return list_records(container)
    hits = [m for m in reversed(_sorted(container.memory.all())) if _matches(m, names, field, q)]
    return [_record(m, names) for m in hits]


def search_fields() -> tuple[str, ...]:
    return _SEARCH_FIELDS


# --- section 3: knowledge cards --------------------------------------------
def _env_has(fp: DatasetFingerprint, words: set[str]) -> bool:
    blob = " ".join(list(fp.environment.values()) + [fp.scene_type]).lower()
    return any(w in blob for w in words)


def _imbalanced(dist: dict[str, int]) -> bool:
    total = sum(dist.values())
    return total > 0 and len(dist) >= 2 and max(dist.values()) / total >= 0.7


# ponytail: thresholds are heuristic "knobs" for grouping measured records into
# recognizable engineering themes; the occurrences/rates shown are all measured.
_CHARACTERISTICS: list[tuple[str, object]] = [
    ("Small Object Detection", lambda fp: fp.small_object_ratio >= 0.3),
    ("High Duplicate Rate", lambda fp: fp.duplicate_ratio >= 0.1),
    ("Dense Urban Scenes", lambda fp: fp.scene_density >= 8.0
     or fp.scene_type.lower() in {"street", "urban", "aerial"}),
    ("Thermal Imagery", lambda fp: _env_has(fp, {"thermal", "ir", "lwir", "infrared"})),
    ("Night Vision", lambda fp: _env_has(fp, {"night", "lowlight", "low-light", "dark"})),
    ("Class Imbalance", lambda fp: _imbalanced(fp.class_distribution)),
]


def knowledge_cards(container: Container) -> list[KnowledgeCard]:
    memories = container.memory.all()
    names = _names(container)
    cards: list[KnowledgeCard] = []
    for title, pred in _CHARACTERISTICS:
        subset = [m for m in memories if pred(m.dataset_fingerprint)]
        if not subset:
            continue
        recs = [r for m in subset for r in m.engineering_recommendations]
        best_rec = max(recs, key=lambda r: r.confidence, default=None)
        strategies = TrendAnalyzer().strategies(subset)
        cards.append(KnowledgeCard(
            title=title, occurrences=len(subset),
            success_rate=round(sum(m.benchmark_summary.quality_score for m in subset) / len(subset), 4),
            best_strategy=strategies[0].strategy if strategies else "—",
            expected_improvement=best_rec.expected_impact if best_rec else "See recommendations.",
            supporting_datasets=sorted({names.get(m.project_id, m.project_id[:8]) for m in subset}),
            confidence=round(sum(m.confidence for m in subset) / len(subset), 3),
        ))
    return cards


# --- section 4: timeline ---------------------------------------------------
def timeline(container: Container) -> list[TimelineEvent]:
    ms = _sorted(container.memory.all())
    names = _names(container)
    events: list[TimelineEvent] = []
    prev: EngineeringMemory | None = None
    for m in ms:
        ds = names.get(m.project_id, m.project_id[:8])
        events.append(TimelineEvent(m.created_at, "Dataset Processed",
                                    f"{ds} (v{m.version}) — health "
                                    f"{round(m.benchmark_summary.quality_score * 100)}, "
                                    f"detector {m.planner_decisions.detector}"))
        if prev is not None:
            if m.execution_metrics.review_rate < prev.execution_metrics.review_rate:
                events.append(TimelineEvent(m.created_at, "Review Reduced",
                    f"{prev.execution_metrics.review_rate:.0%} → {m.execution_metrics.review_rate:.0%}"))
            if m.benchmark_summary.quality_score > prev.benchmark_summary.quality_score:
                events.append(TimelineEvent(m.created_at, "Benchmark Improved",
                    f"quality {prev.benchmark_summary.quality_score} → {m.benchmark_summary.quality_score}"))
            if m.benchmark_summary.avg_confidence > prev.benchmark_summary.avg_confidence:
                events.append(TimelineEvent(m.created_at, "Verification Improved",
                    f"avg confidence {prev.benchmark_summary.avg_confidence:.2f} "
                    f"→ {m.benchmark_summary.avg_confidence:.2f}"))
            if m.planner_decisions != prev.planner_decisions:
                events.append(TimelineEvent(m.created_at, "Planner Updated", _strategy(m)))
        if m.engineering_recommendations:
            events.append(TimelineEvent(m.created_at, "Knowledge Added",
                f"{len(m.engineering_recommendations)} validated recommendation(s) from {ds}"))
        prev = m
    return events


# --- section 5: historical comparison --------------------------------------
def _trend_dir(values: list[float], higher_is_better: bool) -> str:
    if len(values) < 2:
        return ""
    improved = (values[-1] >= values[0]) if higher_is_better else (values[-1] <= values[0])
    return "improved" if improved else "regressed"


def compare_records(container: Container, ids: list[str]) -> Comparison:
    names = _names(container)
    by_id = {m.id: m for m in container.memory.all()}
    chosen = _sorted([by_id[i] for i in ids if i in by_id])
    if not chosen:
        return Comparison([], [])
    labels = [f"{names.get(m.project_id, m.project_id[:8])} v{m.version}" for m in chosen]

    def row(metric: str, fn, higher_is_better: bool, fmt=lambda v: f"{v}") -> ComparisonRow:
        raw = [fn(m) for m in chosen]
        return ComparisonRow(metric, [fmt(v) for v in raw],
                             _trend_dir([float(v) for v in raw], higher_is_better))

    rows = [
        ComparisonRow("Planner Strategy", [_strategy(m) for m in chosen], ""),
        row("Runtime (s)", lambda m: round(m.execution_metrics.runtime_seconds, 2), False),
        row("Review Rate", lambda m: m.execution_metrics.review_rate, False, lambda v: f"{v:.0%}"),
        row("Verification (avg conf)", lambda m: m.benchmark_summary.avg_confidence, True, lambda v: f"{v:.0%}"),
        row("Annotation Quality", lambda m: m.benchmark_summary.approval_rate, True, lambda v: f"{v:.0%}"),
        row("Throughput (img/s)", lambda m: round(m.benchmark_summary.throughput_ips, 2), True),
        row("Dataset Health", lambda m: round(m.benchmark_summary.quality_score * 100), True),
        ComparisonRow("Recommendations",
                      [str(len(m.engineering_recommendations)) for m in chosen], ""),
    ]
    return Comparison(labels, rows)


# --- section 6: lessons learned --------------------------------------------
def lessons_learned(container: Container) -> list[Lesson]:
    memories = container.memory.all()
    names = _names(container)
    groups: dict[tuple[str, str], list[tuple[EngineeringMemory, object]]] = {}
    for m in memories:
        for r in m.engineering_recommendations:  # only validated recs are stored
            groups.setdefault((r.action, r.target), []).append((m, r))
    lessons: list[Lesson] = []
    for (action, _target), pairs in groups.items():
        recs = [r for _m, r in pairs]
        mems = [m for m, _r in pairs]
        best = max(recs, key=lambda r: r.confidence)
        root_causes = [c for m in mems for c in m.analyst_conclusions.root_causes]
        lessons.append(Lesson(
            problem=best.reason,
            root_cause="; ".join(dict.fromkeys(root_causes)) or best.reason,
            solution=action,
            evidence=sorted({s for r in recs for s in r.supporting_metrics}),
            occurrences=len(pairs),
            expected_benefit=best.expected_impact,
            confidence=round(sum(r.confidence for r in recs) / len(recs), 3),
            reference_datasets=sorted({names.get(m.project_id, m.project_id[:8]) for m in mems}),
        ))
    lessons.sort(key=lambda x: (x.occurrences, x.confidence), reverse=True)
    return lessons


# --- filter options --------------------------------------------------------
def filter_options(container: Container) -> FilterOptions:
    ms = container.memory.all()
    names = _names(container)
    return FilterOptions(
        datasets=sorted({names.get(m.project_id, m.project_id[:8]) for m in ms}),
        versions=sorted({m.version for m in ms}),
        scene_types=sorted({m.dataset_fingerprint.scene_type for m in ms}),
        detectors=sorted({m.planner_decisions.detector for m in ms}),
        strategies=sorted({_strategy(m) for m in ms}),
    )
