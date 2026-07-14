"""Memory evolution, trend analysis, and engineering reports (Phase 10).

Deterministic aggregation over the stored memories — no AI, no reasoning. Answers
"what improved over time?", "which strategies work?", "which thresholds are most
effective?" straight from the recorded numbers, and renders the engineering
reports the phase asks for.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from pydantic import BaseModel

from vds.memory.schema import EngineeringMemory


class Trend(BaseModel):
    metric: str
    series: list[float]
    first: float
    last: float
    delta: float
    improved: bool


class StrategyStat(BaseModel):
    strategy: str  # e.g. "detector=yolo,seg=True"
    runs: int
    avg_quality: float
    avg_review_rate: float
    avg_throughput: float


def _series(values: list[float]) -> list[float]:
    return [round(v, 4) for v in values]


def _trend(metric: str, values: list[float], higher_is_better: bool) -> Trend:
    vals = _series(values)
    first, last = (vals[0], vals[-1]) if vals else (0.0, 0.0)
    delta = round(last - first, 4)
    improved = (delta >= 0) if higher_is_better else (delta <= 0)
    return Trend(metric=metric, series=vals, first=first, last=last, delta=delta, improved=improved)


class TrendAnalyzer:
    """All queries take the full memory list (oldest first) and are pure."""

    def evolution(self, memories: list[EngineeringMemory]) -> dict[str, Trend]:
        ms = sorted(memories, key=lambda m: (m.created_at, m.version))
        return {
            "throughput": _trend("throughput_ips", [m.benchmark_summary.throughput_ips for m in ms], True),
            "review_rate": _trend("review_rate", [m.execution_metrics.review_rate for m in ms], False),
            "quality": _trend("quality_score", [m.benchmark_summary.quality_score for m in ms], True),
            "avg_confidence": _trend("avg_confidence", [m.benchmark_summary.avg_confidence for m in ms], True),
        }

    def strategies(self, memories: list[EngineeringMemory]) -> list[StrategyStat]:
        groups: dict[str, list[EngineeringMemory]] = defaultdict(list)
        for m in memories:
            d = m.planner_decisions
            key = f"detector={d.detector},seg={d.segmentation_enabled},conf={d.confidence_threshold},tiling={d.tiling}"
            groups[key].append(m)
        stats = [
            StrategyStat(
                strategy=key, runs=len(g),
                avg_quality=round(sum(m.benchmark_summary.quality_score for m in g) / len(g), 4),
                avg_review_rate=round(sum(m.execution_metrics.review_rate for m in g) / len(g), 4),
                avg_throughput=round(sum(m.benchmark_summary.throughput_ips for m in g) / len(g), 4),
            )
            for key, g in groups.items()
        ]
        stats.sort(key=lambda s: (s.avg_quality, -s.avg_review_rate), reverse=True)
        return stats

    def effective_thresholds(self, memories: list[EngineeringMemory]) -> list[tuple[float, float, float]]:
        """(confidence_threshold, avg_quality, avg_review_rate), best quality first."""
        groups: dict[float, list[EngineeringMemory]] = defaultdict(list)
        for m in memories:
            groups[m.planner_decisions.confidence_threshold].append(m)
        rows = [
            (thr,
             round(sum(m.benchmark_summary.quality_score for m in g) / len(g), 4),
             round(sum(m.execution_metrics.review_rate for m in g) / len(g), 4))
            for thr, g in groups.items()
        ]
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows

    def common_problems(self, memories: list[EngineeringMemory]) -> list[tuple[str, int]]:
        c: Counter = Counter()
        for m in memories:
            for name, count in m.verification_outcomes.common_semantic_failures.items():
                c[name] += count
        return c.most_common()

    # --- reports ---
    def trend_report(self, memories: list[EngineeringMemory]) -> str:
        if not memories:
            return "# Engineering Memory — Trend Report\n\n_no memories recorded yet_\n"
        ev = self.evolution(memories)
        lines = [f"# Engineering Memory — Trend Report ({len(memories)} runs)", ""]
        for t in ev.values():
            arrow = "improved" if t.improved else "regressed"
            lines.append(f"- **{t.metric}**: {t.first} → {t.last} (Δ{t.delta:+}, {arrow})")
        return "\n".join(lines) + "\n"

    def engineering_report(self, memories: list[EngineeringMemory]) -> str:
        if not memories:
            return "# Engineering Report\n\n_no memories recorded yet_\n"
        strategies = self.strategies(memories)
        thresholds = self.effective_thresholds(memories)
        problems = self.common_problems(memories)
        ev = self.evolution(memories)
        best = strategies[0]
        lines = [
            f"# Engineering Report ({len(memories)} runs)", "",
            "## Most Successful Planner Strategies",
            *[f"- `{s.strategy}` — quality {s.avg_quality}, review {s.avg_review_rate}, "
              f"throughput {s.avg_throughput} ({s.runs} runs)" for s in strategies[:5]],
            "",
            f"Best strategy so far: **{best.strategy}** (quality {best.avg_quality}).",
            "",
            "## Most Effective Confidence Thresholds",
            *[f"- threshold {thr}: quality {q}, review {rr}" for thr, q, rr in thresholds[:5]],
            "",
            "## Most Common Dataset / Verification Problems",
            *([f"- {name}: {count}" for name, count in problems[:8]] or ["- none recorded"]),
            "",
            "## Performance & Review-Rate Evolution",
            f"- throughput: {ev['throughput'].first} → {ev['throughput'].last} (Δ{ev['throughput'].delta:+})",
            f"- review rate: {ev['review_rate'].first} → {ev['review_rate'].last} (Δ{ev['review_rate'].delta:+})",
            f"- quality: {ev['quality'].first} → {ev['quality'].last} (Δ{ev['quality'].delta:+})",
        ]
        return "\n".join(lines) + "\n"
