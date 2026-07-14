"""Knowledge Center Markdown export (Phase 16) — presentation of stored knowledge.

Renders sections of the Knowledge Center to Markdown from already-measured / validated
data. The "Knowledge Report" and "Engineering Summary" reuse the existing Engineering
Memory reports verbatim; the rest reformat the view-model. PDF is produced by Qt from
this Markdown on the UI side. Nothing new is computed here.
"""

from __future__ import annotations

from vds.container import Container
from vds.gui import knowledge_view as kv


def _cards(container: Container) -> str:
    out = ["## Reusable Knowledge Cards"]
    cards = kv.knowledge_cards(container)
    if not cards:
        return "\n".join(out + ["- No matching knowledge exists yet."])
    for c in cards:
        out += [f"### {c.title}",
                f"- Occurrences: {c.occurrences}",
                f"- Historical Success Rate: {c.success_rate:.0%}",
                f"- Best Strategy: {c.best_strategy}",
                f"- Expected Improvement: {c.expected_improvement}",
                f"- Supporting Datasets: {', '.join(c.supporting_datasets)}",
                f"- Confidence: {c.confidence}"]
    return "\n".join(out)


def _lessons(container: Container) -> str:
    out = ["## Lessons Learned"]
    lessons = kv.lessons_learned(container)
    if not lessons:
        return "\n".join(out + ["- No validated lessons recorded yet."])
    for lsn in lessons:
        out += [f"### {lsn.solution}",
                f"- Problem: {lsn.problem}",
                f"- Root Cause: {lsn.root_cause}",
                f"- Recommended Solution: {lsn.solution}",
                f"- Supporting Evidence: {', '.join(lsn.evidence) or 'none'}",
                f"- Historical Occurrences: {lsn.occurrences}",
                f"- Expected Benefit: {lsn.expected_benefit}",
                f"- Confidence: {lsn.confidence}",
                f"- Reference Datasets: {', '.join(lsn.reference_datasets)}"]
    return "\n".join(out)


def _comparison(container: Container, ids: list[str]) -> str:
    cmp = kv.compare_records(container, ids)
    if not cmp.datasets:
        return "## Historical Comparison\n\n- Select at least one dataset to compare."
    header = "| Metric | " + " | ".join(cmp.datasets) + " |"
    sep = "| --- " * (len(cmp.datasets) + 1) + "|"
    lines = ["## Historical Comparison", "", header, sep]
    for r in cmp.rows:
        trend = f" _{r.trend}_" if r.trend else ""
        lines.append(f"| {r.metric}{trend} | " + " | ".join(r.values) + " |")
    return "\n".join(lines)


def to_markdown(container: Container, section: str, ids: list[str] | None = None) -> str:
    title = "# AutoDataForge — Knowledge Center\n"
    if section == "knowledge_report":
        return title + "\n" + container.memory.engineering_report()
    if section == "engineering_summary":
        return title + "\n" + container.memory.trend_report()
    if section == "lessons":
        return title + "\n" + _lessons(container) + "\n"
    if section == "comparison":
        return title + "\n" + _comparison(container, ids or []) + "\n"
    # full
    return "\n\n".join([title, _cards(container), _lessons(container),
                        container.memory.engineering_report()]) + "\n"
