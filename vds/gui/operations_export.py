"""Operations Center Markdown export (Phase 17) — presentation of measured ops data.

The Benchmark Report and Historical Trends reuse the existing Engineering Memory
reports verbatim; the Operations Report and Performance Summary reformat the
view-model (measured KPIs, system stats, platform health). Nothing new is computed;
PDF is produced by Qt from this Markdown on the UI side.
"""

from __future__ import annotations

from vds.container import Container
from vds.gui import operations_view as ov


def _operations(container: Container, live: dict) -> str:
    kpis = ov.executive_overview(container, live)
    health = ov.platform_health(container, live)
    system = ov.system_performance(container, live)
    lines = ["## Executive Operations Overview",
             *[f"- {k.label}: {k.value}" + (f" ({k.sub})" if k.sub else "") for k in kpis],
             "", f"## Platform Health — {health.status}",
             *[f"- [{i.status}] {i.name}: {i.detail}" for i in health.indicators]]
    if health.root_causes:
        lines += ["", "### Root Causes", *[f"- {rc}" for rc in health.root_causes]]
    lines += ["", "## System Performance",
              *[f"- {s.name}: {s.value}" for s in system]]
    return "\n".join(lines)


def _performance_summary(container: Container, live: dict) -> str:
    trends = ov.historical_trends(container)
    lines = ["## Performance Summary"]
    if not trends:
        return "\n".join(lines + ["- No benchmark runs recorded yet."])
    for t in trends:
        arrow = "improved" if t.improved else "regressed"
        lines.append(f"- **{t.metric}**: latest {t.last} (Δ{t.delta:+}, {arrow})")
    return "\n".join(lines)


def to_markdown(container: Container, section: str, live: dict | None = None) -> str:
    live = live or ov.live_snapshot()
    title = "# AutoDataForge — Operations & Performance Center\n"
    if section == "benchmark":
        return title + "\n" + container.memory.engineering_report()
    if section == "trends":
        return title + "\n" + container.memory.trend_report()
    if section == "performance_summary":
        return title + "\n" + _performance_summary(container, live) + "\n"
    if section == "operations":
        return title + "\n" + _operations(container, live) + "\n"
    # full
    return "\n\n".join([title, _operations(container, live),
                        _performance_summary(container, live),
                        container.memory.engineering_report()]) + "\n"
