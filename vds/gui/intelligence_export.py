"""Intelligence Markdown export (Phase 15) — presentation of measured/validated data.

Renders the assembled DatasetIntelligence (or a single section) to Markdown. It
reformats values that already came from the Analyst / measured metrics; it computes
nothing new. PDF export is produced by Qt from this Markdown on the UI side.
"""

from __future__ import annotations


def _executive(intel) -> str:
    s = intel.summary
    return "\n".join([
        "## Executive Summary",
        f"- Dataset: {s.dataset} (v{s.version})",
        f"- Size: {s.size_mb} MB, {s.image_count} images",
        f"- Overall Dataset Health: {s.overall_health}/100",
        f"- Annotation Quality: {s.annotation_quality:.0%}",
        f"- Verification Confidence: {s.verification_confidence:.0%}",
        f"- Production Readiness: {s.production_readiness}",
        f"- Overall Recommendation: {s.overall_recommendation}",
        f"- Historical Improvement: {s.historical_improvement}",
        f"- Analyst ({s.source}, confidence {s.analyst_confidence}): {s.analyst_summary}",
    ])


def _health(intel) -> str:
    return "\n".join(["## Dataset Health", *[f"- {k.name}: {k.value}" for k in intel.kpis]])


def _issues(intel) -> str:
    out = ["## Root Cause Analysis"]
    for i in intel.issues:
        out += [f"### {i.title}", f"- Description: {i.description}",
                f"- Evidence: {'; '.join(i.evidence)}", f"- Impact: {i.impact}",
                f"- Recommendation: {i.recommendation}",
                f"- Expected Improvement: {i.expected_improvement}",
                f"- Confidence: {i.confidence}"]
    if not intel.issues:
        out.append("- No dominant issues detected.")
    return "\n".join(out)


def _recommendations(intel) -> str:
    out = ["## Prioritized Recommendations"]
    for r in intel.recommendations:
        out += [f"### [{r.priority}] {r.recommendation}",
                f"- Problem: {r.problem}", f"- Expected Gain: {r.expected_gain}",
                f"- Estimated Effort: {r.estimated_effort}",
                f"- Expected Review Reduction: {r.expected_review_reduction}",
                f"- Expected Runtime Impact: {r.expected_runtime_impact}",
                f"- Rationale: {r.rationale}", f"- Confidence: {r.confidence}"]
    if not intel.recommendations:
        out.append("- No recommendations.")
    return "\n".join(out)


def _readiness(intel) -> str:
    out = ["## Dataset Readiness"]
    for c in intel.readiness:
        out.append(f"- {'✓' if c.met else '✗'} {c.name}: {c.reasoning}")
    return "\n".join(out)


def to_markdown(intel, section: str = "all") -> str:
    title = f"# Dataset Intelligence — {intel.summary.dataset}\n"
    parts = {
        "executive": _executive, "health": _health, "issues": _issues,
        "recommendations": _recommendations, "readiness": _readiness,
    }
    if section == "engineering":
        return title + "\n" + intel.analyst_report_markdown
    if section in parts:
        return title + "\n" + parts[section](intel) + "\n"
    body = "\n\n".join(fn(intel) for fn in parts.values())
    return title + "\n" + body + "\n\n" + intel.analyst_report_markdown
