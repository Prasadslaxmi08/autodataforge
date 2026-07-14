"""Performance-report generation (Phase-5).

Turns one ExecutionReport into a structured markdown report with the sections the
phase requires, plus a flat StageKPIs record for the comparison framework. The
recommendations are deterministic reads of the numbers — no judgement, no AI.
"""

from __future__ import annotations

from pathlib import Path

from vds.core.contracts import ExecutionReport, StageKPIs

REPORTS_DIR = Path("benchmarks/reports")


def to_kpis(report: ExecutionReport, stage: str = "deterministic") -> StageKPIs:
    q, b = report.quality, report.benchmark
    return StageKPIs(
        stage=stage,
        label=f"{stage} baseline",
        created_at=b.created_at,
        images_per_second=b.images_per_second,
        approval_rate=q.approval_rate,
        review_rate=q.review_rate,
        rejection_rate=q.rejection_rate,
        avg_confidence=q.avg_confidence,
        annotation_density=q.annotation_density,
        peak_ram_mb=b.peak_ram_mb,
        invalid_annotations=q.invalid_annotations,
        empty_masks=q.empty_masks,
    )


def _recommendations(r: ExecutionReport) -> list[str]:
    out: list[str] = []
    b, q = r.benchmark, r.quality
    if b.stage_seconds:
        slowest = max(b.stage_seconds, key=b.stage_seconds.get)
        out.append(
            f"Slowest stage is `{slowest}` ({b.stage_seconds[slowest]}s) — optimize it first."
        )
    if q.review_rate > 0.3:
        out.append(
            f"Human-review rate is {q.review_rate:.0%}; a triage/Analyst stage should reduce it."
        )
    if q.images_with_no_detection:
        out.append(
            f"{q.images_with_no_detection} image(s) produced zero detections — candidate misses "
            "for a stronger detector backend."
        )
    if q.duplicate_detections:
        out.append(
            f"{q.duplicate_detections} duplicate detection(s) survived NMS — "
            "tune the IoU threshold."
        )
    if q.avg_confidence < 0.6:
        out.append(
            f"Average confidence is {q.avg_confidence} — the backend is uncertain on this data."
        )
    if not out:
        out.append("No deterministic red flags on this dataset.")
    return out


def build_report(r: ExecutionReport) -> str:
    b, q, e = r.benchmark, r.quality, r.errors
    lines = [
        f"# Performance Report — project `{r.project_id[:8]}`",
        f"_source: {r.source} · generated: {b.created_at}_",
        "",
        "## Executive Summary",
        f"Processed **{r.imported} images** in **{b.total_seconds}s** "
        f"(**{b.images_per_second} img/s**), producing **{r.detections} annotations**. "
        f"Verifier approved {q.approval_rate:.0%}, flagged {q.review_rate:.0%} for review, "
        f"rejected {q.rejection_rate:.0%}. Export: {r.export.format}, "
        f"validated={r.export.validated}.",
        "",
        "## Pipeline Statistics",
        f"- imported: {r.imported}, duplicates skipped: {r.duplicates_skipped}, "
        f"quarantined: {r.quarantined}",
        f"- detections: {r.detections}, masks: {q.masks}, density: {q.annotation_density}/image",
        "",
        "## Performance Metrics",
        f"- throughput: {b.images_per_second} img/s, avg inference {b.avg_inference_ms} ms",
        f"- batches: {b.num_batches}, batch throughput {b.batch_throughput_ips} img/s",
        "- stage seconds: " + ", ".join(f"{k}={v}" for k, v in b.stage_seconds.items()),
        "",
        "## Resource Utilization",
        f"- peak RAM {b.peak_ram_mb} MB, peak VRAM {b.peak_vram_mb} MB",
        f"- CPU {b.cpu_percent}%, GPU {b.gpu_util_percent}%",
        "",
        "## Dataset Statistics",
        f"- approve/review/reject: {q.approval_rate:.0%} / {q.review_rate:.0%} / "
        f"{q.rejection_rate:.0%}",
        f"- avg confidence {q.avg_confidence}, invalid {q.invalid_annotations}, "
        f"empty masks {q.empty_masks}, duplicates {q.duplicate_detections}",
        f"- images with no detection: {q.images_with_no_detection}",
        "",
        "## Failure Analysis (heuristic proxies — no ground truth)",
    ]
    for c in e.categories:
        lines.append(f"- **{c.name}**: {c.count} — {c.description}")
    lines.append("- not measurable in the baseline: " + "; ".join(e.unmeasurable))
    lines += ["", "## Recommendations"]
    lines += [f"- {rec}" for rec in _recommendations(r)]
    lines.append("")
    return "\n".join(lines)


def save_report(r: ExecutionReport, directory: Path = REPORTS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = r.benchmark.created_at.replace(":", "").replace("-", "").replace(".", "_")
    path = directory / f"report_{stamp or r.project_id[:8]}.md"
    path.write_text(build_report(r), encoding="utf-8")
    return path
