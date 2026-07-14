"""Engineering Memory benchmark (Phase 10).

Populates a memory store from a series of simulated runs and measures:
  insertion latency, query latency, similarity-search latency, memory size,
  historical-query latency, planner memory utilization, analyst memory utilization.

Writes benchmarks/memory_metrics.json and benchmarks/memory_report.md (the trend +
engineering reports over the recorded history). No API key / no vector DB.

Run: python scripts/memory_eval.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from vds.core.contracts import (
    BenchmarkResult,
    DatasetQualityReport,
    ErrorAnalysis,
    ErrorCategory,
    ExecutionReport,
    ExportReport,
)
from vds.memory import DatasetFingerprint, EngineeringMemoryService


def _execution(i: int) -> ExecutionReport:
    # A history that visibly improves: review rate falls, throughput rises.
    review = round(max(0.05, 0.4 - i * 0.03), 3)
    approval = round(min(0.92, 0.55 + i * 0.03), 3)
    ips = round(8.0 + i * 0.6, 3)
    imported = 30 + i * 5
    dets = imported * 2
    q = DatasetQualityReport(
        project_id=f"run{i}", images=imported, detections=dets, masks=dets,
        approval_rate=approval, review_rate=review, rejection_rate=round(1 - approval - review, 3),
        invalid_annotations=1, duplicate_detections=2, empty_masks=1,
        annotation_density=2.0, avg_confidence=round(0.7 + i * 0.01, 3), images_with_no_detection=1,
    )
    e = ErrorAnalysis(project_id=f"run{i}", total_annotations=dets, categories=[
        ErrorCategory(name="small_objects", count=max(0, 8 - i), description="small"),
        ErrorCategory(name="low_confidence", count=5, description="lowconf")])
    b = BenchmarkResult(
        project_id=f"run{i}", images_processed=imported, total_seconds=round(imported / ips, 3),
        images_per_second=ips, avg_inference_ms=5.0, stage_seconds={"detector": 1.0},
        num_batches=2, peak_ram_mb=500.0, cpu_percent=40.0, created_at=f"2026-07-{10 + i:02d}T10:00:00")
    ex = ExportReport(format="coco", images=imported, annotations=dets, categories=["object"],
                      output_path="x", validated=True)
    return ExecutionReport(
        project_id=f"run{i}", source="sim", imported=imported, duplicates_skipped=2, quarantined=0,
        detections=dets, verified_approved=int(dets * approval), needs_review=int(dets * review),
        rejected=int(dets * q.rejection_rate), export=ex, benchmark=b, quality=q, errors=e)


def _time(fn) -> tuple[object, float]:
    start = time.perf_counter()
    out = fn()
    return out, round((time.perf_counter() - start) * 1000, 4)


def main() -> None:
    path = Path("benchmarks/_memory_eval.json")
    if path.exists():
        path.unlink()
    svc = EngineeringMemoryService(path, min_similarity=0.4)

    scene = ["aerial", "street", "aerial", "indoor", "aerial", "street"]
    insert_latencies = []
    for i in range(6):
        _m, ms = _time(lambda i=i: svc.record_execution(
            _execution(i), f"2026-07-{10 + i:02d}T10:00:00",
            resolution_mp=round(2.0 + i, 2), scene_type=scene[i]))
        insert_latencies.append(ms)

    query_fp = DatasetFingerprint(resolution_mp=2.1, dataset_size=32, scene_type="aerial")
    matches, sim_ms = _time(lambda: svc.similar(query_fp, top_k=3))
    guidance, recall_ms = _time(lambda: svc.recall(query_fp))
    _all, all_ms = _time(svc.all)
    _trend, trend_ms = _time(svc.trend_report)

    metrics = {
        "records": len(svc.all()),
        "memory_size_bytes": path.stat().st_size,
        "insertion_latency_ms": {"avg": round(sum(insert_latencies) / len(insert_latencies), 4),
                                 "max": max(insert_latencies)},
        "similarity_search_latency_ms": sim_ms,
        "planner_recall_latency_ms": recall_ms,
        "historical_query_latency_ms": all_ms,
        "trend_report_latency_ms": trend_ms,
        "planner_memory_utilization": {
            "matches_found": len(matches),
            "top_match": matches[0].memory.id if matches else None,
            "top_score": matches[0].score if matches else None,
            "note": guidance.note,
        },
        "analyst_memory_utilization": {
            "validated_memories": sum(1 for m in svc.all() if m.validation_status == "validated"),
            "total_recommendations": sum(len(m.engineering_recommendations) for m in svc.all()),
        },
    }
    Path("benchmarks").mkdir(exist_ok=True)
    Path("benchmarks/memory_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = svc.engineering_report() + "\n" + svc.trend_report()
    if matches:
        report += "\n## Example Planner Recall\n" + guidance.render() + "\n"
    Path("benchmarks/memory_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
