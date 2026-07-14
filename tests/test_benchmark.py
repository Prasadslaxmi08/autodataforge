"""Unit test for the benchmark collector."""

from __future__ import annotations

import time
from pathlib import Path

from vds.benchmark import BenchmarkCollector


def test_collector_times_stages_and_saves(tmp_path: Path):
    bench = BenchmarkCollector("p1")
    with bench.stage("detection"):
        time.sleep(0.01)
    bench.record_inference(0.005, count=1)
    result = bench.finish(images_processed=4)

    assert result.images_processed == 4
    assert result.total_seconds > 0
    assert result.images_per_second > 0
    assert "detection" in result.stage_seconds
    assert result.avg_inference_ms > 0

    path = BenchmarkCollector.save(result, directory=tmp_path / "bench")
    assert path.exists()
    assert "images_per_second" in path.read_text()
