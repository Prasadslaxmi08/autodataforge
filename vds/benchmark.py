"""Benchmark collection (Phase 1).

Times each pipeline stage, samples system metrics (psutil), and writes a
BenchmarkResult to `benchmarks/` after every run. Real measured numbers here
become the README's performance claims later.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import psutil

from vds.core.contracts import BenchmarkResult, ProjectId
from vds.logging import get_logger

log = get_logger(__name__)

BENCHMARKS_DIR = Path("benchmarks")


def _gpu_util() -> float | None:
    """Best-effort GPU utilization via nvidia-smi; None if unavailable.

    ponytail: shells out instead of adding a pynvml dependency — GPU metrics are
    optional and this keeps the base install free of NVIDIA libraries.
    """

    return _nvidia_query("utilization.gpu")


def _gpu_mem_mb() -> float | None:
    """Best-effort GPU memory used (MB) via nvidia-smi; None if unavailable."""
    return _nvidia_query("memory.used")


def _nvidia_query(field: str) -> float | None:
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return float(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError, OSError, subprocess.SubprocessError):
        return None


class BenchmarkCollector:
    """Accumulates per-stage timings and a running inference count for one run."""

    def __init__(self, project_id: ProjectId) -> None:
        self.project_id = project_id
        self._stage_seconds: dict[str, float] = {}
        self._inference_count = 0
        self._inference_seconds = 0.0
        self._proc = psutil.Process()
        self._peak_ram_mb = self._ram_mb()
        self._start = time.perf_counter()

    def _ram_mb(self) -> float:
        return self._proc.memory_info().rss / (1024 * 1024)

    @contextmanager
    def stage(self, name: str):
        """Time a named stage; adds to that stage's cumulative seconds."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._stage_seconds[name] = self._stage_seconds.get(name, 0.0) + elapsed
            self._peak_ram_mb = max(self._peak_ram_mb, self._ram_mb())

    def record_inference(self, seconds: float, count: int = 1) -> None:
        """Track model-call time so avg inference latency can be reported."""
        self._inference_seconds += seconds
        self._inference_count += count

    def finish(self, images_processed: int, num_batches: int = 0) -> BenchmarkResult:
        total = time.perf_counter() - self._start
        label_seconds = self._stage_seconds.get("detection", 0.0) + self._stage_seconds.get(
            "segmentation", 0.0
        )
        return BenchmarkResult(
            project_id=self.project_id,
            images_processed=images_processed,
            total_seconds=round(total, 4),
            images_per_second=round(images_processed / total, 4) if total > 0 else 0.0,
            avg_inference_ms=round(
                1000 * self._inference_seconds / self._inference_count, 3
            )
            if self._inference_count
            else 0.0,
            stage_seconds={k: round(v, 4) for k, v in self._stage_seconds.items()},
            num_batches=num_batches,
            batch_throughput_ips=round(images_processed / label_seconds, 4)
            if label_seconds > 0
            else 0.0,
            peak_ram_mb=round(self._peak_ram_mb, 1),
            peak_vram_mb=_gpu_mem_mb(),
            cpu_percent=self._proc.cpu_percent(interval=0.1),
            gpu_util_percent=_gpu_util(),
            created_at=datetime.now(UTC).isoformat(),
        )

    @staticmethod
    def save(result: BenchmarkResult, directory: Path = BENCHMARKS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = result.created_at.replace(":", "").replace("-", "").replace(".", "_")
        path = directory / f"phase1_{stamp}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        log.info("benchmark.saved", path=str(path), imgs_per_sec=result.images_per_second)
        return path
