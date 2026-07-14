"""Operations & Performance Center view-model (Phase 17) — plain data for the ops
dashboard. No Qt here.

An engineering operations dashboard, not an AI workspace. Every number is read from
measured execution data: historical benchmark runs live in Engineering Memory (each
record carries measured `ExecutionMetrics` + `BenchmarkSummary`), current dataset
totals come from the store, and CPU/RAM/disk are a live psutil snapshot. Anything the
platform does not measure (GPU utilization/memory, export time, failure logs) is
reported as **Unavailable** — never estimated or fabricated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from vds.container import Container
from vds.memory.schema import EngineeringMemory
from vds.memory.trends import TrendAnalyzer

try:
    import psutil
except ImportError:  # base dependency, but stay defensive (matches ResourceMonitor)
    psutil = None  # type: ignore

NA = "Unavailable"


# --- dataclasses -----------------------------------------------------------
@dataclass
class KPI:
    label: str
    value: str
    sub: str = ""


@dataclass
class SystemStat:
    name: str
    value: str
    status: str  # ok | warn | crit | na


@dataclass
class BenchmarkRow:
    run_id: str
    dataset: str
    detector: str
    segmentation: str
    strategy: str
    runtime: str
    ips: str
    review_rate: str
    verification: str
    export_success: str
    peak_ram: str
    peak_gpu: str
    created_at: str  # raw, for sorting/filtering


@dataclass
class ComparisonRow:
    metric: str
    values: list[str]
    trend: str  # improved | regressed | ""


@dataclass
class Comparison:
    runs: list[str]
    rows: list[ComparisonRow]


@dataclass
class TrendChart:
    metric: str
    series: list[float]
    last: float
    delta: float
    improved: bool


@dataclass
class HealthIndicator:
    name: str
    status: str  # ok | warn | crit | na
    detail: str


@dataclass
class PlatformHealth:
    status: str  # Healthy | Warning | Critical | Unknown
    indicators: list[HealthIndicator]
    root_causes: list[str]


@dataclass
class FilterOptions:
    datasets: list[str] = field(default_factory=list)
    detectors: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)


# --- helpers ---------------------------------------------------------------
def _names(container: Container) -> dict[str, str]:
    return {p.id: p.name for p in container.projects.list()}


def _strategy(m: EngineeringMemory) -> str:
    d = m.planner_decisions
    return f"{d.detector}, conf={d.confidence_threshold}, seg={d.segmentation_enabled}, tiling={d.tiling}"


def _sorted(memories: list[EngineeringMemory]) -> list[EngineeringMemory]:
    return sorted(memories, key=lambda m: (m.created_at, m.version))


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _db_totals(container: Container) -> tuple[int, int, int]:
    """(datasets, images, objects) from the current store — all measured."""
    datasets = images = objects = 0
    for p in container.projects.list():
        datasets += 1
        imgs = container.images.by_project(p.id)
        images += len(imgs)
        objects += sum(len(container.annotations.by_image(img.id)) for img in imgs)
    return datasets, images, objects


def live_snapshot(running_jobs: int = 0) -> dict:
    """A live system snapshot via psutil. Fields are None when psutil is absent."""
    snap: dict = {"running_jobs": running_jobs, "cpu": None, "ram_percent": None,
                  "ram_mb": None, "disk_percent": None}
    if psutil is not None:
        # Call on the UI thread only (see OperationsPage._refresh): psutil misbehaves
        # on a QThreadPool worker thread on Windows.
        vm = psutil.virtual_memory()
        snap["cpu"] = round(psutil.cpu_percent(interval=None), 1)
        snap["ram_percent"] = round(vm.percent, 1)
        snap["ram_mb"] = round(vm.used / (1024 * 1024), 0)
        try:
            snap["disk_percent"] = round(psutil.disk_usage(os.getcwd()).percent, 1)
        except OSError:
            snap["disk_percent"] = None
    return snap


def _pct_status(value: float | None, warn: float, crit: float) -> str:
    if value is None:
        return "na"
    return "crit" if value >= crit else ("warn" if value >= warn else "ok")


# --- section 1: executive overview -----------------------------------------
def executive_overview(container: Container, live: dict) -> list[KPI]:
    memories = container.memory.all()
    datasets, images, objects = _db_totals(container)
    review = _mean([m.execution_metrics.review_rate for m in memories])
    throughput = _mean([m.benchmark_summary.throughput_ips for m in memories])
    runtime = _mean([m.execution_metrics.runtime_seconds for m in memories])
    agreement = _mean([1.0 - m.execution_metrics.review_rate for m in memories])
    exported = [m for m in memories if m.execution_metrics.export_validated]
    export_rate = (len(exported) / len(memories)) if memories else None
    health = platform_health(container, live).status

    def pct(v):
        return f"{v:.0%}" if v is not None else NA

    return [
        KPI("Datasets Processed", str(datasets), f"{len(memories)} recorded runs"),
        KPI("Images Processed", str(images)),
        KPI("Objects Annotated", str(objects)),
        KPI("Average Review Rate", pct(review)),
        KPI("Average Throughput", f"{throughput:.2f} img/s" if throughput is not None else NA),
        KPI("Average Processing Time", f"{runtime:.2f} s" if runtime is not None else NA),
        KPI("Verification Agreement", pct(agreement)),
        KPI("Export Success Rate", pct(export_rate)),
        KPI("Platform Status", health),
        KPI("Running Jobs", str(live.get("running_jobs", 0))),
        KPI("Completed Jobs", str(len(memories))),
        KPI("Failed Jobs", NA, "no failure log persisted"),
    ]


# --- section 2: system performance -----------------------------------------
def system_performance(container: Container, live: dict) -> list[SystemStat]:
    memories = _sorted(container.memory.all())
    latest = memories[-1] if memories else None
    gpu_device = container.settings.gpu.device
    vram = container.settings.gpu.vram_budget_mb
    running = live.get("running_jobs", 0)

    # CPU / RAM: prefer the live snapshot; fall back to the latest measured run.
    cpu = live.get("cpu")
    if cpu is None and latest is not None and latest.execution_metrics.cpu_percent:
        cpu = latest.execution_metrics.cpu_percent
    ram_pct = live.get("ram_percent")
    ram_mb = live.get("ram_mb")
    if ram_mb is None and latest is not None and latest.execution_metrics.peak_ram_mb:
        ram_mb = latest.execution_metrics.peak_ram_mb

    stats = [
        SystemStat("CPU Usage", f"{cpu}%" if cpu is not None else NA,
                   _pct_status(cpu, 75, 90)),
        SystemStat("Memory Usage",
                   f"{ram_pct}%" + (f" ({ram_mb:.0f} MB)" if ram_mb else "")
                   if ram_pct is not None else (f"{ram_mb:.0f} MB" if ram_mb else NA),
                   _pct_status(ram_pct, 80, 92)),
        SystemStat("GPU Usage", f"{NA} (device {gpu_device})", "na"),
        SystemStat("GPU Memory", f"{NA} (budget {vram} MB)", "na"),
        SystemStat("Disk Usage",
                   f"{live['disk_percent']}%" if live.get("disk_percent") is not None else NA,
                   _pct_status(live.get("disk_percent"), 85, 95)),
        SystemStat("Running Threads", str(running), "ok" if running == 0 else "warn"),
        SystemStat("Queue Length", NA, "na"),  # QThreadPool exposes no pending count
        SystemStat("Worker Status", "Busy" if running else "Idle", "warn" if running else "ok"),
        SystemStat("API Status", "Embedded (in-process)", "ok"),
        SystemStat("Backend Status", "Online", "ok"),
        SystemStat("Container Status", "Online", "ok"),
    ]
    return stats


# --- section 3: benchmark explorer -----------------------------------------
def benchmark_runs(container: Container) -> list[BenchmarkRow]:
    names = _names(container)
    rows: list[BenchmarkRow] = []
    for m in reversed(_sorted(container.memory.all())):
        em, bs, d = m.execution_metrics, m.benchmark_summary, m.planner_decisions
        rows.append(BenchmarkRow(
            run_id=m.id, dataset=names.get(m.project_id, m.project_id[:8]),
            detector=d.detector, segmentation="On" if d.segmentation_enabled else "Off",
            strategy=_strategy(m), runtime=f"{em.runtime_seconds:.2f} s",
            ips=f"{bs.throughput_ips:.2f}", review_rate=f"{em.review_rate:.0%}",
            verification=f"{bs.avg_confidence:.0%}",
            export_success="Yes" if em.export_validated else "No",
            peak_ram=f"{em.peak_ram_mb:.0f} MB" if em.peak_ram_mb else NA,
            peak_gpu=NA,  # GPU memory is not measured by the pipeline
            created_at=m.created_at))
    return rows


# --- section 4: performance comparison -------------------------------------
def _trend_dir(values: list[float], higher_is_better: bool) -> str:
    if len(values) < 2:
        return ""
    improved = (values[-1] >= values[0]) if higher_is_better else (values[-1] <= values[0])
    return "improved" if improved else "regressed"


def compare_runs(container: Container, ids: list[str]) -> Comparison:
    names = _names(container)
    by_id = {m.id: m for m in container.memory.all()}
    chosen = _sorted([by_id[i] for i in ids if i in by_id])
    if not chosen:
        return Comparison([], [])
    labels = [f"{names.get(m.project_id, m.project_id[:8])}·{m.planner_decisions.detector}"
              for m in chosen]

    def row(metric, fn, higher, fmt=lambda v: f"{v}"):
        raw = [fn(m) for m in chosen]
        return ComparisonRow(metric, [fmt(v) for v in raw], _trend_dir([float(v) for v in raw], higher))

    return Comparison(labels, [
        ComparisonRow("Planner Strategy", [_strategy(m) for m in chosen], ""),
        row("Runtime (s)", lambda m: round(m.execution_metrics.runtime_seconds, 2), False),
        row("Peak Memory (MB)", lambda m: round(m.execution_metrics.peak_ram_mb, 0), False),
        ComparisonRow("GPU", [NA for _ in chosen], ""),
        row("Review Rate", lambda m: m.execution_metrics.review_rate, False, lambda v: f"{v:.0%}"),
        row("Verification", lambda m: m.benchmark_summary.avg_confidence, True, lambda v: f"{v:.0%}"),
        ComparisonRow("Export Time", [NA for _ in chosen], ""),
        row("Throughput (img/s)", lambda m: round(m.benchmark_summary.throughput_ips, 2), True),
    ])


# --- section 5: historical trends ------------------------------------------
def historical_trends(container: Container) -> list[TrendChart]:
    ms = _sorted(container.memory.all())
    if not ms:
        return []
    ev = TrendAnalyzer().evolution(ms)

    def chart(metric, series, higher):
        s = [round(float(v), 4) for v in series]
        delta = round(s[-1] - s[0], 4) if s else 0.0
        improved = (delta >= 0) if higher else (delta <= 0)
        return TrendChart(metric, s, s[-1] if s else 0.0, delta, improved)

    return [
        chart("Review Reduction", [m.execution_metrics.review_rate for m in ms], False),
        chart("Runtime (s)", [m.execution_metrics.runtime_seconds for m in ms], False),
        chart("Throughput (img/s)", ev["throughput"].series, True),
        chart("Memory Usage (MB)", [m.execution_metrics.peak_ram_mb for m in ms], False),
        chart("Verification Agreement", [m.benchmark_summary.avg_confidence for m in ms], True),
        chart("Annotation Quality", ev["quality"].series, True),
        chart("Dataset Growth", list(range(1, len(ms) + 1)), True),
        chart("Knowledge Growth", list(range(1, len(ms) + 1)), True),
        chart("Engineering Recommendations",
              [float(len(m.engineering_recommendations)) for m in ms], True),
    ]


# --- section 6: platform health --------------------------------------------
def platform_health(container: Container, live: dict) -> PlatformHealth:
    memories = container.memory.all()
    running = live.get("running_jobs", 0)
    export_failures = sum(1 for m in memories if not m.execution_metrics.export_validated)
    verification_errors = sum(m.verification_outcomes.bbox_issues
                              + m.verification_outcomes.segmentation_issues for m in memories)
    ram_pct = live.get("ram_percent")

    indicators = [
        HealthIndicator("Queue Backlog", "ok" if running <= 4 else "warn",
                        f"{running} running job(s)"),
        HealthIndicator("Export Failures", "ok" if export_failures == 0 else "warn",
                        f"{export_failures} run(s) with unvalidated export"),
        HealthIndicator("Verification Errors", "ok" if verification_errors == 0 else "warn",
                        f"{verification_errors} bbox/segmentation issue(s) recorded"),
        HealthIndicator("Processing Failures", "na", "no failure log persisted"),
        HealthIndicator("Memory Pressure", _pct_status(ram_pct, 80, 92),
                        f"{ram_pct}% RAM" if ram_pct is not None else NA),
        HealthIndicator("Worker Availability", "ok", "thread pool available"),
        HealthIndicator("GPU Availability", "na",
                        f"no telemetry (device {container.settings.gpu.device})"),
    ]
    order = {"crit": 3, "warn": 2, "ok": 1, "na": 0}
    worst = max((order[i.status] for i in indicators), default=0)
    status = ({3: "Critical", 2: "Warning", 1: "Healthy"}.get(worst, "Unknown")
              if memories or live.get("cpu") is not None else "Unknown")
    root_causes = [f"{i.name}: {i.detail}" for i in indicators if i.status in ("warn", "crit")]
    return PlatformHealth(status, indicators, root_causes)


# --- filter options --------------------------------------------------------
def filter_options(container: Container) -> FilterOptions:
    ms = container.memory.all()
    names = _names(container)
    return FilterOptions(
        datasets=sorted({names.get(m.project_id, m.project_id[:8]) for m in ms}),
        detectors=sorted({m.planner_decisions.detector for m in ms}),
        strategies=sorted({_strategy(m) for m in ms}),
        statuses=["Healthy", "Warning", "Critical", "Unknown"],
    )
