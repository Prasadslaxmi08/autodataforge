"""Unit test for the performance-report generator."""

from __future__ import annotations

from pathlib import Path

from vds.container import Container
from vds.reporting import build_report, save_report, to_kpis


def test_report_has_all_sections(container: Container, dataset_dir: Path, tmp_path: Path):
    report = container.pipeline.run(str(dataset_dir), dest=str(tmp_path / "e"))
    md = build_report(report)
    for section in [
        "Executive Summary", "Pipeline Statistics", "Performance Metrics",
        "Resource Utilization", "Dataset Statistics", "Failure Analysis",
        "Recommendations",
    ]:
        assert f"## {section}" in md


def test_save_report_writes_utf8(container: Container, dataset_dir: Path, tmp_path: Path):
    report = container.pipeline.run(str(dataset_dir), dest=str(tmp_path / "e"))
    path = save_report(report, tmp_path / "reports")
    assert path.exists() and path.suffix == ".md"
    text = path.read_text(encoding="utf-8")  # must decode as UTF-8, not cp1252
    assert "—" in text  # em dash survived the round-trip


def test_to_kpis_maps_fields(container: Container, dataset_dir: Path, tmp_path: Path):
    report = container.pipeline.run(str(dataset_dir), dest=str(tmp_path / "e"))
    kpis = to_kpis(report, stage="deterministic")
    assert kpis.stage == "deterministic"
    assert kpis.approval_rate == report.quality.approval_rate
    assert kpis.images_per_second == report.benchmark.images_per_second
