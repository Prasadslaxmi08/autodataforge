"""Integration test: the complete pipeline, folder in -> exported dataset out."""

from __future__ import annotations

from pathlib import Path

from vds.container import Container


def test_full_pipeline(container: Container, dataset_dir: Path, tmp_path: Path):
    report = container.pipeline.run(
        str(dataset_dir), export_format="coco", dest=str(tmp_path / "export")
    )

    # Import
    assert report.imported == 3
    assert report.quarantined == 0

    # Detection + verification: 2 objects/image, all solid -> approved.
    assert report.detections == 6
    assert report.verified_approved == 6
    assert report.rejected == 0
    assert report.verified_approved + report.needs_review + report.rejected == report.detections

    # Export produced a validated package on disk.
    assert report.export.validated
    assert (tmp_path / "export" / "annotations.json").exists()
    assert (tmp_path / "export" / "images").is_dir()

    # Benchmark populated.
    assert report.benchmark.images_processed == 3
    assert report.benchmark.images_per_second > 0
    assert {"ingest", "detection", "segmentation", "verification", "export"} <= set(
        report.benchmark.stage_seconds
    )


def test_pipeline_yolo_export(container: Container, dataset_dir: Path, tmp_path: Path):
    report = container.pipeline.run(
        str(dataset_dir), export_format="yolo", dest=str(tmp_path / "y")
    )
    assert report.export.format == "yolo"
    assert (tmp_path / "y" / "labels").is_dir()
    assert (tmp_path / "y" / "classes.txt").read_text().strip() == "object"
