"""Unit tests for the deterministic quality analyzer and error analysis."""

from __future__ import annotations

from pathlib import Path

from vds.container import Container


def test_quality_metrics(container: Container, dataset_dir: Path, tmp_path: Path):
    report = container.pipeline.run(str(dataset_dir), dest=str(tmp_path / "e"))
    q = report.quality
    assert q.images == 3
    assert q.detections == 6
    assert q.masks == 6  # every detection is segmented
    assert 0.0 <= q.approval_rate <= 1.0
    assert abs(q.approval_rate + q.review_rate + q.rejection_rate - 1.0) < 1e-6
    assert q.annotation_density == 2.0  # 6 detections / 3 images
    assert q.avg_confidence > 0


def test_error_analysis_categories(container: Container, dataset_dir: Path, tmp_path: Path):
    report = container.pipeline.run(str(dataset_dir), dest=str(tmp_path / "e"))
    names = {c.name for c in report.errors.categories}
    assert {
        "low_confidence", "small_objects", "duplicate_annotations",
        "empty_masks", "crowded_scenes", "possible_missed_detection",
    } == names
    assert all(c.count >= 0 for c in report.errors.categories)
    assert report.errors.unmeasurable  # ground-truth-dependent modes are declared


def test_quality_deterministic(container: Container, dataset_dir: Path, tmp_path: Path):
    r1 = container.pipeline.run(str(dataset_dir), dest=str(tmp_path / "e1"))
    # Fresh project on the same DB -> same measured quality.
    r2 = container.pipeline.run(str(dataset_dir), dest=str(tmp_path / "e2"))
    assert r1.quality.model_dump(exclude={"project_id"}) == r2.quality.model_dump(
        exclude={"project_id"}
    )
