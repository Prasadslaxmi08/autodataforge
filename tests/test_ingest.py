"""Unit tests for the dataset import service."""

from __future__ import annotations

from pathlib import Path

from vds.container import Container


def test_import_counts(container: Container, dataset_dir: Path):
    result = container.importer.import_folder("p1", str(dataset_dir))
    assert result.imported == 3
    assert result.duplicates_skipped == 0
    assert result.quarantined == 0
    assert len(result.image_ids) == 3


def test_import_dedup(container: Container, dataset_dir: Path, tmp_path: Path):
    # Copy an existing image under a new name -> near-duplicate, must be skipped.
    dup = dataset_dir / "a_copy.png"
    dup.write_bytes((dataset_dir / "a.png").read_bytes())
    result = container.importer.import_folder("p1", str(dataset_dir))
    assert result.imported == 3
    assert result.duplicates_skipped == 1


def test_import_quarantines_corrupt(container: Container, dataset_dir: Path):
    (dataset_dir / "broken.png").write_bytes(b"not really a png")
    result = container.importer.import_folder("p1", str(dataset_dir))
    assert result.quarantined == 1
    assert result.imported == 3


def test_import_stores_metadata_and_cas(container: Container, dataset_dir: Path):
    container.importer.import_folder("p1", str(dataset_dir))
    records = container.images.by_project("p1")
    assert len(records) == 3
    for r in records:
        assert r.width == 64 and r.height == 64
        assert container.cas.exists(r.sha256)  # bytes actually stored
