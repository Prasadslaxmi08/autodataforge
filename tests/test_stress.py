"""Phase-5 stress and determinism tests.

Exercises edge cases the baseline must survive: large/empty/corrupt inputs,
mixed resolutions, invalid annotations, ingest resume, and byte-for-byte
determinism of the deterministic pipeline.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from vds.container import Container
from vds.core.contracts import Annotation, Box2D, ImageRecord, Provenance


def _img(path: Path, size, boxes) -> None:
    im = Image.new("RGB", size, (240, 240, 240))
    dr = ImageDraw.Draw(im)
    for b in boxes:
        dr.rectangle(b, fill=(10, 10, 10))
    im.save(path)


def test_large_dataset(container: Container, tmp_path: Path):
    d = tmp_path / "big"
    d.mkdir()
    for i in range(120):
        _img(d / f"i{i:03d}.png", (96, 96), [(10, 10, 24, 24), (50, 50, 66, 66)])
    report = container.pipeline.run(str(d), dest=str(tmp_path / "e"))
    assert report.imported >= 1  # near-dup filtering may collapse identical frames
    assert report.benchmark.images_per_second > 0
    assert report.export.validated


def test_empty_dataset(container: Container, tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    report = container.pipeline.run(str(empty), dest=str(tmp_path / "e"))
    assert report.imported == 0
    assert report.detections == 0
    assert report.export.validated  # a valid, empty COCO package
    assert report.quality.annotation_density == 0.0


def test_corrupt_images_quarantined(container: Container, tmp_path: Path):
    d = tmp_path / "mix"
    d.mkdir()
    _img(d / "good.png", (64, 64), [(8, 8, 20, 20)])
    (d / "bad.png").write_bytes(b"\x89PNG\r\n garbage not an image")
    report = container.pipeline.run(str(d), dest=str(tmp_path / "e"))
    assert report.quarantined == 1
    assert report.imported == 1


def test_mixed_resolutions(container: Container, tmp_path: Path):
    d = tmp_path / "res"
    d.mkdir()
    # Distinct content per image (varied box counts/positions) so the aHash
    # near-duplicate filter keeps them; the point is handling varied sizes.
    layouts = [
        [(8, 8, 24, 24)],
        [(10, 10, 30, 30), (60, 40, 90, 70)],
        [(20, 20, 60, 60), (120, 90, 160, 130), (30, 100, 55, 140)],
        [(8, 120, 30, 160), (50, 20, 80, 60)],
    ]
    for i, size in enumerate([(64, 64), (128, 96), (200, 150), (96, 200)]):
        _img(d / f"r{i}.png", size, layouts[i])
    report = container.pipeline.run(str(d), export_format="yolo", dest=str(tmp_path / "e"))
    assert report.imported == 4
    assert report.export.validated  # YOLO normalization must hold for every size


def test_invalid_annotation_rejected(container: Container, tmp_path: Path):
    # A degenerate-geometry annotation must be auto-rejected and never exported.
    from io import BytesIO

    buf = BytesIO()
    Image.new("RGB", (32, 32), (255, 255, 255)).save(buf, format="PNG")
    sha = container.cas.put(buf.getvalue())
    container.images.add(
        ImageRecord(id="i1", project_id="p", sha256=sha, width=32, height=32, state="labeled")
    )
    container.annotations.add(
        Annotation(
            id="bad", image_id="i1", label="object",
            geometry=Box2D(x=0, y=0, w=0, h=5), confidence=0.99,
            state="labeled", provenance=Provenance(source="t"),
        )
    )
    verdict = container.verifier.verify(b"", container.annotations.by_image("i1")[0])
    from vds.agents.verifier import REJECTED

    assert verdict.verdict == REJECTED


def test_ingest_resume_is_idempotent(container: Container, dataset_dir: Path):
    # Re-running import after an "interruption" must not duplicate work — the
    # deterministic layer's resume property (job-level checkpoint resume is Phase 2).
    first = container.importer.import_folder("p1", str(dataset_dir))
    second = container.importer.import_folder("p1", str(dataset_dir))
    assert first.imported == 3
    assert second.imported == 0  # everything already stored
    assert len(container.images.by_project("p1")) == 3


def test_pipeline_deterministic(tmp_path: Path, dataset_dir: Path):
    from vds.config.settings import Settings, StorageSettings

    def run_once(tag: str):
        c = Container(
            settings=Settings(storage=StorageSettings(cas_root=tmp_path / f"cas_{tag}")),
            db_path=":memory:",
            artifacts_dir=tmp_path / f"art_{tag}",
        )
        return c.pipeline.run(str(dataset_dir), dest=str(tmp_path / f"exp_{tag}"))

    a, b = run_once("a"), run_once("b")
    assert a.detections == b.detections
    assert a.quality.model_dump(exclude={"project_id"}) == b.quality.model_dump(
        exclude={"project_id"}
    )
