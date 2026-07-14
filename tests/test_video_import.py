"""Video Dataset Import Engine tests (Phase 17.5): probing, extraction strategies,
duplicate detection, metadata preservation, planner pre-analysis, pipeline & export
compatibility, thread safety, cancellation, resume, and invalid-video handling.

Videos are generated as multi-frame GIF/TIFF sequences (PIL-native, no ffmpeg needed),
so the whole engine is exercised headlessly. The core guarantee under test: a video
becomes a standard image dataset that the EXISTING pipeline imports unchanged."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

pytest.importorskip("PySide6")

from vds.gui.controller import BackendController  # noqa: E402
from vds.video import (  # noqa: E402
    ExtractionConfig,
    VideoImportError,
    extract_frames,
    frame_indices,
    open_source,
    probe,
)


# --- fixtures / helpers ----------------------------------------------------
def _distinct_frames(n: int, size=(160, 120)) -> list[Image.Image]:
    """n visually distinct frames (a shape that moves far each frame, so no two are
    near-duplicates under average-hash)."""
    w, h = size
    frames = []
    for i in range(n):
        im = Image.new("RGB", (w, h), (8, 8, 8))
        d = ImageDraw.Draw(im)
        x = int((i / max(1, n - 1)) * (w - 50))
        d.rectangle([x, 20, x + 45, 90], fill=(240 - i * 7 % 240, (i * 37) % 240, 60 + i * 5 % 190))
        d.ellipse([w - 50, (i * 11) % (h - 30), w - 15, (i * 11) % (h - 30) + 28],
                  fill=((i * 23) % 240, 200, 120))
        frames.append(im)
    return frames


def _make_gif(path: Path, frames: list[Image.Image], duration_ms: int = 100) -> Path:
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms,
                   loop=0, disposal=2)
    return path


def _make_tiff(path: Path, frames: list[Image.Image]) -> Path:
    frames[0].save(path, save_all=True, append_images=frames[1:], format="TIFF")
    return path


@pytest.fixture
def small_gif(tmp_path) -> Path:
    return _make_gif(tmp_path / "small.gif", _distinct_frames(6))


@pytest.fixture
def dup_gif(tmp_path) -> Path:
    frames = _distinct_frames(6)
    frames.append(frames[0].copy())  # an exact duplicate of frame 0
    return _make_gif(tmp_path / "dup.gif", frames)


# --- probing / metadata ----------------------------------------------------
def test_probe_reads_metadata(small_gif):
    info = probe(small_gif)
    assert info.total_frames == 6
    assert info.width == 160 and info.height == 120
    assert info.codec == "GIF"
    assert info.fps and info.fps > 0
    assert info.megapixels == round(160 * 120 / 1e6, 3)


def test_high_resolution_video(tmp_path):
    info = probe(_make_gif(tmp_path / "hires.gif", _distinct_frames(3, size=(800, 600))))
    assert info.width == 800 and info.height == 600
    assert info.megapixels == round(800 * 600 / 1e6, 3)


def test_large_video_many_frames(tmp_path):
    info = probe(_make_gif(tmp_path / "large.gif", _distinct_frames(40)))
    assert info.total_frames == 40


def test_different_container_tiff(tmp_path):
    info = probe(_make_tiff(tmp_path / "clip.tif", _distinct_frames(5)))
    assert info.total_frames == 5
    assert info.codec == "TIFF"


def test_invalid_video_raises(tmp_path):
    with pytest.raises(VideoImportError):
        probe(tmp_path / "missing.mp4")
    bogus = tmp_path / "bogus.gif"
    bogus.write_bytes(b"not a real gif")
    with pytest.raises(VideoImportError):
        probe(bogus)


# --- extraction strategies -------------------------------------------------
def test_strategy_indices(small_gif):
    info = probe(small_gif)
    assert frame_indices(ExtractionConfig("every_frame"), info.total_frames, info.fps) == list(range(6))
    assert frame_indices(ExtractionConfig("every_n", every_n=2), info.total_frames, info.fps) == [0, 2, 4]
    assert frame_indices(ExtractionConfig("fixed_count", count=3), info.total_frames, info.fps) == [0, 2, 5]
    secs = frame_indices(ExtractionConfig("every_seconds", seconds=0.2), info.total_frames, info.fps)
    assert secs[0] == 0 and len(secs) >= 1


def test_every_seconds_needs_fps(tmp_path):
    # multi-page TIFF has no frame duration -> fps is None -> honest error, not a guess
    info = probe(_make_tiff(tmp_path / "no_fps.tif", _distinct_frames(4)))
    assert info.fps is None
    with pytest.raises(VideoImportError):
        frame_indices(ExtractionConfig("every_seconds", seconds=1.0), info.total_frames, info.fps)


def test_scene_change_extraction(small_gif, tmp_path):
    src = open_source(small_gif)
    stats, manifest = extract_frames(src, ExtractionConfig("scene_change", scene_threshold=8),
                                     tmp_path / "scene", dedup=False)
    src.close()
    assert stats.unique_frames >= 1
    assert len(manifest) == stats.unique_frames


# --- duplicate detection (reuses the pipeline's average-hash) ---------------
def test_duplicate_detection(dup_gif, tmp_path):
    src = open_source(dup_gif)
    stats, _ = extract_frames(src, ExtractionConfig("every_frame"), tmp_path / "out", dedup=True)
    src.close()
    assert stats.frames_removed >= 1  # the exact duplicate is removed
    assert stats.unique_frames == stats.frames_extracted - stats.frames_removed
    assert 0 < stats.duplicate_percentage <= 100


def test_dedup_disabled_keeps_all(dup_gif, tmp_path):
    src = open_source(dup_gif)
    stats, _ = extract_frames(src, ExtractionConfig("every_frame"), tmp_path / "out", dedup=False)
    src.close()
    assert stats.frames_removed == 0
    assert stats.unique_frames == stats.frames_extracted


# --- metadata preservation -------------------------------------------------
def test_manifest_metadata(small_gif, tmp_path):
    src = open_source(small_gif)
    _stats, manifest = extract_frames(src, ExtractionConfig("every_n", every_n=2),
                                      tmp_path / "out", dedup=False)
    src.close()
    m = manifest[0]
    for key in ("file", "original_video", "frame_number", "timestamp", "resolution",
                "fps", "extraction_strategy", "extraction_parameters", "import_date"):
        assert key in m
    assert m["original_video"] == "small.gif"
    assert m["extraction_strategy"] == "every_n"


# --- cancellation + resume -------------------------------------------------
def test_cancellation(small_gif, tmp_path):
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 3  # stop after 3 frames

    src = open_source(small_gif)
    stats, _ = extract_frames(src, ExtractionConfig("every_frame"), tmp_path / "out",
                              dedup=False, cancel=cancel)
    src.close()
    assert stats.cancelled is True
    assert stats.unique_frames < 6


def test_resume(small_gif, tmp_path):
    out = tmp_path / "out"
    calls = {"n": 0}
    src = open_source(small_gif)
    extract_frames(src, ExtractionConfig("every_frame"), out, dedup=False,
                   cancel=lambda: (calls.__setitem__("n", calls["n"] + 1) or calls["n"] > 2))
    src.close()
    partial = len(list(out.glob("*.png")))
    assert partial < 6
    src2 = open_source(small_gif)
    stats, _ = extract_frames(src2, ExtractionConfig("every_frame"), out, dedup=False, resume=True)
    src2.close()
    assert stats.cancelled is False
    assert len(list(out.glob("*.png"))) == 6  # resumed to completion


# --- pipeline + export compatibility (the whole point) ---------------------
def test_pipeline_compatibility(container, small_gif):
    ctrl = BackendController(container)
    report, stats, info = ctrl.import_video_dataset(str(small_gif), "vidset", ExtractionConfig("every_frame"))
    assert report is not None
    assert report.imported == stats.unique_frames  # frames became normal dataset images
    datasets = {d.name: d for d in ctrl.list_datasets()}
    assert datasets["vidset"].image_count == stats.unique_frames
    # metadata manifest is preserved and accessible after import
    manifest = ctrl.video_manifest(report.project_id)
    assert manifest and manifest["video"]["name"] == "small.gif"
    assert len(manifest["frames"]) == stats.unique_frames


@pytest.mark.parametrize("fmt", ["coco", "yolo"])
def test_export_compatibility(container, small_gif, fmt):
    ctrl = BackendController(container)
    report, _stats, _info = ctrl.import_video_dataset(
        str(small_gif), f"vid_{fmt}", ExtractionConfig("every_n", every_n=2), export_format=fmt)
    assert report is not None
    export_dir = Path(ctrl.export_dir(f"vid_{fmt}"))
    assert export_dir.exists()
    assert any(export_dir.rglob("*"))  # export produced files


# --- planner pre-analysis --------------------------------------------------
def test_planner_pre_analysis(container, small_gif):
    ctrl = BackendController(container)
    info = ctrl.probe_video(str(small_gif))
    plan = ctrl.plan_video(info, ExtractionConfig("every_frame"))
    assert plan.estimated_dataset_size == 6
    assert plan.recommended_detector  # a real planner recommendation
    assert plan.recommended_batch_size > 0
    assert plan.source in ("ai", "deterministic")


def test_cancelled_import_creates_no_dataset(container, small_gif):
    ctrl = BackendController(container)
    report, stats, _info = ctrl.import_video_dataset(
        str(small_gif), "cancelled", ExtractionConfig("every_frame"), cancel=lambda: True)
    assert report is None
    assert stats.cancelled is True
    assert ctrl.list_datasets() == []  # nothing entered the pipeline


# --- thread safety ---------------------------------------------------------
def test_import_off_ui_thread(qtbot, container, small_gif):
    from vds.gui.threads import ThreadManager

    ctrl = BackendController(container)
    tm = ThreadManager()
    out = {}
    worker = tm.submit(ctrl.import_video_dataset, str(small_gif), "threaded",
                       ExtractionConfig("every_frame"), on_finished=lambda r: out.setdefault("r", r))
    with qtbot.waitSignal(worker.signals.finished, timeout=60000):
        pass
    assert out["r"][0].imported > 0
    assert tm.active == 0


# --- import dialog widget --------------------------------------------------
def test_video_dialog_populates(qtbot, container, small_gif):
    from vds.gui.notifications import NotificationSystem
    from vds.gui.threads import ThreadManager
    from vds.gui.video_import_dialog import VideoImportDialog

    ctrl = BackendController(container)
    dlg = VideoImportDialog(ctrl, ThreadManager(), NotificationSystem(),
                            str(small_gif), "dlgset")
    qtbot.addWidget(dlg)
    dlg._on_probed(ctrl.probe_video(str(small_gif)))  # populate synchronously
    assert dlg._info.total_frames == 6
    assert dlg._info_grid.count() > 0
    dlg._strategy.setCurrentText("Fixed Number of Frames")
    assert dlg._config().strategy == "fixed_count"
    dlg._on_planned(ctrl.plan_video(dlg._info, dlg._config()))
    assert dlg._plan_grid.count() > 0
    report, stats, info = ctrl.import_video_dataset(str(small_gif), "dlgset", dlg._config())
    dlg._on_imported((report, stats, info))
    assert "Unique:" in dlg._stats.text()
