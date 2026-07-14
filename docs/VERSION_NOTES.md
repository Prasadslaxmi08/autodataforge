# AutoDataForge — Version Notes

Chronological, newest first. Each entry lists what changed and, importantly, what did
**not** change — the pipeline below the import layer is a stable contract.

## v0.17.5 — Video Dataset Import Engine

**Added**

- **Video files as a first-class dataset source.** `vds/video/` (`engine.py`) probes a
  video, extracts frames to a standard image folder, and the **existing** pipeline
  imports them like any folder. New GUI branch: *Dataset Manager → Import Video…*
  (`vds/gui/video_import_dialog.py`).
- **Native decoding, zero new dependencies** for multi-frame sequences (GIF / APNG /
  WebP / multi-page TIFF) via PIL; **ffmpeg/ffprobe** shell-out for real codecs
  (MP4/MOV/MKV/AVI/WebM/…) when installed, with a clear error when absent.
- **Extensible extraction strategies** (`STRATEGIES` registry): every frame, every N
  frames, every X seconds, fixed count, and scene change (experimental).
- **Deterministic duplicate reduction** reusing the pipeline's own average-hash
  (`vds/ingest/service.py:_average_hash`, Hamming ≤ 4) — video frames are deduped
  exactly like folder imports. Reports frames extracted / removed / unique / duplicate %.
- **Planner pre-analysis** before extraction: the existing Planner Agent runs over a
  synthetic dataset context built from the video metadata + chosen strategy and returns
  estimates/recommendations (override-able).
- **Per-frame metadata manifest** (original video, frame number, timestamp, resolution,
  fps, strategy, parameters, import date), preserved beside the dataset and readable via
  `BackendController.video_manifest(project_id)`.
- **Cancellation and resume** for extraction; all heavy work runs off the UI thread.
- New docs: [`ROADMAP.md`](../ROADMAP.md); expanded `docs/desktop_gui.md`; screenshot
  `docs/screenshots/10_video_import.png`.

**Unchanged (by design)**

- The Planner, Annotation, Verification, Analyst, Engineering Memory, Dataset
  Intelligence, Knowledge Center, and Operations Center are used **exactly as they are**.
- No second annotation pipeline. No downstream schema changes.
- **COCO and YOLO export** continue to work unchanged for video-derived datasets.

**Tested** — `tests/test_video_import.py` (20 cases): probing, strategies, dedup,
metadata preservation, cancellation, resume, invalid-video handling, planner
pre-analysis, and pipeline + COCO/YOLO export compatibility. Full suite green.
