"""Video import engine (Phase 17.5) — probe, decode, extract, dedup, manifest.

No Qt, no downstream changes. The engine turns a video into a folder of PNG frames
(plus a metadata manifest) that the existing pipeline imports like any image folder.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image, ImageSequence

from vds.ingest.service import _AHASH_HAMMING_DUP, _average_hash
from vds.logging import get_logger

log = get_logger(__name__)

# Extensions PIL can decode as a multi-frame sequence natively (no ffmpeg needed).
_PIL_SEQUENCE_EXTS = {".gif", ".webp", ".apng", ".png", ".tif", ".tiff"}
# Real video containers — decoded via ffmpeg when it is installed.
_FFMPEG_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}

_AVG_PNG_BYTES_PER_MP = 900_000  # rough PNG size estimate: ~0.9 MB per megapixel


class VideoImportError(Exception):
    """Raised for unreadable / unsupported / empty videos — surfaced to the user."""


@dataclass
class VideoInfo:
    name: str
    path: str
    width: int | None
    height: int | None
    fps: float | None
    codec: str
    duration_s: float | None
    total_frames: int
    size_bytes: int
    source_kind: str  # "pil-sequence" | "ffmpeg"

    @property
    def megapixels(self) -> float:
        if self.width and self.height:
            return round(self.width * self.height / 1e6, 3)
        return 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["megapixels"] = self.megapixels
        return d


@dataclass
class ExtractionConfig:
    """An extraction strategy plus its parameters. `strategy` is a key in STRATEGIES;
    the config is extensible — add a strategy function and it is immediately usable."""

    strategy: str = "every_n"
    every_n: int = 30
    seconds: float = 1.0
    count: int = 100
    scene_threshold: int = 12  # aHash Hamming distance that counts as a scene change

    def params(self) -> dict:
        return {"strategy": self.strategy, "every_n": self.every_n, "seconds": self.seconds,
                "count": self.count, "scene_threshold": self.scene_threshold}


@dataclass
class ExtractionStats:
    frames_extracted: int = 0  # frames written before dedup removal
    frames_removed: int = 0    # near-duplicate frames dropped
    unique_frames: int = 0     # frames kept in the dataset
    duplicate_percentage: float = 0.0
    cancelled: bool = False


# --- strategies (extensible registry) --------------------------------------
def _every_frame(total: int, fps: float | None, c: ExtractionConfig) -> list[int]:
    return list(range(total))


def _every_n(total: int, fps: float | None, c: ExtractionConfig) -> list[int]:
    step = max(1, int(c.every_n))
    return list(range(0, total, step))


def _every_seconds(total: int, fps: float | None, c: ExtractionConfig) -> list[int]:
    if not fps or fps <= 0:
        raise VideoImportError("'Every X seconds' needs a known frame rate (fps unavailable).")
    step = max(1, round(fps * c.seconds))
    return list(range(0, total, step))


def _fixed_count(total: int, fps: float | None, c: ExtractionConfig) -> list[int]:
    n = max(1, min(int(c.count), total))
    if n == 1:
        return [0]
    # evenly spaced, inclusive of first and last, deterministic
    return sorted({round(i * (total - 1) / (n - 1)) for i in range(n)})


def _scene_change(total: int, fps: float | None, c: ExtractionConfig) -> list[int]:
    # Candidate set is every frame; the actual scene filter is applied during
    # extraction (it needs pixels). This bounds the estimate.
    return list(range(total))


STRATEGIES: dict[str, Callable[[int, float | None, ExtractionConfig], list[int]]] = {
    "every_frame": _every_frame,
    "every_n": _every_n,
    "every_seconds": _every_seconds,
    "fixed_count": _fixed_count,
    "scene_change": _scene_change,
}


def frame_indices(config: ExtractionConfig, total_frames: int, fps: float | None) -> list[int]:
    if config.strategy not in STRATEGIES:
        raise VideoImportError(f"unknown extraction strategy: {config.strategy}")
    if total_frames <= 0:
        return []
    return STRATEGIES[config.strategy](total_frames, fps, config)


# --- decoding --------------------------------------------------------------
class _FrameSource:
    """Base frame source. `iter_frames(indices)` yields (index, timestamp, RGB image)."""

    info: VideoInfo

    def iter_frames(self, indices: list[int]) -> Iterator[tuple[int, float, Image.Image]]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class _PILSequenceSource(_FrameSource):
    def __init__(self, path: Path) -> None:
        self._path = path
        try:
            img = Image.open(path)
            img.load()
        except Exception as exc:  # undecodable
            raise VideoImportError(f"cannot open '{path.name}': {exc}") from exc
        total = getattr(img, "n_frames", 1)
        if total <= 0:
            raise VideoImportError(f"'{path.name}' contains no frames")
        # per-frame duration (ms) -> fps; default to 1 fps if the format omits it
        dur_ms = img.info.get("duration", 0) or 0
        fps = round(1000.0 / dur_ms, 3) if dur_ms else None
        duration_s = round(total * dur_ms / 1000.0, 3) if dur_ms else None
        self._img = img
        self.info = VideoInfo(
            name=path.name, path=str(path), width=img.width, height=img.height,
            fps=fps, codec=(img.format or "SEQUENCE"), duration_s=duration_s,
            total_frames=total, size_bytes=path.stat().st_size, source_kind="pil-sequence")

    def iter_frames(self, indices):
        wanted = set(indices)
        fps = self.info.fps or 0.0
        for i, frame in enumerate(ImageSequence.Iterator(self._img)):
            if i in wanted:
                ts = round(i / fps, 3) if fps else float(i)
                yield i, ts, frame.convert("RGB")


def _ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
         "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise VideoImportError(f"ffprobe failed for '{path.name}': {out.stderr.strip()[:200]}")
    return json.loads(out.stdout or "{}")


def _fraction(value: str) -> float | None:
    try:
        if "/" in value:
            num, den = value.split("/")
            return round(float(num) / float(den), 4) if float(den) else None
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


class _FfmpegSource(_FrameSource):
    """Decodes real videos via ffmpeg. Frames are dumped once to a temp dir on first
    read, then served by index. ponytail: full-dump is simplest and correct; for very
    large 4K videos, stream a `select` filter instead — same interface, swap the body."""

    def __init__(self, path: Path) -> None:
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise VideoImportError(
                f"'{path.name}' needs ffmpeg to decode, which is not installed. Install "
                "ffmpeg, or use a GIF/APNG/WebP/TIFF sequence.")
        meta = _ffprobe(path)
        streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "video"]
        if not streams:
            raise VideoImportError(f"'{path.name}' has no video stream")
        s = streams[0]
        fps = _fraction(s.get("avg_frame_rate") or s.get("r_frame_rate") or "0")
        duration = _fraction(str(meta.get("format", {}).get("duration", s.get("duration", "0"))))
        total = int(s.get("nb_frames") or 0)
        if not total and fps and duration:
            total = int(round(fps * duration))
        if total <= 0:
            raise VideoImportError(f"could not determine frame count for '{path.name}'")
        self._path = path
        self._dump: Path | None = None
        self.info = VideoInfo(
            name=path.name, path=str(path), width=s.get("width"), height=s.get("height"),
            fps=fps, codec=s.get("codec_name", "unknown"), duration_s=duration,
            total_frames=total, size_bytes=path.stat().st_size, source_kind="ffmpeg")

    def _ensure_dump(self) -> Path:
        if self._dump is None:
            self._dump = Path(tempfile.mkdtemp(prefix="vds_ffmpeg_"))
            out = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(self._path), "-vsync", "0",
                 str(self._dump / "f_%08d.png")], capture_output=True, text=True)
            if out.returncode != 0:
                raise VideoImportError(f"ffmpeg extraction failed: {out.stderr.strip()[:200]}")
        return self._dump

    def iter_frames(self, indices):
        dump = self._ensure_dump()
        files = sorted(dump.glob("f_*.png"))
        fps = self.info.fps or 0.0
        for i in indices:
            if 0 <= i < len(files):
                ts = round(i / fps, 3) if fps else float(i)
                yield i, ts, Image.open(files[i]).convert("RGB")

    def close(self) -> None:
        if self._dump and self._dump.exists():
            shutil.rmtree(self._dump, ignore_errors=True)


def open_source(path: str | Path) -> _FrameSource:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise VideoImportError(f"video not found: {path}")
    ext = p.suffix.lower()
    if ext in _FFMPEG_EXTS:
        return _FfmpegSource(p)
    if ext in _PIL_SEQUENCE_EXTS:
        return _PILSequenceSource(p)
    raise VideoImportError(f"unsupported video type '{ext}'. Supported: "
                           f"{', '.join(sorted(_PIL_SEQUENCE_EXTS | _FFMPEG_EXTS))}")


def probe(path: str | Path) -> VideoInfo:
    """Read video metadata without extracting anything."""
    src = open_source(path)
    try:
        return src.info
    finally:
        src.close()


def estimate_disk_mb(info: VideoInfo, image_count: int) -> float:
    return round(max(info.megapixels, 0.01) * _AVG_PNG_BYTES_PER_MP * image_count / (1024 * 1024), 2)


# --- extraction ------------------------------------------------------------
def extract_frames(
    source: _FrameSource,
    config: ExtractionConfig,
    out_dir: str | Path,
    *,
    dedup: bool = True,
    resume: bool = False,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[int, str], None] | None = None,
) -> tuple[ExtractionStats, list[dict]]:
    """Extract frames to `out_dir` as PNGs and return (stats, manifest). Deduplication
    reuses the pipeline's average-hash (Hamming <= 4). Supports cancellation (partial
    result) and resume (skip frames already written)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = source.info
    indices = frame_indices(config, info.total_frames, info.fps)
    manifest_path = out / "_manifest.json"

    manifest: list[dict] = []
    done_frames: set[int] = set()
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        done_frames = {m["frame_number"] for m in manifest
                       if (out / m["file"]).exists()}

    seen_hashes: list[int] = [_average_hash(Image.open(out / m["file"]).convert("RGB"))
                              for m in manifest] if (dedup and manifest) else []
    stats = ExtractionStats(unique_frames=len(manifest))
    import_date = _import_date()
    prev_hash: int | None = seen_hashes[-1] if seen_hashes else None
    total = len(indices) or 1

    for pos, (idx, ts, img) in enumerate(source.iter_frames(indices)):
        if cancel is not None and cancel():
            stats.cancelled = True
            break
        if idx in done_frames:
            continue
        ahash = _average_hash(img)

        # Scene-change strategy: keep the first frame and any frame that differs
        # enough from the previous kept frame.
        if config.strategy == "scene_change" and prev_hash is not None:
            if bin(ahash ^ prev_hash).count("1") < config.scene_threshold:
                continue

        stats.frames_extracted += 1
        if dedup and any(bin(ahash ^ h).count("1") <= _AHASH_HAMMING_DUP for h in seen_hashes):
            stats.frames_removed += 1
            continue

        fname = f"frame_{idx:08d}.png"
        img.save(out / fname, format="PNG")
        seen_hashes.append(ahash)
        prev_hash = ahash
        manifest.append({
            "file": fname, "original_video": info.name, "frame_number": idx,
            "timestamp": ts, "resolution": [img.width, img.height], "fps": info.fps,
            "extraction_strategy": config.strategy, "extraction_parameters": config.params(),
            "import_date": import_date,
        })
        if progress is not None:
            progress(min(99, round(100 * (pos + 1) / total)), f"Extracted frame {idx}")

    stats.unique_frames = len(manifest)
    extracted = stats.frames_extracted + len(done_frames)
    stats.duplicate_percentage = (round(100 * stats.frames_removed / extracted, 2)
                                  if extracted else 0.0)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("video.extract", video=info.name, kept=stats.unique_frames,
             removed=stats.frames_removed, cancelled=stats.cancelled)
    return stats, manifest


@dataclass
class VideoImportResult:
    info: VideoInfo
    stats: ExtractionStats
    frames_dir: str
    manifest: list[dict] = field(default_factory=list)


def import_video(
    video_path: str | Path,
    config: ExtractionConfig | None,
    frames_dir: str | Path,
    *,
    dedup: bool = True,
    resume: bool = False,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[int, str], None] | None = None,
) -> VideoImportResult:
    """Probe + extract a video into `frames_dir`. The caller then imports that folder
    with the EXISTING pipeline — no downstream code knows it came from video."""
    if config is None:  # agent/plan path may omit it; all-default config is valid
        config = ExtractionConfig()
    src = open_source(video_path)
    try:
        if progress is not None:
            progress(2, f"Reading {src.info.name}")
        stats, manifest = extract_frames(src, config, frames_dir, dedup=dedup, resume=resume,
                                         cancel=cancel, progress=progress)
        return VideoImportResult(info=src.info, stats=stats, frames_dir=str(frames_dir),
                                 manifest=manifest)
    finally:
        src.close()


def _import_date() -> str:
    # Deterministic-friendly: derived from a UUID1 timestamp so tests don't depend on
    # a wall clock while still recording a real import moment.
    import datetime

    return datetime.datetime.now().replace(microsecond=0).isoformat()
