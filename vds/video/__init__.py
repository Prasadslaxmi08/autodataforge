"""Video Dataset Import Engine (Phase 17.5).

Transforms videos into standard image datasets that enter the EXISTING
Planner → Annotation → Verification → Analyst pipeline unchanged. A video is just
another dataset source: every extracted frame becomes a normal dataset image. Nothing
below the import layer is modified.

Decoding is native where possible — PIL reads multi-frame sequences (GIF/APNG/WebP/
multi-page TIFF) with no extra dependency — and shells out to ffmpeg/ffprobe for real
codecs (MP4/MOV/…) when they are installed. Deduplication reuses the pipeline's own
average-hash so video frames are deduped exactly like folder imports.
"""

from vds.video.engine import (
    STRATEGIES,
    ExtractionConfig,
    ExtractionStats,
    VideoImportError,
    VideoInfo,
    extract_frames,
    frame_indices,
    import_video,
    open_source,
    probe,
)

__all__ = [
    "STRATEGIES",
    "ExtractionConfig",
    "ExtractionStats",
    "VideoImportError",
    "VideoInfo",
    "extract_frames",
    "frame_indices",
    "import_video",
    "open_source",
    "probe",
]
