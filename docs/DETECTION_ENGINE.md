# Phase 18.5 — Production AI Detection Engine

AutoDataForge now detects with a **real Ultralytics YOLO model** instead of the
classical `BuiltinAdapter` blob finder. This is the single change that turns the app
from an image-processing demo into a real AI dataset generator. **Only the detector was
replaced** — pipeline, planner, analyst, verification, editor, review, export, GUI
architecture, controller, threads, and container are unchanged.

## Why

A full investigation proved the poor annotations came entirely from the model:
`BuiltinAdapter` thresholds pixels by brightness and boxes each connected blob, with a
**fabricated confidence** (`0.5 + 0.5·fill_ratio`). Parsing, NMS, coordinate scaling,
and rendering were all faithful — they carried through exactly what the blob detector
emitted. So the fix is to swap the model, nothing else.

## What changed

| File | Change |
|---|---|
| `vds/models/adapters/yolo.py` | Rewritten as a production adapter (detect **and** segment). |
| `vds/models/adapters/yolo_config.py` | New — GUI-settable runtime knobs (model/conf/iou/imgsz/segment). |
| `vds.toml` | Default `detector`/`segmenter` → `YoloAdapter`. |
| `vds/gui/widgets/import_wizard.py` | Advanced section: Model, Confidence, IoU, Max size, Segmentation. |
| `vds/gui/controller.py`, `settings.py` | `set_detector_config(...)`; remembered detector prefs. |
| `tests/conftest.py` | Pins builtin via env for the offline test suite. |

The classical `BuiltinAdapter` **remains in the repo for tests only**.

## Architecture (unchanged seam)

The pipeline calls `detector.detect(images, prompts, params)` then, per detection,
`segmenter.segment(image, [box])` (labeler.py). `YoloAdapter` implements **both**
capabilities and runs inference **once**: `detect()` caches each image's `Results`, and
`segment()` reads that cache and rasterizes the matching instance mask — no double
inference, no pipeline change.

```
Image / Video ─▶ YOLO (one forward pass) ─▶ boxes + confidence + (masks)
                                              │
                    detect() ────────────────┤ cache Results by image hash
                    segment() ◀───────────────┘ reuse cached masks
                                              ▼
              internal Box2D / Mask ─▶ Annotation Editor ─▶ Review ─▶ Export (COCO/YOLO)
```

Coordinates: YOLO returns `xyxy` already in **original-image pixels** (it reverses its
own letterboxing), converted to `Box2D(x, y, w, h)`. The existing faithful path (labeler
stores verbatim, editor renders the same pixels) then carries it through unchanged.

## Model selection

The Import Wizard's **Advanced** section (collapsed by default) offers:

- **Model**: YOLO11n (default) · YOLO11s · YOLO11m · **Custom `.pt`**
- **Detection confidence**, **IoU threshold**, **Max image size**
- **Segmentation** (uses a `-seg` model; produces instance masks)

Choices are remembered in `GuiSettings` and applied via a small process-global the
adapter reads before each run — so the frozen pipeline needs no new parameters. The
plugin *selection* stays config-driven (`vds.toml`).

Supported weights: YOLOv8 / YOLO11 detection & segmentation `.pt` (pose-compatible).
`.onnx` / TensorRT are accepted as a **future interface** (not part of this phase's
validated path).

## Confidence

**Confidence comes from the model** (`boxes.conf`), never fabricated. The wizard's
confidence is passed to the model as the inference floor.

## Device

Auto: CUDA when a CUDA-enabled torch is present, else CPU. This phase ships with **CPU
torch** (reliable everywhere; YOLO11n ≈ 50–150 ms/image on CPU). Installing a
Blackwell-capable CUDA torch later enables the RTX 5060 GPU with no code change.

## Install

```bash
pip install ultralytics        # optional extra: pip install -e ".[detect]"
```

Without `ultralytics`, `YoloAdapter.load()` raises a clear `ConfigError`; pin the
classical backend with `VDS_MODELS__DETECTOR=vds.models.adapters.builtin:BuiltinAdapter`.

## Evidence (builtin → YOLO)

Probe resolves the detector exactly as the shipped app does (`Container` → registry):
`registry.load_adapter path=vds.models.adapters.yolo:YoloAdapter`,
`BuiltinAdapter executing? False`.

| | BuiltinAdapter (before) | YOLO11n (after) |
|---|---|---|
| Detector | classical blobs | real object detector |
| Boxes per frame | loose regions + edge blobs | one tight box per object |
| Confidence | fabricated `0.5+0.5·fill` | model score |
| Full-frame / border blobs | yes | none |

Raw detections, YOLO11n @ conf 0.25 (xywh in original-image pixels, area = % of frame):

| Image | Object | conf | box (x,y,w,h) | area |
|---|---|---|---|---|
| `bus.jpg` 810×1080 | bus | 0.940 | (4,229,792,499) | 45% |
| | person | 0.888 | (671,395,139,484) | 8% |
| | person | 0.878 | (47,400,192,505) | 11% |
| | person | 0.856 | (223,409,121,452) | 6% |
| | person | 0.622 | (0,556,69,316) | 2% |
| `Ship/frame_00000000.png` 640×480 | boat | 0.805 | (183,250,304,79) | 8% |
| `Ship/frame_00000210.png` 640×480 | boat | 0.708 | (167,252,306,81) | 8% |

One physical object → one tight box, model confidence, no full-frame/border blobs.
(Contrast: on `frame_00000000` the builtin returned a single loose ~28%-of-frame region.)

**Detect-only correctness:** YOLO11n has no masks, so `segment()` returns `None` (no mask),
*not* a non-None empty mask. An empty mask trips `RuleBasedVerifier`'s empty-mask rejection
and would silently drop every detect-only annotation; with `None` the verifier judges by
confidence, so detect-only models export normally. Proven by `test_yolo_pipeline_end_to_end`
(bus.jpg → exportable, model-confident `person`/`bus` annotations).

## Tests

`tests/test_yolo.py` (skipped unless `ultralytics` is installed): loading, custom model,
real detections with model confidence, COCO class names, batch inference, segmentation
masks, detect-only graceful empty mask, and a full-pipeline end-to-end that exports
model-confident annotations. The 239-strong existing suite stays green on builtin via
the conftest env pin.
