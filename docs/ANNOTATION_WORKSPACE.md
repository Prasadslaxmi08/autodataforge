# Phase 19 — Annotation Workspace

A professional manual annotation editor that completes the primary workflow
(**Project → Import → Configure → Annotate → Review → Export**). It replaces the
placeholder "Annotation" page; the pipeline *monitor* moves to Developer Tools.

**The backend is frozen.** No service, pipeline, store, or API was changed. Editing
persists entirely through the two methods that already exist:

- **delete** → `AnnotationRepo.set_state(id, REJECTED)` (the exporter already excludes
  rejected states, `export/service.py`).
- **create** → `AnnotationRepo.add()` (state `accepted`).
- **edit** (move/resize/relabel) → reject the old row + `add()` a corrected one (`fixed`).

All of this is wrapped by GUI-layer methods on `BackendController`
(`project_images`, `image_boxes`, `save_edits`, `ai_annotate`, `resegment`,
`project_classes`, `rename/merge/delete_class`). AI actions reuse the model registry
(`container.models.get(DETECTOR|SEGMENTER)`).

## Layout

```
┌──────────── Toolbar (always visible) ─────────────────────────┐
│ Undo Redo Save | Prev Next | Zoom Fit Center | AI-Annotate Export │
├───────────┬──────────────────────────────┬────────────────────┤
│  Class    │                              │   Properties       │
│  Manager  │        Box Canvas            │  (label, conf,     │
│  (left)   │   (QGraphicsView, zoom/pan)  │   re-segment)      │
├───────────┴──────────────────────────────┴────────────────────┤
│                    Filmstrip (thumbnails)                       │
└────────────────────────────────────────────────────────────────┘
```

## Editing model

Edits happen in an **in-memory session**: the canvas holds boxes as plain dicts, undo/
redo are snapshots, and **Save** diffs the session against what was loaded and commits
the minimal set of create/delete/edit operations. This keeps the interactive canvas
fully decoupled from persistence.

- **Boxes**: drag on empty image to draw; drag body to move; drag an edge/corner to
  resize; Delete to remove; Ctrl+D to duplicate; relabel from the Properties panel.
- **Masks**: read-only overlay for the selected box; **Re-segment** regenerates it with
  the builtin segmenter. (Mask painting, polygon editing, and SAM2 need backend support
  and are intentionally out of scope — shown disabled.)
- **AI Annotate**: re-runs the configured detector on the current image and adds the
  proposals to the session (committed on Save).

## Class Manager

Distinct labels across the project's non-rejected annotations, each with a colour swatch
and a count. Enable/disable (view filter), search, **Rename**, **Merge**, **Delete** —
the last three are bulk relabel/reject operations over the whole project, run off the UI
thread.

## Confidence / review filter

Filters both the canvas and which boxes are emphasized: **All**, **Confidence < 30%**,
**Confidence < 50%**, **Needs review**, **Missing masks**, **Duplicate objects** (IoU >
0.85). Greatly reduces review time on large datasets.

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Delete` | Delete selected box |
| `Ctrl+Z` / `Ctrl+Y` | Undo / Redo |
| `Ctrl+S` | Save |
| `Ctrl+A` | Select all boxes |
| `Ctrl+D` | Duplicate selected box |
| `←` / `→` | Previous / Next image |
| `Esc` | Deselect |
| `Mouse wheel` | Zoom to cursor |
| `Middle-mouse drag` | Pan |

## Performance

Thumbnails are cached per image; the canvas uses `QGraphicsView` (hardware-friendly
scene rendering, cosmetic pens so outlines stay crisp at any zoom); the image auto-fits
until the user zooms/pans. Long operations (Save, AI-Annotate, class ops) run on the
shell's `ThreadManager` so the UI never freezes.

## Deliberately out of scope (needs backend / forbidden this phase)

- **SAM2 refinement, polygon editing, raster mask painting** — no backend support.
- **Persistence churn** — soft-delete leaves `rejected` tombstone rows and edited boxes
  get new ids. The clean alternative (`AnnotationRepo.update/delete`) is *not* taken: it
  would modify the frozen backend. A `ponytail:` note marks this in the controller.

## Verify

```bash
.venv/Scripts/python.exe -m pytest tests/test_gui.py -q   # editor tests
.venv/Scripts/python.exe -m vds.gui                        # launch
```

Open a project → **Annotation** → draw/move/resize boxes, wheel-zoom, pan, navigate the
filmstrip, filter by confidence, rename a class, **Save**, then **Export** and confirm
the exported dataset reflects the edits (rejected/edited boxes handled correctly).
Screenshots in [`images/`](images/) are captured headless (offscreen Qt), so text may
render as placeholder glyphs there; the layout is exact and text renders normally on a
real display.
