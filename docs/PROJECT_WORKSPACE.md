# Phase 18 — Project Workspace

A UX redesign that makes the desktop app feel like a commercial dataset-generation
product instead of an engineering console. **No backend, pipeline, or service was
changed** — this phase only reorganizes and reskins the GUI shell over the existing
`BackendController`.

## The main flow

```
Projects ──▶ Import ──▶ (pipeline runs) ──▶ Project Dashboard ──▶ Annotation · Review · Export
```

A first-time user sees only this. Everything advanced is one collapse away.

## What's new

### Project Workspace — the landing page  (`vds/gui/pages/workspace.py`)
The home screen. A hero header, three large action cards — **Create Project**,
**Import Dataset** (both open the Import Wizard), **Continue Project** (resumes the
most recent one) — and a **Recent Projects** grid of thumbnail cards read from
`controller.list_datasets(thumbnails=1)`. Drag & drop a folder, a `.zip`, or a video
onto the page to start an import preloaded with that path.

### Import Wizard  (`vds/gui/widgets/import_wizard.py`)
A guided four-step flow replacing the bare file dialogs:

1. **Source** — Folder · ZIP · Video (COCO / YOLO are shown disabled; the backend has
   no dataset-label ingest and this phase does not add one).
2. **Preview** — image count, estimated storage, duplicate note.
3. **Configure** — dataset name, detector, export format.
4. **Summary** — review, then **Begin Import**.

The wizard only *collects inputs*; threading stays in the shell (the one place it
lives). On finish it emits a request and `MainWindow` runs the existing
`import_dataset` on a `ThreadManager` worker. **ZIP** is unzipped to a temp folder and
handed to the normal folder import (a UI-only shim — no backend change). **Video**
delegates to the existing, unchanged `VideoImportDialog`.

### Project Dashboard  (`vds/gui/pages/project_dashboard.py`)
Opened from the workspace. Shows a project's stats (via the existing
`dataset_detail`) as metric tiles and the next-step actions **Start Annotation**,
**Review Dataset**, **Export Dataset**, plus **Rename / Delete**. It runs nothing
itself — the buttons emit `request_nav(page, project_id)` and the shell routes to the
Annotation Pipeline / VLM Verification / Export pages.

### Export page  (`vds/gui/pages/export.py`)
Pick a project + format → re-export via the existing `ExportService`
(`controller.export_project` → `container.exporter.run`), off-thread.

### Navigation  (`vds/gui/widgets/navigation.py`)
A grouped `QTreeWidget` replacing the flat list:

| Group | Items |
|-------|-------|
| **Workspace** (always open) | Projects · Annotation · Review · Export |
| **Developer Tools** (collapsible, collapsed by default) | Planner · Dataset Intelligence · Knowledge Center · Benchmark Center · Dashboard · Reports · Settings |

Friendly labels map to existing page names (e.g. *Annotation* → `Annotation Pipeline`,
*Review* → `VLM Verification`). The shell contract is unchanged: the panel still emits
`navigated(page_name)` and exposes `select(page_name)`, so `MainWindow` was barely
touched. No feature was removed — everything the old flat nav reached is still one
click away under Developer Tools.

## Constraints honored

- **No backend / pipeline change.** The monolithic `Phase1Pipeline.run()` is called
  exactly as before; the only new controller method, `export_project`, is a thin
  pass-through to the existing `ExportService`.
- **All services reused.** Import, video import, detector/export options, dataset
  stats, rename/delete, and export all go through the existing `BackendController`.
- **Advanced features hidden, not removed.**

## Not included (would require backend work, out of scope this phase)

- **COCO / YOLO dataset ingest** (re-using their labels) — no ingest path exists;
  shown as disabled "coming soon" tiles.
- **Empty projects** — the pipeline creates a project only by importing data, so
  "Create Project" opens the Import Wizard (create == import).

## Verify

```bash
.venv/Scripts/python.exe -m pytest tests/test_gui.py -q   # GUI suite
.venv/Scripts/python.exe -m vds.gui                        # launch the app
```

Then: land on the Project Workspace → **Import Dataset** → pick `sample_data` → walk
the wizard → land on the Project Dashboard → try Annotation / Review / Export →
expand/collapse **Developer Tools** → toggle dark/light.

Screenshots in [`images/`](images/) (`project_workspace_dark.png`,
`project_workspace_light.png`, `import_wizard.png`) are captured headless (offscreen
Qt), so text may render as placeholder glyphs there; the layout is exact and text
renders normally on a real display.
