"""Render desktop-app screenshots headlessly (Phase 11).

Builds the app on the offscreen Qt platform over a temp backend seeded with the
repo's sample_data, then grabs each key page to a PNG in docs/screenshots/. No
display required — runs in CI.

Run: QT_QPA_PLATFORM=offscreen python scripts/gui_screenshots.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from vds.config.settings import Settings, StorageSettings  # noqa: E402
from vds.container import Container  # noqa: E402
from vds.gui.controller import BackendController  # noqa: E402
from vds.gui.main_window import MainWindow  # noqa: E402
from vds.gui.settings import GuiSettings  # noqa: E402
from vds.gui.theme import ThemeManager  # noqa: E402

OUT = Path("docs/screenshots")


def _seed(container: Container) -> None:
    sample = Path("sample_data")
    if sample.exists():
        BackendController(container).import_dataset(str(sample), "sample-dataset")


def _grab(window: MainWindow, page: str, name: str) -> None:
    window.navigate(page)
    QApplication.instance().processEvents()
    OUT.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(OUT / name))
    print("wrote", OUT / name)


def _sample_clip(tmp: Path) -> Path:
    """A small multi-frame GIF (PIL-native, no ffmpeg) to demo the video importer."""
    from PIL import Image, ImageDraw

    frames = []
    for i in range(10):
        im = Image.new("RGB", (200, 140), (12, 14, 20))
        d = ImageDraw.Draw(im)
        x = int(i / 9 * 150)
        d.rectangle([x, 40, x + 44, 100], fill=(230 - i * 12, 90 + i * 12, 120))
        d.ellipse([150, (i * 12) % 100, 185, (i * 12) % 100 + 30], fill=(60, 200, 160))
        frames.append(im)
    path = tmp / "sample_clip.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0, disposal=2)
    return path


def _grab_video_import(window: MainWindow, container: Container, tmp: Path) -> None:
    from vds.gui.notifications import NotificationSystem
    from vds.gui.threads import ThreadManager
    from vds.gui.video_import_dialog import VideoImportDialog

    window.navigate("Dataset Manager")
    clip = _sample_clip(tmp)
    ctrl = BackendController(container)
    dialog = VideoImportDialog(ctrl, ThreadManager(), NotificationSystem(),
                               str(clip), "sample-video", parent=window)
    dialog._on_probed(ctrl.probe_video(str(clip)))  # populate synchronously
    dialog._strategy.setCurrentText("Every N Frames")
    dialog._on_planned(ctrl.plan_video(dialog._info, dialog._config()))
    dialog.show()
    QApplication.instance().processEvents()
    dialog.grab().save(str(OUT / "10_video_import.png"))
    print("wrote", OUT / "10_video_import.png")
    dialog.close()


def main() -> None:
    app = QApplication.instance() or QApplication([])
    tmp = Path(tempfile.mkdtemp())
    container = Container(
        settings=Settings(storage=StorageSettings(cas_root=tmp / "cas")),
        db_path=str(tmp / "shots.db"), artifacts_dir=tmp / "art",
    )
    _seed(container)

    gs = GuiSettings(settings=QSettings(str(tmp / "gui.ini"), QSettings.Format.IniFormat))
    theme = ThemeManager(app, initial="dark")
    window = MainWindow(BackendController(container), theme, gs)
    window.resize(1280, 820)
    window.show()
    app.processEvents()

    _grab(window, "Dashboard", "01_dashboard.png")
    _grab(window, "Dataset Manager", "02_dataset_manager.png")

    # Planner workspace: generate a plan synchronously so the panels are populated.
    window.navigate("Planner")
    planner = window._pages["Planner"]
    planner.on_show()
    pid = planner._dataset.currentData()
    if pid is not None:
        view = BackendController(container).plan_dataset(pid)
        planner._original = view
        planner._current = view
        planner._populate(view)
        planner._sync_controls(view)
    app.processEvents()
    OUT.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(OUT / "03_planner_workspace.png"))
    print("wrote", OUT / "03_planner_workspace.png")

    # Pipeline workspace: run synchronously, then populate the monitoring view.
    window.navigate("Annotation Pipeline")
    pipeline = window._pages["Annotation Pipeline"]
    pipeline._source, pipeline._name = "sample_data", "sample-run"
    report = BackendController(container).run_pipeline("sample_data", "sample-run")
    pipeline._on_done(report)
    app.processEvents()
    OUT.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(OUT / "05_pipeline_workspace.png"))
    print("wrote", OUT / "05_pipeline_workspace.png")

    # Verification workspace: load verdicts and select the first object.
    window.navigate("VLM Verification")
    verification = window._pages["VLM Verification"]
    verification.on_show()
    vpid = verification._dataset.currentData()
    if vpid is not None:
        ctrl = BackendController(container)
        verification._verdicts = ctrl.object_verdicts(vpid)
        verification._project_id = vpid
        verification._on_loaded(verification._verdicts)
        if verification._table.rowCount():
            verification._table.selectRow(0)
    app.processEvents()
    window.grab().save(str(OUT / "06_verification_workspace.png"))
    print("wrote", OUT / "06_verification_workspace.png")

    # Intelligence workspace: run the Analyst over the cached report synchronously.
    window.navigate("AI Dataset Analyst")
    intelligence = window._pages["AI Dataset Analyst"]
    intelligence.on_show()
    ipid = intelligence._dataset.currentData()
    if ipid is not None:
        intel = BackendController(container).analyze_dataset(ipid, "2026-07-10T12:00:00")
        intelligence._on_analyzed(intel)
    app.processEvents()
    window.grab().save(str(OUT / "07_intelligence_workspace.png"))
    print("wrote", OUT / "07_intelligence_workspace.png")

    # Knowledge Center: the analyze step above recorded a memory; load it synchronously.
    window.navigate("Engineering Memory")
    knowledge = window._pages["Engineering Memory"]
    knowledge._on_loaded(knowledge._load())
    app.processEvents()
    window.grab().save(str(OUT / "08_knowledge_center.png"))
    print("wrote", OUT / "08_knowledge_center.png")

    # Operations Center: recorded runs above; build the ops dashboard synchronously.
    window.navigate("Benchmark Center")
    operations = window._pages["Benchmark Center"]
    operations._refresh()  # synchronous load on the UI thread
    app.processEvents()
    window.grab().save(str(OUT / "09_operations_center.png"))
    print("wrote", OUT / "09_operations_center.png")

    # Video Import dialog: generate a sample clip, populate the dialog synchronously.
    _grab_video_import(window, container, tmp)

    theme.apply("light")
    app.processEvents()
    _grab(window, "Dashboard", "04_dashboard_light.png")
    window.close()


if __name__ == "__main__":
    main()
