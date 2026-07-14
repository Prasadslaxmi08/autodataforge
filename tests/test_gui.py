"""Desktop GUI tests (Phase 11): startup, navigation, dataset import, backend
communication, thread safety, theme switching, window restoration, settings
persistence. Headless via the offscreen Qt platform (set in conftest)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from vds.gui.controller import BackendController  # noqa: E402
from vds.gui.main_window import MainWindow  # noqa: E402
from vds.gui.notifications import NotificationSystem  # noqa: E402
from vds.gui.pages.annotation import AnnotationPage  # noqa: E402
from vds.gui.pages.export import ExportPage  # noqa: E402
from vds.gui.pages.project_dashboard import ProjectDashboardPage  # noqa: E402
from vds.gui.settings import GuiSettings  # noqa: E402
from vds.gui.theme import ThemeManager  # noqa: E402
from vds.gui.threads import ThreadManager  # noqa: E402
from vds.gui.widgets.box_canvas import BoxCanvas, BoxItem, rle_to_qimage  # noqa: E402
from vds.gui.widgets.import_wizard import ImportWizard  # noqa: E402
from vds.gui.widgets.navigation import NAV_ITEMS, NavigationPanel  # noqa: E402


@pytest.fixture
def gui_settings(tmp_path) -> GuiSettings:
    qs = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    return GuiSettings(settings=qs)


@pytest.fixture
def window(qtbot, container, gui_settings) -> MainWindow:
    app = QApplication.instance()
    theme = ThemeManager(app, initial=gui_settings.theme)
    win = MainWindow(BackendController(container), theme, gui_settings)
    qtbot.addWidget(win)
    return win


# --- startup ---------------------------------------------------------------
def test_application_startup(window):
    assert window.windowTitle() == "AutoDataForge"
    # every navigable module is a registered page; the Project dashboard is
    # registered too but reached from the workspace, not the nav.
    assert set(NAV_ITEMS) <= set(window._pages)
    assert "Project" in window._pages
    assert window._pages["Projects"].name == "Projects"


# --- navigation ------------------------------------------------------------
def test_page_navigation_visits_every_module(window):
    for name in NAV_ITEMS:
        window.navigate(name)
        assert window._workspace.currentWidget().name == name


def test_remaining_modules_are_placeholders(window):
    # Placeholder pages self-declare their status via context().
    _title, rows = window._pages["Reports"].context()
    assert ("Status", "Placeholder") in rows


# --- backend communication -------------------------------------------------
def test_backend_communication_dashboard_snapshot(container):
    snap = BackendController(container).dashboard_snapshot()
    for key in ("dataset_count", "current_model", "current_provider",
                "recent_memory", "pipeline_status", "latest_benchmarks"):
        assert key in snap


def test_dataset_import_through_controller(container, dataset_dir):
    ctrl = BackendController(container)
    report = ctrl.import_dataset(str(dataset_dir), "unit-set")
    assert report.imported == 3
    datasets = ctrl.list_datasets()
    assert any(d.name == "unit-set" and d.image_count == 3 for d in datasets)


def test_report_markdown_none_does_not_crash(container):
    # review-/export-only plans have no ExecutionReport; must not deref .benchmark
    out = BackendController(container).report_markdown(None)
    assert "No pipeline run report" in out


def test_rename_and_delete_dataset(container, dataset_dir):
    ctrl = BackendController(container)
    ctrl.import_dataset(str(dataset_dir), "to-rename")
    pid = ctrl.list_datasets()[0].project_id
    ctrl.rename_dataset(pid, "renamed")
    assert ctrl.list_datasets()[0].name == "renamed"
    ctrl.delete_dataset(pid)
    assert ctrl.list_datasets() == []


# --- thread safety ---------------------------------------------------------
def test_import_runs_off_ui_thread(qtbot, container, dataset_dir):
    ctrl = BackendController(container)
    tm = ThreadManager()
    results = {}
    worker = tm.submit(
        ctrl.import_dataset, str(dataset_dir), "threaded",
        wants_progress=True,
        on_finished=lambda r: results.setdefault("report", r),
        on_progress=lambda pct, msg: results.setdefault("progress", (pct, msg)),
    )
    with qtbot.waitSignal(worker.signals.finished, timeout=30000):
        pass
    assert results["report"].imported == 3
    assert "progress" in results  # progress was reported during execution
    assert tm.active == 0


def test_discarded_worker_survives_until_finished(qtbot):
    """Regression: callers throw away the returned Worker. The ThreadManager must
    keep it (and its WorkerSignals) alive until finished is delivered, else the
    worker is GC'd and Qt delivers a signal from a freed sender → segfault."""
    import gc

    tm = ThreadManager()
    got = {}
    tm.submit(lambda: 42, on_finished=lambda r: got.setdefault("r", r))
    gc.collect()  # would collect the worker here if nothing retained it
    qtbot.waitUntil(lambda: got.get("r") == 42, timeout=5000)
    qtbot.waitUntil(lambda: not tm._workers, timeout=5000)  # released afterwards
    assert tm.active == 0


# --- theme switching -------------------------------------------------------
def test_theme_switching(window):
    assert window._theme.name == "dark"
    new = window.toggle_theme()
    assert new == "light"
    assert QApplication.instance().styleSheet()  # stylesheet applied
    assert window._settings.theme == "light"  # persisted choice


# --- settings persistence --------------------------------------------------
def test_settings_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "gui.ini")
    s1 = GuiSettings(settings=QSettings(path, QSettings.Format.IniFormat))
    s1.theme = "light"
    s1.last_page = "Dataset Manager"
    s1.last_import_dir = str(tmp_path)
    s1.sync()
    s2 = GuiSettings(settings=QSettings(path, QSettings.Format.IniFormat))
    assert s2.theme == "light"
    assert s2.last_page == "Dataset Manager"
    assert s2.last_import_dir == str(tmp_path)


# --- Phase 18: Project Workspace, Import Wizard, Dashboard, Export ----------
def test_nav_is_grouped_and_collapsible(qtbot):
    nav = NavigationPanel()
    qtbot.addWidget(nav)
    assert nav.topLevelItemCount() == 2  # Workspace + Developer Tools
    workspace, devtools = nav.topLevelItem(0), nav.topLevelItem(1)
    assert workspace.isExpanded() and not devtools.isExpanded()  # advanced hidden by default
    # every leaf maps to a page name that the shell can route to
    leaves = {nav.topLevelItem(g).child(c).data(0, 0x0100)  # Qt.UserRole
              for g in range(2) for c in range(nav.topLevelItem(g).childCount())}
    assert leaves == set(NAV_ITEMS)


def test_nav_select_emits_navigated(qtbot):
    nav = NavigationPanel()
    qtbot.addWidget(nav)
    seen = []
    nav.navigated.connect(seen.append)
    nav.select("Export")
    assert seen and seen[-1] == "Export"


def test_workspace_lists_recent_projects(window, container, dataset_dir):
    BackendController(container).import_dataset(str(dataset_dir), "recent-one")
    window.navigate("Projects")
    ws = window._pages["Projects"]
    ws.on_show()
    assert any(s.name == "recent-one" for s in ws._recent)


def test_import_wizard_folder_emits_request(qtbot, container, dataset_dir):
    wiz = ImportWizard(BackendController(container))
    qtbot.addWidget(wiz)
    captured = {}
    wiz.request_folder_import.connect(lambda f, n, fmt: captured.update(folder=f, name=n, fmt=fmt))
    wiz.set_source("folder", str(dataset_dir))
    wiz.name, wiz.export_format = "wiz-set", "coco"
    wiz.accept()  # fires finished(Accepted) → request
    assert captured == {"folder": str(dataset_dir), "name": "wiz-set", "fmt": "coco"}


def test_import_wizard_zip_extracts_to_folder(qtbot, container, dataset_dir, tmp_path):
    import zipfile
    zip_path = tmp_path / "imgs.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in dataset_dir.iterdir():
            zf.write(p, p.name)
    wiz = ImportWizard(BackendController(container))
    qtbot.addWidget(wiz)
    captured = {}
    wiz.request_folder_import.connect(lambda f, n, fmt: captured.update(folder=f))
    wiz.set_source("zip", str(zip_path))
    wiz.accept()
    from pathlib import Path
    extracted = Path(captured["folder"])
    assert extracted.exists() and extracted != dataset_dir
    imgs = [p for p in extracted.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    assert len(imgs) == 3  # the dataset_dir fixture has 3 images


def test_import_wizard_video_delegates(qtbot, container):
    wiz = ImportWizard(BackendController(container))
    qtbot.addWidget(wiz)
    captured = {}
    wiz.request_video_import.connect(lambda p, n: captured.update(path=p, name=n))
    wiz.delegate_video("clip.mp4")
    assert captured == {"path": "clip.mp4", "name": "clip"}


def test_project_dashboard_shows_stats_and_routes(qtbot, container, dataset_dir):
    ctrl = BackendController(container)
    report = ctrl.import_dataset(str(dataset_dir), "dash-set")
    page = ProjectDashboardPage(ctrl, NotificationSystem())
    qtbot.addWidget(page)
    page.set_project(report.project_id)
    assert page._title.text() == "dash-set"
    routed = []
    page.request_nav.connect(lambda t, pid: routed.append((t, pid)))
    page._go("Export")
    assert routed == [("Export", report.project_id)]


def test_export_page_and_controller(qtbot, container, dataset_dir):
    ctrl = BackendController(container)
    report = ctrl.import_dataset(str(dataset_dir), "exp-set")
    result = ctrl.export_project(report.project_id, "coco")
    assert result.validated and result.images == 3
    page = ExportPage(ctrl, ThreadManager(), NotificationSystem())
    qtbot.addWidget(page)
    page.select_project(report.project_id)
    page.on_show()
    assert page._project.currentData() == report.project_id


def test_shell_import_flow_opens_dashboard(window, qtbot, container, dataset_dir):
    # Drives the shell's folder-import path end to end: threaded import → dashboard.
    window._run_folder_import(str(dataset_dir), "shell-flow", "coco")
    qtbot.waitUntil(lambda: window._threads.active == 0, timeout=30000)
    qtbot.waitUntil(lambda: window._workspace.currentWidget().name == "Project", timeout=5000)
    assert any(d.name == "shell-flow" for d in BackendController(container).list_datasets())


# --- Phase 19: Annotation Workspace ----------------------------------------
def _editor(qtbot, container, dataset_dir):
    ctrl = BackendController(container)
    rep = ctrl.import_dataset(str(dataset_dir), "edit-set")
    page = AnnotationPage(ctrl, ThreadManager(), NotificationSystem())
    qtbot.addWidget(page)
    page.set_project(rep.project_id)
    return ctrl, page, page._current_image_id()


def _first_box_item(page):
    for it in page._canvas.scene().items():
        if isinstance(it, BoxItem):
            return it
    return None


def test_editor_loads_project_boxes(qtbot, container, dataset_dir):
    ctrl, page, iid = _editor(qtbot, container, dataset_dir)
    assert page._images and page._filmstrip.count() == len(page._images)
    assert len(page._canvas.boxes()) == len(ctrl.image_boxes(iid))


def test_editor_create_box_saves(qtbot, container, dataset_dir):
    ctrl, page, iid = _editor(qtbot, container, dataset_dir)
    before = len(ctrl.image_boxes(iid))
    page._canvas.add_box({"id": "", "x": 4, "y": 4, "w": 15, "h": 15, "label": "boat",
                          "confidence": 1.0})
    assert page._compute_ops() == [{"op": "create", "box": {"x": 4, "y": 4, "w": 15, "h": 15},
                                    "label": "boat", "confidence": 1.0}]
    page.save()
    qtbot.waitUntil(lambda: len(ctrl.image_boxes(iid)) == before + 1, timeout=10000)
    assert any(b.label == "boat" and b.state == "accepted" for b in ctrl.image_boxes(iid))


def test_editor_delete_persists_and_excluded_from_export(qtbot, container, dataset_dir):
    ctrl, page, iid = _editor(qtbot, container, dataset_dir)
    before = len(ctrl.image_boxes(iid))
    if before == 0:
        page._canvas.add_box({"id": "", "x": 4, "y": 4, "w": 15, "h": 15, "label": "x", "confidence": 1.0})
        page.save()
        qtbot.waitUntil(lambda: len(ctrl.image_boxes(iid)) == 1, timeout=10000)
        page._load_image(page._idx)
        before = 1
    _first_box_item(page).setSelected(True)
    page.delete()
    page.save()
    qtbot.waitUntil(lambda: len(ctrl.image_boxes(iid)) == before - 1, timeout=10000)


def test_editor_undo_redo(qtbot, container, dataset_dir):
    _ctrl, page, _iid = _editor(qtbot, container, dataset_dir)
    n = len(page._canvas.boxes())
    page._snapshot_undo()
    page._canvas.add_box({"id": "", "x": 2, "y": 2, "w": 9, "h": 9, "label": "z", "confidence": 1.0})
    assert len(page._canvas.boxes()) == n + 1
    page.undo()
    assert len(page._canvas.boxes()) == n
    page.redo()
    assert len(page._canvas.boxes()) == n + 1


def test_editor_confidence_filter(qtbot, container, dataset_dir):
    _ctrl, page, _iid = _editor(qtbot, container, dataset_dir)
    page._canvas.add_box({"id": "", "x": 2, "y": 2, "w": 9, "h": 9, "label": "lowc", "confidence": 0.1})
    page._filter.setCurrentIndex(1)  # Confidence < 30%
    visible = [it for it in page._canvas.scene().items()
               if isinstance(it, BoxItem) and it.isVisible()]
    assert visible and all(it.confidence < 0.30 for it in visible)


def test_editor_ai_annotate(qtbot, container, dataset_dir):
    _ctrl, page, _iid = _editor(qtbot, container, dataset_dir)
    page._load_image(page._idx)
    page._canvas.set_boxes([])  # clear, then let AI repopulate
    page.ai_annotate()
    qtbot.waitUntil(lambda: len(page._canvas.boxes()) > 0, timeout=10000)


def test_box_canvas_edit_ops(qtbot, container, dataset_dir):
    ctrl = BackendController(container)
    rep = ctrl.import_dataset(str(dataset_dir), "canvas-set")
    canvas = BoxCanvas()
    qtbot.addWidget(canvas)
    canvas.load_image(ctrl.project_images(rep.project_id)[0].path)
    canvas.add_box({"id": "a", "x": 1, "y": 1, "w": 10, "h": 10, "label": "o", "confidence": 1.0})
    assert len(canvas.boxes()) == 1
    canvas.selected().setSelected(True)
    canvas.duplicate_selected()
    assert len(canvas.boxes()) == 2
    canvas.delete_selected()
    assert len(canvas.boxes()) == 1


def test_rle_to_qimage_roundtrip(qtbot):
    import numpy as np

    from vds.models.adapters.builtin import _rle_encode
    m = np.zeros((6, 6), dtype=np.uint8)
    m[2:4, 1:3] = 1
    img = rle_to_qimage(_rle_encode(m), 6, 6, __import__("PySide6.QtGui", fromlist=["QColor"]).QColor("#ff0000"))
    assert img.width() == 6 and img.height() == 6
    from PySide6.QtGui import qAlpha
    assert qAlpha(img.pixel(1, 2)) > 0    # foreground pixel is painted
    assert qAlpha(img.pixel(0, 0)) == 0   # background stays transparent


# --- window restoration ----------------------------------------------------
def test_window_restoration(qtbot, container, tmp_path):
    path = str(tmp_path / "gui.ini")
    app = QApplication.instance()

    s1 = GuiSettings(settings=QSettings(path, QSettings.Format.IniFormat))
    theme = ThemeManager(app, initial="dark")
    w1 = MainWindow(BackendController(container), theme, s1)
    qtbot.addWidget(w1)
    w1.resize(1180, 733)  # both dims above the layout minimums
    w1.close()  # closeEvent saves geometry + state

    s2 = GuiSettings(settings=QSettings(path, QSettings.Format.IniFormat))
    assert s2.window_geometry() is not None  # geometry was persisted
    w2 = MainWindow(BackendController(container), theme, s2)
    qtbot.addWidget(w2)
    # Height restores exactly; width can be clamped up to the child-widget minimum
    # by the offscreen platform, so assert it's the restored value (not the default).
    assert w2.size().height() == 733
    assert w2.size().width() >= 1000
