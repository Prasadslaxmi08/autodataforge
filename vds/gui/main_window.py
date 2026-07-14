"""MainWindow — the application shell (Phase 11).

Assembles the whole layout and wires the cross-cutting managers:

  ┌───────┬──────────────────────────────┬─────────┐
  │  Nav  │        Workspace (pages)      │ Context │
  │ (left)│      QStackedWidget           │ (right) │
  │       ├──────────────────────────────┴─────────┤
  │       │        Bottom panel (log / tasks / res) │
  └───────┴──────────────────────────────┬─────────┘
                                     Status bar

The shell owns navigation, notification routing, resource sampling, window
restoration, and theme switching. Pages never reference each other or the
backend directly — only their injected controller/threads/notifications.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMainWindow, QSplitter, QStackedWidget, QWidget

from vds.gui.controller import BackendController
from vds.gui.notifications import NotificationSystem
from vds.gui.pages.annotation import AnnotationPage
from vds.gui.pages.base import Page
from vds.gui.pages.dashboard import DashboardPage
from vds.gui.pages.export import ExportPage
from vds.gui.pages.intelligence import IntelligencePage
from vds.gui.pages.knowledge import KnowledgePage
from vds.gui.pages.operations import OperationsPage
from vds.gui.pages.pipeline import PipelinePage
from vds.gui.pages.placeholder import PLACEHOLDER_SPECS, make_placeholder
from vds.gui.pages.planner import PlannerPage
from vds.gui.pages.project_dashboard import ProjectDashboardPage
from vds.gui.pages.verification import VerificationPage
from vds.gui.pages.workspace import ProjectWorkspacePage
from vds.gui.settings import GuiSettings
from vds.gui.theme import ThemeManager
from vds.gui.threads import ThreadManager
from vds.gui.widgets.bottom_panel import BottomPanel
from vds.gui.widgets.navigation import NavigationPanel
from vds.gui.widgets.resources import ResourceMonitor
from vds.gui.widgets.right_sidebar import ContextSidebar
from vds.gui.widgets.status_bar import StatusBar


class MainWindow(QMainWindow):
    def __init__(
        self,
        controller: BackendController,
        theme: ThemeManager,
        gui_settings: GuiSettings,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._theme = theme
        self._settings = gui_settings
        self._threads = ThreadManager()
        self._notify = NotificationSystem()

        self.setWindowTitle("AutoDataForge")
        self.resize(1280, 820)
        self.setMinimumSize(1000, 640)

        self._nav = NavigationPanel()
        self._workspace = QStackedWidget()
        self._context = ContextSidebar()
        self._bottom = BottomPanel(controller.container.settings.gpu.device,
                                   controller.container.settings.gpu.vram_budget_mb)
        self._status = StatusBar()
        self.setStatusBar(self._status)

        self._pages: dict[str, Page] = {}
        self._build_pages()
        self._build_layout()
        self._build_menu()
        self._wire()

        self._monitor = ResourceMonitor()
        self._monitor.sampled.connect(self._on_resources)
        self._monitor.start()

        self._restore_window()
        self._nav.select(self._settings.last_page)

    # --- pages ---
    def _build_pages(self) -> None:
        self._add_page(ProjectWorkspacePage(self._controller))
        self._add_page(ProjectDashboardPage(self._controller, self._notify))
        self._add_page(AnnotationPage(self._controller, self._threads, self._notify))
        self._add_page(ExportPage(self._controller, self._threads, self._notify))
        self._add_page(DashboardPage(self._controller))
        self._add_page(PlannerPage(self._controller, self._threads, self._notify))
        self._add_page(PipelinePage(self._controller, self._threads, self._notify))
        self._add_page(VerificationPage(self._controller, self._threads, self._notify))
        self._add_page(IntelligencePage(self._controller, self._threads, self._notify))
        self._add_page(KnowledgePage(self._controller, self._threads, self._notify))
        self._add_page(OperationsPage(self._controller, self._threads, self._notify))
        for name, (subtitle, sections) in PLACEHOLDER_SPECS.items():
            self._add_page(make_placeholder(name, subtitle, sections)())

    def _add_page(self, page: Page) -> None:
        self._pages[page.name] = page
        self._workspace.addWidget(page)

    # --- layout ---
    def _build_layout(self) -> None:
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self._workspace)
        top.addWidget(self._context)
        top.setStretchFactor(0, 1)
        top.setStretchFactor(1, 0)
        top.setSizes([980, 260])

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(top)
        right.addWidget(self._bottom)
        right.setStretchFactor(0, 1)
        right.setStretchFactor(1, 0)
        right.setSizes([620, 180])

        main = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(self._nav)
        main.addWidget(right)
        main.setStretchFactor(0, 0)
        main.setStretchFactor(1, 1)
        main.setCollapsible(0, False)

        container = QWidget()
        from PySide6.QtWidgets import QHBoxLayout

        lay = QHBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(main)
        self.setCentralWidget(container)

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        act_import = QAction("&Import Dataset…", self)
        act_import.setShortcut(QKeySequence("Ctrl+I"))
        act_import.triggered.connect(self._quick_import)
        file_menu.addAction(act_import)
        file_menu.addSeparator()
        act_quit = QAction("E&xit", self)
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view = bar.addMenu("&View")
        act_theme = QAction("Toggle &Theme", self)
        act_theme.setShortcut(QKeySequence("Ctrl+T"))
        act_theme.triggered.connect(self.toggle_theme)
        view.addAction(act_theme)

    # --- wiring ---
    def _wire(self) -> None:
        self._nav.navigated.connect(self.navigate)
        self._notify.notified.connect(self._on_notify)
        workspace = self._pages["Projects"]
        assert isinstance(workspace, ProjectWorkspacePage)
        workspace.start_import.connect(self._open_import_wizard)
        workspace.open_project.connect(self._open_project)
        project = self._pages["Project"]
        assert isinstance(project, ProjectDashboardPage)
        project.request_nav.connect(self._route_from_project)
        project.changed.connect(self._on_project_changed)
        export = self._pages["Export"]
        assert isinstance(export, ExportPage)
        export.busy.connect(self._on_busy)
        annotation = self._pages["Annotation"]
        assert isinstance(annotation, AnnotationPage)
        annotation.busy.connect(self._on_busy)
        annotation.export_requested.connect(lambda pid: self._route_from_project("Export", pid))
        planner = self._pages["Planner"]
        assert isinstance(planner, PlannerPage)
        planner.busy.connect(self._on_busy)
        pipeline = self._pages["Annotation Pipeline"]
        assert isinstance(pipeline, PipelinePage)
        pipeline.busy.connect(self._on_busy)
        verification = self._pages["VLM Verification"]
        assert isinstance(verification, VerificationPage)
        verification.busy.connect(self._on_busy)
        intelligence = self._pages["AI Dataset Analyst"]
        assert isinstance(intelligence, IntelligencePage)
        intelligence.busy.connect(self._on_busy)
        knowledge = self._pages["Engineering Memory"]
        assert isinstance(knowledge, KnowledgePage)
        knowledge.busy.connect(self._on_busy)
        operations = self._pages["Benchmark Center"]
        assert isinstance(operations, OperationsPage)
        operations.busy.connect(self._on_busy)

    # --- slots ---
    def navigate(self, page_name: str) -> None:
        page = self._pages.get(page_name)
        if page is None:
            return
        self._workspace.setCurrentWidget(page)
        page.on_show()
        title, rows = page.context()
        self._context.set_context(title, rows)
        self._settings.last_page = page_name

    def toggle_theme(self) -> str:
        name = self._theme.toggle()
        self._settings.theme = name
        self._status.set_theme(name)
        self._notify.info(f"Theme switched to {name}.")
        return name

    def _quick_import(self) -> None:
        self._open_import_wizard("")

    # --- project workspace flow ---
    _VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".gif", ".tif", ".tiff"}

    def _open_import_wizard(self, preset: str = "") -> None:
        from pathlib import Path

        from vds.gui.widgets.import_wizard import ImportWizard

        if preset and Path(preset).suffix.lower() in self._VIDEO_EXTS:
            self._open_video_import(preset, Path(preset).stem or "video-dataset")
            return
        wizard = ImportWizard(self._controller, self._settings, preset, self)
        wizard.request_folder_import.connect(self._run_folder_import)
        wizard.request_video_import.connect(self._open_video_import)
        wizard.exec()

    def _run_folder_import(self, folder: str, name: str, export_format: str) -> None:
        self._on_busy(True)
        self._notify.info(f"Importing '{name}' — running the pipeline in the background…")
        self._threads.submit(
            self._controller.import_dataset, folder, name,
            export_format=export_format, wants_progress=True,
            on_progress=lambda pct, msg: (self._context.set_progress(pct, msg),
                                          self._notify.info(msg)),
            on_finished=self._on_import_done,
            on_error=lambda m: (self._on_busy(False), self._notify.error(f"Import failed — {m}")),
        )

    def _open_video_import(self, path: str, name: str) -> None:
        from vds.gui.video_import_dialog import VideoImportDialog

        dialog = VideoImportDialog(self._controller, self._threads, self._notify,
                                   path, name, parent=self)
        dialog.imported.connect(self._on_import_done)
        dialog.exec()

    def _on_import_done(self, report) -> None:
        self._on_busy(False)
        self._notify.success(
            f"Import complete: {report.imported} images, {report.detections} annotations.")
        self._pages["Projects"].on_show()
        self._open_project(report.project_id)

    def _open_project(self, project_id: str) -> None:
        project = self._pages["Project"]
        assert isinstance(project, ProjectDashboardPage)
        project.set_project(project_id)
        self.navigate("Project")

    def _route_from_project(self, target: str, project_id: str) -> None:
        if target == "Export":
            export = self._pages["Export"]
            assert isinstance(export, ExportPage)
            export.select_project(project_id)
        elif target == "Annotation":
            annotation = self._pages["Annotation"]
            assert isinstance(annotation, AnnotationPage)
            annotation.set_project(project_id)
        page = self._nav._by_page.get(target)
        if page is not None:  # a nav leaf — select it so the sidebar reflects the move
            self._nav.select(target)
        else:
            self.navigate(target)

    def _on_project_changed(self) -> None:
        self._pages["Projects"].on_show()
        self._nav.select("Projects")

    def _on_notify(self, level: str, message: str) -> None:
        self._status.set_message(level, message)
        self._bottom.log(level, message)

    def _on_busy(self, busy: bool) -> None:
        self._context.show_progress(busy)
        self._status.set_tasks(self._threads.active)

    def _on_resources(self, cpu: float, ram_pct: float, ram_mb: float) -> None:
        self._bottom.update_resources(cpu, ram_pct, ram_mb)
        self._status.set_resources(cpu, ram_pct)
        self._status.set_tasks(self._threads.active)

    # --- window restoration + settings persistence ---
    def _restore_window(self) -> None:
        geo = self._settings.window_geometry()
        state = self._settings.window_state()
        if geo is not None:
            self.restoreGeometry(geo)
        if state is not None:
            self.restoreState(state)
        self._status.set_theme(self._theme.name)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # A worker running on the shared QThreadPool is a live C++ thread; tearing
        # Qt down now segfaults (0xC0000005). Pool workers can't be safely killed
        # mid-run (they may be writing the DB), so veto the close and let it finish.
        if self._threads.active > 0:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self, "Task running",
                "A background task is still running. Wait for it to finish before quitting.",
            )
            event.ignore()
            return
        self._monitor.stop()
        self._threads.wait(3000)  # drain any just-finished worker's C++ side before teardown
        self._settings.save_window(self.saveGeometry(), self.saveState())
        self._settings.theme = self._theme.name
        self._settings.sync()
        super().closeEvent(event)
