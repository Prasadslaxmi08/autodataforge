"""Export page (Phase 18).

Pick a project and a format, then re-export via the existing ExportService
(``controller.export_project`` → ``container.exporter.run``). The export runs on the
shell's ThreadManager so the UI never blocks. No pipeline run, no backend change.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QVBoxLayout

from vds.gui.controller import BackendController
from vds.gui.notifications import NotificationSystem
from vds.gui.pages.base import Page
from vds.gui.threads import ThreadManager
from vds.gui.widgets.common import Card, label


class ExportPage(Page):
    name = "Export"

    busy = Signal(bool)

    def __init__(
        self,
        controller: BackendController,
        threads: ThreadManager,
        notifications: NotificationSystem,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._threads = threads
        self._notify = notifications
        self._preselect: str | None = None
        self._last_dest: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)
        root.addWidget(label("Export Dataset", "H1"))
        root.addWidget(label("Write a validated COCO or YOLO dataset from a finished "
                             "project.", "Muted"))

        card = Card("Export")
        self._project = QComboBox()
        self._format = QComboBox()
        self._format.addItems(self._controller.export_options())
        card.add(label("Project", "Muted"))
        card.add(self._project)
        card.add(label("Format", "Muted"))
        card.add(self._format)
        row = QHBoxLayout()
        self._export_btn = QPushButton("Export")
        self._export_btn.setObjectName("Primary")
        self._export_btn.clicked.connect(self._export)
        self._open_btn = QPushButton("Open Output Folder")
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._open_output)
        row.addWidget(self._export_btn)
        row.addWidget(self._open_btn)
        row.addStretch(1)
        card.body.addLayout(row)
        root.addWidget(card)
        root.addStretch(1)

    def select_project(self, project_id: str) -> None:
        self._preselect = project_id

    def on_show(self) -> None:
        self._project.clear()
        for d in self._controller.list_datasets():
            self._project.addItem(f"{d.name}  ({d.image_count} imgs)", d.project_id)
        if self._preselect is not None:
            idx = self._project.findData(self._preselect)
            if idx >= 0:
                self._project.setCurrentIndex(idx)
            self._preselect = None

    def _export(self) -> None:
        pid = self._project.currentData()
        if pid is None:
            self._notify.warning("No project to export.")
            return
        fmt = self._format.currentText()
        dest = str(Path("export") / pid)
        self._last_dest = dest
        self._export_btn.setEnabled(False)
        self.busy.emit(True)
        self._notify.info(f"Exporting as {fmt}…")
        self._threads.submit(
            self._controller.export_project, pid, fmt, dest,
            on_finished=self._on_done,
            on_error=self._on_error,
        )

    def _on_done(self, report) -> None:
        self._export_btn.setEnabled(True)
        self._open_btn.setEnabled(True)
        self.busy.emit(False)
        ok = "validated" if report.validated else "NOT validated"
        self._notify.success(
            f"Exported {report.annotations} annotations across {report.images} images "
            f"({report.format}, {ok}).")

    def _on_error(self, message: str) -> None:
        self._export_btn.setEnabled(True)
        self.busy.emit(False)
        self._notify.error(f"Export failed — {message}")

    def _open_output(self) -> None:
        if self._last_dest:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._last_dest))

    def context(self) -> tuple[str, list[tuple[str, str]]]:
        return ("Export", [("Projects", str(self._project.count()))])
