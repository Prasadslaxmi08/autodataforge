"""Project Dashboard — a single project's home (Phase 18).

Opened from the Project Workspace. Shows the project's stats (read via the existing
``dataset_detail``) and the four next-step actions. It never runs work itself — the
action buttons emit ``request_nav(page_name, project_id)`` and the shell routes to
the Annotation Pipeline / VLM Verification / Export pages. Rename and delete reuse
the existing controller methods.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vds.gui.controller import BackendController
from vds.gui.notifications import NotificationSystem
from vds.gui.pages.base import Page
from vds.gui.widgets.common import Card, MetricTile, label


class ProjectDashboardPage(Page):
    name = "Project"

    #: (target page name, project_id) — the shell navigates there
    request_nav = Signal(str, str)
    #: ask the shell to refresh the workspace / return home after a delete
    changed = Signal()

    def __init__(self, controller: BackendController, notifications: NotificationSystem) -> None:
        super().__init__()
        self._controller = controller
        self._notify = notifications
        self._pid: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)
        self._title = label("Project", "Hero")
        root.addWidget(self._title)
        self._status = label("", "Badge")
        root.addWidget(self._status)

        self._tiles = QHBoxLayout()
        self._tiles.setSpacing(12)
        holder = QWidget()
        holder.setLayout(self._tiles)
        root.addWidget(holder)

        root.addWidget(self._actions_card())
        root.addStretch(1)

    def _actions_card(self) -> Card:
        card = Card("Next steps")
        row = QHBoxLayout()
        row.setSpacing(10)
        specs = [
            ("Start Annotation", "Annotation", "Primary"),
            ("Review Dataset", "VLM Verification", ""),
            ("Export Dataset", "Export", ""),
        ]
        for text, target, obj in specs:
            b = QPushButton(text)
            if obj:
                b.setObjectName(obj)
            b.clicked.connect(lambda _c=False, t=target: self._go(t))
            row.addWidget(b)
        row.addStretch(1)
        self._rename_btn = QPushButton("Rename")
        self._rename_btn.clicked.connect(self._rename)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._delete)
        row.addWidget(self._rename_btn)
        row.addWidget(self._delete_btn)
        card.body.addLayout(row)
        return card

    # --- data ---
    def set_project(self, project_id: str) -> None:
        self._pid = project_id
        self.on_show()

    def on_show(self) -> None:
        while self._tiles.count():
            item = self._tiles.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if self._pid is None:
            return
        detail = self._controller.dataset_detail(self._pid)
        if detail is None:
            self._title.setText("Project not found")
            self._status.setText("")
            return
        self._title.setText(detail.name)
        self._status.setText(f"Status: {detail.phase}")
        tiles = [
            ("Images", detail.image_count), ("Objects", detail.annotation_count),
            ("Approved", detail.approved), ("Needs review", detail.needs_review),
            ("Rejected", detail.rejected),
        ]
        for caption, value in tiles:
            self._tiles.addWidget(MetricTile(caption, str(value)))

    def _go(self, target: str) -> None:
        if self._pid:
            self.request_nav.emit(target, self._pid)

    def _rename(self) -> None:
        if self._pid is None:
            return
        new_name, ok = QInputDialog.getText(self, "Rename project", "New name:")
        if not ok or not new_name.strip():
            return
        self._controller.rename_dataset(self._pid, new_name.strip())
        self._notify.success(f"Renamed to '{new_name.strip()}'.")
        self.on_show()
        self.changed.emit()

    def _delete(self) -> None:
        if self._pid is None:
            return
        confirm = QMessageBox.question(
            self, "Delete project",
            "Delete this project and its annotations? This cannot be undone.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._controller.delete_dataset(self._pid)
        self._notify.success("Project deleted.")
        self._pid = None
        self.changed.emit()

    def context(self) -> tuple[str, list[tuple[str, str]]]:
        return ("Project", [("Project", self._pid or "none")])
