"""Project Workspace — the landing page (Phase 18).

The first thing a user sees: a clean home with the product name, the three primary
actions (Create / Import / Continue), and a grid of recent-project cards. It reads
straight from the BackendController; it holds no backend logic. Navigation to the
wizard or a project dashboard happens by signal — the shell owns routing.

Drag & drop a folder, a .zip, or a video onto the page to start an import preloaded
with that path.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from vds.gui.controller import BackendController
from vds.gui.pages.base import Page
from vds.gui.widgets.common import Card, label


class _ActionCard(Card):
    """A big clickable action tile."""

    clicked = Signal()

    def __init__(self, icon: str, title: str, desc: str) -> None:
        super().__init__()
        self.setObjectName("ActionCard")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add(label(icon, "CardIcon"))
        self.add(label(title, "H2"))
        self.add(label(desc, "Muted", wrap=True))
        self.body.addStretch(1)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class _ProjectCard(Card):
    """A recent-project tile: thumbnail + name + status + size."""

    clicked = Signal(str)

    def __init__(self, summary) -> None:
        super().__init__()
        self.setObjectName("ActionCard")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pid = summary.project_id
        thumb = label("", "")
        if summary.thumbnails:
            pix = QPixmap(summary.thumbnails[0])
            if not pix.isNull():
                thumb.setPixmap(pix.scaledToWidth(180, Qt.TransformationMode.SmoothTransformation))
        self.add(thumb)
        self.add(label(summary.name, "H2"))
        self.add(label(f"{summary.image_count} images · {summary.annotation_count} objects", "Muted"))
        self.add(label(summary.phase, "Badge"))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._pid)
        super().mouseReleaseEvent(event)


class ProjectWorkspacePage(Page):
    name = "Projects"

    #: open the import wizard, optionally preloaded with a dropped path
    start_import = Signal(str)
    #: open a project's dashboard
    open_project = Signal(str)

    def __init__(self, controller: BackendController) -> None:
        super().__init__()
        self._controller = controller
        self.setAcceptDrops(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll)
        content = QWidget()
        self._root = QVBoxLayout(content)
        self._root.setContentsMargins(28, 24, 28, 24)
        self._root.setSpacing(16)
        scroll.setWidget(content)

        self._root.addWidget(label("AutoDataForge", "Hero"))
        self._root.addWidget(label("Create AI-labelled image datasets — import, and the "
                                   "pipeline does the rest.", "Muted"))

        self._root.addLayout(self._actions())
        self._root.addWidget(label("Recent Projects", "H1"))
        self._recent_holder = QWidget()
        self._recent_grid = QGridLayout(self._recent_holder)
        self._recent_grid.setSpacing(12)
        self._recent_grid.setContentsMargins(0, 0, 0, 0)
        self._root.addWidget(self._recent_holder)
        self._root.addStretch(1)

    def _actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        create = _ActionCard("➕", "Create Project", "Start a new dataset generation project.")
        create.clicked.connect(lambda: self.start_import.emit(""))
        imp = _ActionCard("📥", "Import Dataset", "Images, folder, ZIP, or video — drag & drop too.")
        imp.clicked.connect(lambda: self.start_import.emit(""))
        self._continue = _ActionCard("▶", "Continue Project", "Resume your most recent project.")
        self._continue.clicked.connect(self._continue_recent)
        for c in (create, imp, self._continue):
            row.addWidget(c)
        return row

    # --- data ---
    def on_show(self) -> None:
        while self._recent_grid.count():
            item = self._recent_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        summaries = self._controller.list_datasets(thumbnails=1)
        self._recent = summaries
        if not summaries:
            self._recent_grid.addWidget(
                label("No projects yet — Create or Import to begin.", "Muted"), 0, 0)
            self._continue.setEnabled(False)
            return
        self._continue.setEnabled(True)
        for i, s in enumerate(reversed(summaries[-9:])):  # newest first, cap the grid
            card = _ProjectCard(s)
            card.clicked.connect(self.open_project)
            self._recent_grid.addWidget(card, i // 3, i % 3)

    def _continue_recent(self) -> None:
        if getattr(self, "_recent", None):
            self.open_project.emit(self._recent[-1].project_id)

    # --- drag & drop ---
    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt override)
        urls = event.mimeData().urls()
        if urls:
            self.start_import.emit(urls[0].toLocalFile())
            event.acceptProposedAction()

    def context(self) -> tuple[str, list[tuple[str, str]]]:
        return ("Projects", [("Recent", str(len(getattr(self, "_recent", []))))])
