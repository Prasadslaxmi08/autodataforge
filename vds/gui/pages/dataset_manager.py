"""Dataset Manager (Phase 11) — the fully functional data page.

Import (runs the real backend pipeline on a folder, off the UI thread), preview
thumbnails, per-dataset statistics, rename, delete, open, and a recent list. All
backend access goes through the BackendController; the long import runs on a
ThreadManager worker so the UI never blocks and progress is shown throughout.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vds.gui.controller import BackendController, DatasetSummary
from vds.gui.notifications import NotificationSystem
from vds.gui.pages.base import Page
from vds.gui.threads import ThreadManager
from vds.gui.widgets.common import Card, label

_HEADERS = ["Name", "Images", "Annotations", "Approved", "Review", "Rejected", "Phase"]


class DatasetManagerPage(Page):
    name = "Dataset Manager"

    #: forwarded to the shell so the context sidebar can show import progress
    progressed = Signal(int, str)
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
        self._selected: str | None = None
        self._importing = False

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)
        root.addWidget(label("Dataset Manager — Import Workspace", "H1"))
        root.addWidget(label("Import from an image, folder, ZIP, or video — every source "
                             "becomes a standard image dataset for the same AI pipeline.", "Muted"))

        root.addLayout(self._toolbar())

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._table_card(), 3)
        body.addWidget(self._detail_card(), 2)
        root.addLayout(body, 1)

    # --- construction helpers ---
    def _toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self._btn_import = QPushButton("Import Folder")
        self._btn_import.setObjectName("Primary")
        self._btn_import.clicked.connect(self._import)
        self._btn_video = QPushButton("Import Video…")
        self._btn_video.clicked.connect(self._import_video)
        self._btn_open = QPushButton("Open")
        self._btn_open.clicked.connect(self._open)
        self._btn_rename = QPushButton("Rename")
        self._btn_rename.clicked.connect(self._rename)
        self._btn_delete = QPushButton("Delete")
        self._btn_delete.clicked.connect(self._delete)
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.clicked.connect(self.on_show)
        for b in (self._btn_import, self._btn_video, self._btn_open, self._btn_rename,
                  self._btn_delete, self._btn_refresh):
            bar.addWidget(b)
        bar.addStretch(1)
        return bar

    def _table_card(self) -> Card:
        card = Card("Datasets")
        self._table = QTableWidget(0, len(_HEADERS))
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        self._table.doubleClicked.connect(self._open)
        card.add(self._table)
        return card

    def _detail_card(self) -> Card:
        card = Card("Preview & Statistics")
        self._stats = QWidget()
        self._stats_layout = QVBoxLayout(self._stats)
        self._stats_layout.setContentsMargins(0, 0, 0, 0)
        card.add(self._stats)
        self._thumbs = QListWidget()
        self._thumbs.setViewMode(QListWidget.ViewMode.IconMode)
        self._thumbs.setIconSize(QSize(72, 72))
        self._thumbs.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._thumbs.setMovement(QListWidget.Movement.Static)
        card.add(self._thumbs)
        self._render_stats(None)
        return card

    # --- data flow ---
    def on_show(self) -> None:
        rows = self._controller.list_datasets()
        self._table.setRowCount(len(rows))
        for r, d in enumerate(rows):
            values = [d.name, d.image_count, d.annotation_count, d.approved,
                      d.needs_review, d.rejected, d.phase]
            for col, val in enumerate(values):
                item = QTableWidgetItem(str(val))
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, d.project_id)
                self._table.setItem(r, col, item)
        # keep selection if still present
        if self._selected and not any(d.project_id == self._selected for d in rows):
            self._selected = None
            self._render_stats(None)

    def _current_project_id(self) -> str | None:
        item = self._table.item(self._table.currentRow(), 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _on_row_selected(self) -> None:
        self._selected = self._current_project_id()

    def _open(self) -> None:
        pid = self._current_project_id()
        if pid is None:
            self._notify.warning("Select a dataset to open.")
            return
        self._selected = pid
        detail = self._controller.dataset_detail(pid)
        self._render_stats(detail)
        self._notify.info(f"Opened dataset '{detail.name}'." if detail else "Dataset not found.")

    def _render_stats(self, detail: DatasetSummary | None) -> None:
        while self._stats_layout.count():
            item = self._stats_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._thumbs.clear()
        if detail is None:
            self._stats_layout.addWidget(label("No dataset selected.", "Muted"))
            return
        rows = [
            ("Name", detail.name), ("Phase", detail.phase),
            ("Images", str(detail.image_count)), ("Annotations", str(detail.annotation_count)),
            ("Approved", str(detail.approved)), ("Needs review", str(detail.needs_review)),
            ("Rejected", str(detail.rejected)),
        ]
        for k, v in rows:
            line = QWidget()
            h = QHBoxLayout(line)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(label(k, "Muted"))
            h.addStretch(1)
            h.addWidget(label(v))
            self._stats_layout.addWidget(line)
        for path in detail.thumbnails:
            pix = QPixmap(path)
            if not pix.isNull():
                self._thumbs.addItem(QListWidgetItem(QIcon(pix), ""))

    # --- actions ---
    def _import(self) -> None:
        if self._importing:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select image folder to import")
        if not folder:
            return
        name, ok = QInputDialog.getText(self, "Dataset name", "Name:", text="dataset")
        if not ok or not name.strip():
            return
        self._set_importing(True)
        self._notify.info(f"Importing '{name}' — running the pipeline in the background…")
        self._threads.submit(
            self._controller.import_dataset, folder, name.strip(),
            wants_progress=True,
            on_progress=self._on_progress,
            on_finished=self._on_import_done,
            on_error=self._on_import_error,
        )

    def _import_video(self) -> None:
        if self._importing:
            return
        exts = "Videos (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.gif *.webp *.apng *.tif *.tiff)"
        path, _ = QFileDialog.getOpenFileName(self, "Select a video to import", "", exts)
        if not path:
            return
        name, ok = QInputDialog.getText(self, "Dataset name", "Name:", text="video-dataset")
        if not ok or not name.strip():
            return
        from vds.gui.video_import_dialog import VideoImportDialog

        dialog = VideoImportDialog(self._controller, self._threads, self._notify,
                                   path, name.strip(), parent=self)
        dialog.imported.connect(lambda _report: self.on_show())
        dialog.exec()

    def _on_progress(self, pct: int, msg: str) -> None:
        self._progress.setValue(pct)
        self.progressed.emit(pct, msg)
        self._notify.info(msg)

    def _on_import_done(self, report) -> None:
        self._set_importing(False)
        self._notify.success(
            f"Import complete: {report.imported} images, {report.detections} annotations, "
            f"{report.verified_approved} approved."
        )
        self.on_show()

    def _on_import_error(self, message: str) -> None:
        self._set_importing(False)
        self._notify.error(f"Import failed — {message}")

    def _set_importing(self, value: bool) -> None:
        self._importing = value
        self._btn_import.setEnabled(not value)
        self._btn_import.setText("Importing…" if value else "Import Dataset")
        self._progress.setVisible(value)
        if not value:
            self._progress.setValue(0)
        self.busy.emit(value)

    def _rename(self) -> None:
        pid = self._current_project_id()
        if pid is None:
            self._notify.warning("Select a dataset to rename.")
            return
        new_name, ok = QInputDialog.getText(self, "Rename dataset", "New name:")
        if not ok or not new_name.strip():
            return
        self._controller.rename_dataset(pid, new_name.strip())
        self._notify.success(f"Renamed to '{new_name.strip()}'.")
        self.on_show()

    def _delete(self) -> None:
        pid = self._current_project_id()
        if pid is None:
            self._notify.warning("Select a dataset to delete.")
            return
        confirm = QMessageBox.question(
            self, "Delete dataset",
            "Delete this dataset and its annotations? This cannot be undone.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._controller.delete_dataset(pid)
        self._selected = None
        self._render_stats(None)
        self._notify.success("Dataset deleted.")
        self.on_show()

    def context(self) -> tuple[str, list[tuple[str, str]]]:
        return ("Dataset Manager", [
            ("Datasets", str(self._table.rowCount())),
            ("Selected", self._selected or "none"),
            ("Import", "running" if self._importing else "idle"),
        ])
