"""Video Import dialog (Phase 17.5) — the video branch of the Import Workspace.

Shows video metadata, lets the engineer choose an extraction strategy, runs the
existing Planner for pre-extraction recommendations, previews the estimate, then
extracts frames and hands the resulting image folder to the EXISTING pipeline. All
heavy work runs on the ThreadManager; extraction is cancellable. Every number shown
comes from the video metadata, the real Planner, or measured extraction stats.
"""

from __future__ import annotations

import tempfile

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vds.gui.controller import BackendController
from vds.gui.notifications import NotificationSystem
from vds.gui.threads import ThreadManager
from vds.gui.widgets.common import Card, label
from vds.video import ExtractionConfig

# friendly label -> strategy key
_STRATEGIES = {
    "Every Frame": "every_frame",
    "Every N Frames": "every_n",
    "Every X Seconds": "every_seconds",
    "Fixed Number of Frames": "fixed_count",
    "Scene Change (experimental)": "scene_change",
}


class VideoImportDialog(QDialog):
    #: emitted with the ExecutionReport when the video was imported successfully
    imported = Signal(object)

    def __init__(
        self,
        controller: BackendController,
        threads: ThreadManager,
        notifications: NotificationSystem,
        video_path: str,
        name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._threads = threads
        self._notify = notifications
        self._video_path = video_path
        self._name = name
        self._info = None
        self._cancel = False
        self._busy = False

        self.setWindowTitle(f"Import Video — {name}")
        self.resize(720, 640)
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)
        root.addWidget(label("Video Dataset Import", "H1"))
        root.addWidget(label("Every source — image, folder, ZIP, or video — becomes a "
                             "standard image dataset that enters the same AI pipeline.", "Muted"))

        body = QHBoxLayout()
        body.addWidget(self._info_card(), 3)
        body.addWidget(self._thumb_card(), 2)
        root.addLayout(body)
        root.addWidget(self._settings_card())
        root.addWidget(self._plan_card())

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        root.addWidget(self._progress)
        root.addLayout(self._buttons())

        self._probe()

    # --- construction ---
    def _info_card(self) -> Card:
        card = Card("Video Information")
        self._info_grid = QGridLayout()
        self._info_grid.setColumnStretch(1, 1)
        card.body.addLayout(self._info_grid)
        return card

    def _thumb_card(self) -> Card:
        card = Card("Thumbnail Preview")
        self._thumb = QLabel("—")
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setMinimumHeight(160)
        card.add(self._thumb)
        return card

    def _settings_card(self) -> Card:
        card = Card("Frame Extraction")
        row = QHBoxLayout()
        row.addWidget(label("Strategy", "Muted"))
        self._strategy = QComboBox()
        self._strategy.addItems(list(_STRATEGIES))
        self._strategy.setCurrentText("Every N Frames")
        self._strategy.currentTextChanged.connect(self._on_strategy)
        row.addWidget(self._strategy)

        self._every_n = QSpinBox()
        self._every_n.setRange(1, 100000)
        self._every_n.setValue(30)
        self._every_n.setPrefix("N = ")
        self._seconds = QDoubleSpinBox()
        self._seconds.setRange(0.05, 3600.0)
        self._seconds.setValue(1.0)
        self._seconds.setSuffix(" s")
        self._count = QSpinBox()
        self._count.setRange(1, 1000000)
        self._count.setValue(100)
        self._count.setPrefix("count = ")
        self._scene = QSpinBox()
        self._scene.setRange(1, 64)
        self._scene.setValue(12)
        self._scene.setPrefix("Δ = ")
        for w in (self._every_n, self._seconds, self._count, self._scene):
            w.valueChanged.connect(lambda _v: self._update_estimate())
            row.addWidget(w)
        self._dedup = QCheckBox("Remove duplicate frames")
        self._dedup.setChecked(True)
        row.addWidget(self._dedup)
        row.addStretch(1)
        card.body.addLayout(row)
        self._estimate = label("—", "Muted")
        card.add(self._estimate)
        self._on_strategy(self._strategy.currentText())
        return card

    def _plan_card(self) -> Card:
        card = Card("Planner Pre-Analysis & Preview")
        analyze = QPushButton("Analyze with Planner")
        analyze.clicked.connect(self._analyze)
        card.add(analyze)
        self._plan_grid = QGridLayout()
        self._plan_grid.setColumnStretch(1, 1)
        card.body.addLayout(self._plan_grid)
        self._stats = label("", "Badge")
        card.add(self._stats)
        return card

    def _buttons(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addStretch(1)
        self._cancel_btn = QPushButton("Cancel Extraction")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._request_cancel)
        bar.addWidget(self._cancel_btn)
        self._start_btn = QPushButton("Start Extraction & Import")
        self._start_btn.setObjectName("Primary")
        self._start_btn.clicked.connect(self._start)
        bar.addWidget(self._start_btn)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        bar.addWidget(close)
        return bar

    # --- config ---
    def _config(self) -> ExtractionConfig:
        return ExtractionConfig(
            strategy=_STRATEGIES[self._strategy.currentText()],
            every_n=self._every_n.value(), seconds=self._seconds.value(),
            count=self._count.value(), scene_threshold=self._scene.value())

    def _on_strategy(self, text: str) -> None:
        key = _STRATEGIES[text]
        self._every_n.setVisible(key == "every_n")
        self._seconds.setVisible(key == "every_seconds")
        self._count.setVisible(key == "fixed_count")
        self._scene.setVisible(key == "scene_change")
        self._update_estimate()

    # --- probe ---
    def _probe(self) -> None:
        self._set_busy(True)
        self._threads.submit(self._controller.probe_video, self._video_path,
                             on_finished=self._on_probed, on_error=self._on_error)

    def _on_probed(self, info) -> None:
        self._set_busy(False)
        self._info = info
        rows = [
            ("Video Name", info.name),
            ("Duration", f"{info.duration_s} s" if info.duration_s else "Unavailable"),
            ("Resolution", f"{info.width}×{info.height}" if info.width else "Unavailable"),
            ("FPS", str(info.fps) if info.fps else "Unavailable"),
            ("Codec", info.codec),
            ("Total Frames", str(info.total_frames)),
            ("File Size", f"{info.size_bytes / (1024 * 1024):.2f} MB"),
        ]
        self._fill_grid(self._info_grid, rows)
        self._update_estimate()
        self._load_thumbnail()

    def _load_thumbnail(self) -> None:
        try:
            path = self._controller.video_thumbnail(self._video_path, tempfile.mkdtemp())
        except Exception:
            path = None
        if path:
            pix = QPixmap(path)
            if not pix.isNull():
                self._thumb.setPixmap(pix.scaledToWidth(220, Qt.TransformationMode.SmoothTransformation))

    def _update_estimate(self) -> None:
        if self._info is None:
            return
        try:
            count = self._controller.video_frame_estimate(self._info, self._config())
        except Exception as exc:
            self._estimate.setText(f"Estimate unavailable — {exc}")
            return
        from vds.video.engine import estimate_disk_mb

        disk = estimate_disk_mb(self._info, count)
        self._estimate.setText(f"Estimated extracted images: {count}  ·  Estimated storage: ~{disk} MB "
                               f"(before duplicate removal)")

    # --- planner pre-analysis ---
    def _analyze(self) -> None:
        if self._info is None or self._busy:
            return
        self._set_busy(True)
        self._notify.info("Running the Planner over the estimated dataset…")
        self._threads.submit(self._controller.plan_video, self._info, self._config(),
                             on_finished=self._on_planned, on_error=self._on_error)

    def _on_planned(self, plan) -> None:
        self._set_busy(False)
        rows = [
            ("Estimated Dataset Size", f"{plan.estimated_dataset_size} images"),
            ("Expected Processing Time", plan.expected_processing_time),
            ("Expected Review Rate", plan.expected_review_rate),
            ("Expected Duplicate %", plan.expected_duplicate_pct),
            ("Recommended Detector", plan.recommended_detector),
            ("Recommended Batch Size", str(plan.recommended_batch_size)),
            ("Recommended Tiling", plan.recommended_tiling),
            ("Recommended Segmentation", plan.recommended_segmentation),
            ("Estimated Runtime", plan.estimated_runtime),
            ("Expected Export Size", plan.expected_export_size),
            (f"Planner ({plan.source})", plan.note),
        ]
        self._fill_grid(self._plan_grid, rows)
        self._notify.success("Planner recommendations ready — you may override the settings.")

    # --- extraction + import ---
    def _start(self) -> None:
        if self._info is None or self._busy:
            return
        self._cancel = False
        self._set_busy(True)
        self._cancel_btn.setVisible(True)
        self._progress.setVisible(True)
        self._notify.info("Extracting frames and running the pipeline…")
        self._threads.submit(
            self._controller.import_video_dataset, self._video_path, self._name, self._config(),
            dedup=self._dedup.isChecked(), cancel=lambda: self._cancel,
            wants_progress=True, on_progress=self._on_progress,
            on_finished=self._on_imported, on_error=self._on_error)

    def _request_cancel(self) -> None:
        self._cancel = True
        self._notify.warning("Cancelling extraction…")

    def _on_progress(self, pct: int, msg: str) -> None:
        self._progress.setValue(pct)

    def _on_imported(self, result) -> None:
        self._set_busy(False)
        self._cancel_btn.setVisible(False)
        report, stats, _info = result
        self._stats.setText(
            f"Frames Extracted: {stats.frames_extracted}  ·  Removed: {stats.frames_removed}  ·  "
            f"Unique: {stats.unique_frames}  ·  Duplicate %: {stats.duplicate_percentage}")
        if stats.cancelled or report is None:
            self._notify.warning("Extraction cancelled — no dataset was created.")
            return
        self._notify.success(f"Imported {report.imported} images from video into '{self._name}'.")
        self.imported.emit(report)
        self.accept()

    def _on_error(self, message: str) -> None:
        self._set_busy(False)
        self._cancel_btn.setVisible(False)
        self._progress.setVisible(False)
        self._notify.error(f"Video import failed — {message}")

    # --- helpers ---
    def _set_busy(self, value: bool) -> None:
        self._busy = value
        self._start_btn.setEnabled(not value)

    def _fill_grid(self, grid: QGridLayout, rows: list[tuple[str, str]]) -> None:
        while grid.count():
            item = grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, (k, v) in enumerate(rows):
            grid.addWidget(label(k, "Muted"), i, 0, Qt.AlignmentFlag.AlignTop)
            val = QLabel(v)
            val.setWordWrap(True)
            grid.addWidget(val, i, 1)
