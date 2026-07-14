"""Annotation Pipeline Workspace (Phase 13) — real-time execution monitoring.

Five sections: execution timeline, live image preview, AI-model activity, live
metrics, and a processing console — plus execution controls and a post-run summary.
The existing `Phase1Pipeline.run()` is monolithic and exposes no live hooks, so it
runs unchanged on a worker thread while genuinely-live signals (elapsed time, CPU/
RAM, console milestones) update on a UI timer; exact per-stage detail is rendered
from the real ExecutionReport when the pipeline reports it. Everything goes through
BackendController; no backend logic lives here.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

from vds.gui.controller import BackendController
from vds.gui.notifications import NotificationSystem
from vds.gui.pages.base import Page
from vds.gui.threads import ThreadManager
from vds.gui.widgets.common import Card, label
from vds.gui.widgets.image_preview import ImagePreview
from vds.gui.widgets.sparkline import Sparkline

_STATUS_COLOR = {
    "Waiting": "#8b9096", "Running": "#3d7eff", "Completed": "#4caf82",
    "Failed": "#e0605e", "Skipped": "#e0a458",
}
_LEVELS = ["All", "info", "warning", "error"]


class PipelinePage(Page):
    name = "Annotation Pipeline"

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
        self._state = "idle"
        self._report = None
        self._source: str | None = None
        self._name = ""
        self._cancelled = False
        self._start_ts = 0.0
        self._elapsed = 0.0
        self._preview_items: list = []
        self._preview_idx = 0
        self._log: list[tuple[str, str, str]] = []  # (ts, level, message)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll)
        content = QWidget()
        self._root = QVBoxLayout(content)
        self._root.setContentsMargins(20, 18, 20, 18)
        self._root.setSpacing(12)
        scroll.setWidget(content)

        self._root.addWidget(label("Annotation Pipeline Workspace", "H1"))
        self._status = label("Idle — start a run to monitor execution.", "Muted")
        self._root.addWidget(self._status)
        self._root.addLayout(self._controls())

        grid = QGridLayout()
        grid.setSpacing(12)
        grid.addWidget(self._timeline_card(), 0, 0)
        grid.addWidget(self._metrics_card(), 0, 1)
        holder = QWidget()
        holder.setLayout(grid)
        self._root.addWidget(holder)

        self._root.addWidget(self._preview_card())
        self._root.addWidget(self._activity_card())
        self._root.addWidget(self._console_card())
        self._root.addWidget(self._summary_card())
        self._root.addStretch(1)
        self._refresh_controls()

    # --- controls ---
    def _controls(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self._buttons = {}
        specs = [
            ("start", "Start", True, self._start), ("pause", "Pause", False, self._pause),
            ("resume", "Resume", False, self._resume), ("cancel", "Cancel", False, self._cancel),
            ("restart", "Restart", False, self._restart),
            ("output", "Open Output Folder", False, self._output),
            ("export", "Export Dataset", False, self._export),
            ("report", "Generate Report", False, self._generate_report),
        ]
        for key, text, primary, handler in specs:
            b = QPushButton(text)
            if primary:
                b.setObjectName("Primary")
            b.clicked.connect(handler)
            self._buttons[key] = b
            bar.addWidget(b)
        bar.addStretch(1)
        return bar

    # --- section 1: timeline ---
    def _timeline_card(self) -> Card:
        card = Card("Pipeline Execution Timeline")
        self._timeline = QTableWidget(0, 5)
        self._timeline.setHorizontalHeaderLabels(["Stage", "Status", "Duration", "Items", "%"])
        self._timeline.verticalHeader().setVisible(False)
        self._timeline.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._timeline.horizontalHeader().setStretchLastSection(True)
        card.add(self._timeline)
        return card

    # --- section 4: metrics ---
    def _metrics_card(self) -> Card:
        card = Card("Live Metrics")
        self._metrics_grid = QGridLayout()
        self._metrics_grid.setColumnStretch(1, 1)
        self._metrics_grid.setColumnStretch(3, 1)
        card.body.addLayout(self._metrics_grid)
        self._cpu_spark = Sparkline("CPU %", color="#3d7eff")
        self._ram_spark = Sparkline("RAM %", color="#4caf82")
        card.add(self._cpu_spark)
        card.add(self._ram_spark)
        self._set_metrics({"Status": "idle"})
        return card

    # --- section 2: preview ---
    def _preview_card(self) -> Card:
        card = Card("Live Image Preview — Original vs Annotated")
        self._preview_caption = label("No image loaded.", "Muted")
        card.add(self._preview_caption)
        row = QHBoxLayout()
        self._orig = ImagePreview()
        self._orig.setMinimumHeight(220)
        self._annot = ImagePreview()
        self._annot.setMinimumHeight(220)
        row.addWidget(self._orig)
        row.addWidget(self._annot)
        card.body.addLayout(row)
        nav = QHBoxLayout()
        for text, slot in [("◀ Prev", self._prev_image), ("Next ▶", self._next_image),
                           ("Zoom +", self._zoom_in), ("Zoom −", self._zoom_out),
                           ("Fit", self._fit)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            nav.addWidget(b)
        nav.addStretch(1)
        card.body.addLayout(nav)
        return card

    # --- section 3: model activity ---
    def _activity_card(self) -> Card:
        card = Card("AI Model Activity")
        self._activity = QVBoxLayout()
        card.body.addLayout(self._activity)
        self._set_activity(None)
        return card

    # --- section 5: console ---
    def _console_card(self) -> Card:
        card = Card("Processing Console")
        bar = QHBoxLayout()
        self._severity = QComboBox()
        self._severity.addItems(_LEVELS)
        self._severity.currentTextChanged.connect(lambda _t: self._render_console())
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search…")
        self._search.textChanged.connect(lambda _t: self._render_console())
        self._autoscroll = QCheckBox("Auto-scroll")
        self._autoscroll.setChecked(True)
        copy_b = QPushButton("Copy Log")
        copy_b.clicked.connect(self._copy_log)
        save_b = QPushButton("Save Log")
        save_b.clicked.connect(self._save_log)
        bar.addWidget(label("Severity", "Muted"))
        bar.addWidget(self._severity)
        bar.addWidget(self._search, 1)
        bar.addWidget(self._autoscroll)
        bar.addWidget(copy_b)
        bar.addWidget(save_b)
        card.body.addLayout(bar)
        self._console = QPlainTextEdit()
        self._console.setReadOnly(True)
        self._console.setMaximumBlockCount(5000)
        card.add(self._console)
        return card

    # --- summary ---
    def _summary_card(self) -> Card:
        self._summary = Card("Pipeline Summary")
        self._summary_grid = QGridLayout()
        self._summary_grid.setColumnStretch(1, 1)
        self._summary.body.addLayout(self._summary_grid)
        row = QHBoxLayout()
        md = QPushButton("Export Summary (Markdown)")
        md.clicked.connect(lambda: self._export_summary("md"))
        pdf = QPushButton("Export Summary (PDF)")
        pdf.clicked.connect(lambda: self._export_summary("pdf"))
        row.addStretch(1)
        row.addWidget(md)
        row.addWidget(pdf)
        self._summary.body.addLayout(row)
        self._summary.setVisible(False)
        return self._summary

    # =================== control actions ===================
    def _start(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select image folder to process")
        if not folder:
            return
        name, ok = QInputDialog.getText(self, "Run name", "Name:", text="pipeline-run")
        if not ok or not name.strip():
            return
        self._source, self._name = folder, name.strip()
        self._launch()

    def _restart(self) -> None:
        if not self._source:
            self._notify.warning("Nothing to restart — start a run first.")
            return
        self._launch()

    def _launch(self) -> None:
        self._cancelled = False
        self._report = None
        self._state = "running"
        self._log.clear()
        self._start_ts = time.monotonic()
        self._elapsed = 0.0
        self._cpu_spark.clear()
        self._ram_spark.clear()
        self._summary.setVisible(False)
        self._render_timeline(self._controller.running_timeline())
        self._preview_items = [
            _PreviewStub(p) for p in self._controller.source_images(self._source or "")
        ]
        self._preview_idx = 0
        self._add_log("info", f"Pipeline started for '{self._name}'")
        self._timer.start()
        self.busy.emit(True)
        self._refresh_controls()
        self._threads.submit(
            self._controller.run_pipeline, self._source, self._name,
            wants_progress=True,
            on_progress=lambda pct, msg: self._add_log("info", msg),
            on_finished=self._on_done,
            on_error=self._on_error,
        )

    def _pause(self) -> None:
        if self._state != "running":
            return
        self._timer.stop()
        self._state = "paused"
        # ponytail: the backend pipeline is atomic and cannot be interrupted; Pause
        # freezes the live monitoring view, not backend execution. Made explicit in UX.
        self._add_log("warning", "Monitoring paused (backend run continues to completion)")
        self._set_status()
        self._refresh_controls()

    def _resume(self) -> None:
        if self._state != "paused":
            return
        self._state = "running"
        self._timer.start()
        self._add_log("info", "Monitoring resumed")
        self._set_status()
        self._refresh_controls()

    def _cancel(self) -> None:
        if self._state not in ("running", "paused"):
            return
        self._cancelled = True
        self._timer.stop()
        self._state = "cancelling"
        self._add_log("warning", "Cancel requested — the current run will be rolled back on finish")
        self._set_status()
        self._refresh_controls()

    def _output(self) -> None:
        path = self._controller.export_dir(self._name or "")
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        self._notify.info(f"Output folder: {path}")

    def _export(self) -> None:
        if self._report is None:
            self._notify.warning("Run the pipeline before exporting.")
            return
        self._notify.info(f"Dataset already exported to {self._controller.export_dir(self._name)} "
                          f"({self._report.export.format}).")
        self._output()

    def _generate_report(self) -> None:
        if self._report is None:
            self._notify.warning("Run the pipeline before generating a report.")
            return
        path = self._controller.save_report_file(self._report)
        self._notify.success(f"Report saved via reporting infrastructure: {path}")

    # =================== worker callbacks ===================
    def _on_done(self, report) -> None:
        self._timer.stop()
        self.busy.emit(False)
        if self._cancelled:
            self._controller.delete_dataset(report.project_id)  # rollback via existing backend
            self._state = "cancelled"
            self._add_log("warning", "Run cancelled — dataset rolled back")
            self._set_status()
            self._refresh_controls()
            return
        self._report = report
        self._state = "completed"
        self._render_timeline(self._controller.stage_timeline(report))
        self._set_activity(self._controller.model_activity(report))
        self._update_metrics_final(report)
        for level, msg in self._controller.console_events(report):
            self._add_log(level, msg)
        self._load_previews(report)
        self._render_summary(self._controller.pipeline_summary(report))
        self._set_status()
        self._refresh_controls()
        self._notify.success(
            f"Pipeline completed: {report.imported} images, {report.detections} annotations.")

    def _on_error(self, message: str) -> None:
        self._timer.stop()
        self.busy.emit(False)
        self._state = "failed"
        rows = self._controller.running_timeline()
        rows[0].status = "Failed"
        self._render_timeline(rows)
        self._add_log("error", f"Pipeline failed: {message}")
        self._set_status()
        self._refresh_controls()
        self._notify.error(f"Pipeline failed — {message}")

    # =================== live tick ===================
    def _tick(self) -> None:
        self._elapsed = time.monotonic() - self._start_ts
        cpu = ram = 0.0
        if psutil is not None:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
        self._cpu_spark.push(cpu)
        self._ram_spark.push(ram)
        remaining = len(self._preview_items)
        self._set_metrics({
            "Status": self._state, "Elapsed": f"{self._elapsed:.0f} s",
            "CPU": f"{cpu:.0f}%", "RAM": f"{ram:.0f}%",
            "Images (source)": str(remaining), "Stage": "running",
        })
        if self._preview_items:  # cycle originals to show activity
            self._preview_idx = (self._preview_idx + 1) % len(self._preview_items)
            self._show_preview()

    # =================== rendering ===================
    def _render_timeline(self, stages) -> None:
        self._timeline.setRowCount(len(stages))
        for r, s in enumerate(stages):
            cells = [s.name, s.status,
                     "—" if s.duration_s is None else f"{s.duration_s:.3f}s",
                     str(s.items), f"{s.progress_pct}%"]
            for c, val in enumerate(cells):
                item = QTableWidgetItem(val)
                if c == 1:
                    item.setForeground(QColor(_STATUS_COLOR.get(s.status, "#d7dae0")))
                self._timeline.setItem(r, c, item)
        self._timeline.resizeColumnsToContents()

    def _set_metrics(self, rows: dict) -> None:
        while self._metrics_grid.count():
            item = self._metrics_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        items = list(rows.items())
        for i, (k, v) in enumerate(items):
            r, c = i // 2, (i % 2) * 2
            self._metrics_grid.addWidget(label(k, "Muted"), r, c)
            self._metrics_grid.addWidget(QLabel(str(v)), r, c + 1)

    def _update_metrics_final(self, report) -> None:
        cpu = psutil.cpu_percent(interval=None) if psutil else 0.0
        ram = psutil.virtual_memory().used / (1024 * 1024) if psutil else 0.0
        m = self._controller.pipeline_metrics(report, elapsed_s=self._elapsed, cpu=cpu, ram_mb=ram)
        self._set_metrics({
            "Images Processed": m.images_processed, "Images Remaining": m.images_remaining,
            "Images / sec": m.images_per_second, "Avg Latency": f"{m.avg_latency_ms} ms",
            "GPU Usage": "—" if m.gpu_util is None else f"{m.gpu_util}%",
            "GPU Memory": "—" if m.gpu_mem_mb is None else f"{m.gpu_mem_mb} MB",
            "CPU Usage": f"{m.cpu_percent:.0f}%", "RAM Usage": f"{m.ram_mb:.0f} MB",
            "Elapsed": f"{m.elapsed_s} s", "Throughput": f"{m.throughput_ips} img/s",
            "Export Count": m.export_count, "Failed": m.failed,
            "Skipped": m.skipped, "Duplicates Removed": m.duplicates_removed,
        })

    def _set_activity(self, activity) -> None:
        while self._activity.count():
            item = self._activity.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if activity is None:
            self._activity.addWidget(label("Model activity appears once a run completes.", "Muted"))
            return
        blocks = [
            ("Detection", [("Model", activity.detection["model"]),
                           ("Device", activity.detection["device"]),
                           ("Inference", f"{activity.detection['inference_ms']} ms"),
                           ("Objects", str(activity.detection["objects"])),
                           ("Avg Confidence", str(activity.detection["avg_confidence"]))]),
            ("Segmentation", [("Model", activity.segmentation["model"]),
                              ("Masks", str(activity.segmentation["masks"])),
                              ("Avg IoU", str(activity.segmentation["avg_iou"])),
                              ("Inference", f"{activity.segmentation['inference_ms']} ms")]),
            ("Verification", [("VLM/Model", activity.verification["model"]),
                              ("Flagged", str(activity.verification["flagged"])),
                              ("Result", activity.verification["result"]),
                              ("Confidence", str(activity.verification["confidence"])),
                              ("Reason", activity.verification["reason"])]),
            ("Quality Analysis", [("Recommendations", "; ".join(activity.quality["recommendations"][:3])),
                                  ("Warnings", "; ".join(activity.quality["warnings"]))]),
            ("Engineering Memory", [("Match", activity.memory["match"]),
                                    ("Similarity", str(activity.memory["similarity"])),
                                    ("Historical Strategy", activity.memory["strategy"]),
                                    ("Prev Review", activity.memory["review_reduction"])]),
        ]
        for title, rows in blocks:
            self._activity.addWidget(label(title, "H2"))
            for k, v in rows:
                line = QWidget()
                h = QHBoxLayout(line)
                h.setContentsMargins(0, 0, 0, 0)
                h.addWidget(label(k, "Muted"))
                h.addStretch(1)
                val = QLabel(v)
                val.setWordWrap(True)
                h.addWidget(val, 2)
                self._activity.addWidget(line)

    def _render_summary(self, s) -> None:
        rows = [
            ("Dataset", s.dataset), ("Execution Time", f"{s.execution_time_s} s"),
            ("Total Images", str(s.total_images)), ("Processed", str(s.processed)),
            ("Successful", str(s.successful)), ("Failed", str(s.failed)),
            ("Skipped", str(s.skipped)), ("Duplicate Images", str(s.duplicates)),
            ("Avg Detection Time", f"{s.avg_detection_ms} ms"),
            ("Avg Segmentation Time", f"{s.avg_segmentation_ms} ms"),
            ("Avg Verification Time", f"{s.avg_verification_ms} ms"),
            ("Avg Review Rate", f"{s.avg_review_rate:.0%}"),
            ("Planner Strategy", s.planner_strategy),
            ("Engineering Memory Influence", s.memory_influence),
            ("Analyst Summary", s.analyst_summary),
            ("Export Statistics", s.export_stats),
        ]
        while self._summary_grid.count():
            item = self._summary_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, (k, v) in enumerate(rows):
            self._summary_grid.addWidget(label(k, "Muted"), i, 0, Qt.AlignmentFlag.AlignTop)
            val = QLabel(v)
            val.setWordWrap(True)
            self._summary_grid.addWidget(val, i, 1)
        self._summary.setVisible(True)

    # =================== preview ===================
    def _load_previews(self, report) -> None:
        self._preview_items = self._controller.pipeline_preview(report.project_id, 12)
        self._preview_idx = 0
        self._show_preview()

    def _show_preview(self) -> None:
        if not self._preview_items:
            return
        item = self._preview_items[self._preview_idx]
        boxes = getattr(item, "boxes", [])
        self._orig.set_image(item.path, [])
        self._annot.set_image(item.path, boxes)
        self._preview_caption.setText(
            f"{getattr(item, 'name', '?')}  ·  {getattr(item, 'width', 0)}×"
            f"{getattr(item, 'height', 0)}  ·  image {self._preview_idx + 1}/{len(self._preview_items)}"
            f"  ·  {len(boxes)} annotations")

    def _prev_image(self) -> None:
        if self._preview_items:
            self._preview_idx = (self._preview_idx - 1) % len(self._preview_items)
            self._show_preview()

    def _next_image(self) -> None:
        if self._preview_items:
            self._preview_idx = (self._preview_idx + 1) % len(self._preview_items)
            self._show_preview()

    def _zoom_in(self) -> None:
        self._orig.zoom_in()
        self._annot.zoom_in()

    def _zoom_out(self) -> None:
        self._orig.zoom_out()
        self._annot.zoom_out()

    def _fit(self) -> None:
        self._orig.fit()
        self._annot.fit()

    # =================== console ===================
    def _add_log(self, level: str, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._log.append((ts, level, message))
        self._render_console()

    def _render_console(self) -> None:
        sev = self._severity.currentText()
        query = self._search.text().lower()
        colors = {"info": "#8b9096", "warning": "#e0a458", "error": "#e0605e"}
        lines = []
        for ts, level, msg in self._log:
            if sev != "All" and level != sev:
                continue
            if query and query not in msg.lower():
                continue
            c = colors.get(level, "#8b9096")
            lines.append(f'<span style="color:{c}">[{ts} {level.upper():7}]</span> {msg}')
        self._console.clear()
        self._console.appendHtml("<br>".join(lines))
        if self._autoscroll.isChecked():
            sb = self._console.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _copy_log(self) -> None:
        from PySide6.QtWidgets import QApplication

        text = "\n".join(f"[{ts} {lvl.upper()}] {msg}" for ts, lvl, msg in self._log)
        QApplication.clipboard().setText(text)
        self._notify.info("Log copied to clipboard.")

    def _save_log(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save log", "pipeline_log.txt", "Text (*.txt)")
        if not path:
            return
        text = "\n".join(f"[{ts} {lvl.upper()}] {msg}" for ts, lvl, msg in self._log)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        self._notify.success(f"Log saved: {path}")

    # =================== summary export ===================
    def _export_summary(self, fmt: str) -> None:
        if self._report is None:
            self._notify.warning("Run the pipeline first.")
            return
        markdown = self._controller.report_markdown(self._report)
        if fmt == "md":
            path, _ = QFileDialog.getSaveFileName(self, "Export summary", "pipeline_summary.md",
                                                  "Markdown (*.md)")
            if not path:
                return
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(markdown)
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Export summary", "pipeline_summary.pdf",
                                                  "PDF (*.pdf)")
            if not path:
                return
            self._write_pdf(markdown, path)
        self._notify.success(f"Summary exported: {path}")

    @staticmethod
    def _write_pdf(markdown: str, path: str) -> None:
        # Qt renders the report to PDF — no extra dependency (uses the existing
        # reporting markdown as the source).
        from PySide6.QtGui import QPageSize, QPdfWriter, QTextDocument

        writer = QPdfWriter(path)
        writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        doc = QTextDocument()
        doc.setMarkdown(markdown)
        doc.print_(writer)

    # =================== state helpers ===================
    def _set_status(self) -> None:
        text = {
            "idle": "Idle — start a run to monitor execution.",
            "running": "● Running — the pipeline is executing.",
            "paused": "❚❚ Monitoring paused (backend continues).",
            "cancelling": "Cancelling — will roll back on finish.",
            "cancelled": "Cancelled — dataset rolled back.",
            "completed": "✓ Completed.",
            "failed": "✕ Failed — see the console.",
        }.get(self._state, self._state)
        self._status.setText(text)

    def _refresh_controls(self) -> None:
        running = self._state == "running"
        paused = self._state == "paused"
        active = running or paused or self._state == "cancelling"
        done = self._state in ("completed", "failed", "cancelled")
        self._buttons["start"].setEnabled(not active)
        self._buttons["pause"].setEnabled(running)
        self._buttons["resume"].setEnabled(paused)
        self._buttons["cancel"].setEnabled(running or paused)
        self._buttons["restart"].setEnabled(not active and self._source is not None)
        self._buttons["output"].setEnabled(self._report is not None or done)
        self._buttons["export"].setEnabled(self._report is not None)
        self._buttons["report"].setEnabled(self._report is not None)
        self._set_status()

    def context(self) -> tuple[str, list[tuple[str, str]]]:
        return ("Pipeline", [
            ("State", self._state),
            ("Elapsed", f"{self._elapsed:.0f} s"),
            ("Log lines", str(len(self._log))),
        ])


class _PreviewStub:
    """A source image (pre-run) with no annotations — cycled during execution."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        self.width = 0
        self.height = 0
        self.boxes: list = []
