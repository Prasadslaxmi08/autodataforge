"""Planner Workspace (Phase 12) — the central AI decision-support page.

Four panels (Dataset Summary · AI Planning · Engineering Memory · Plan Evaluation),
engineer controls that re-run ONLY the Planner (never annotation), a plan
comparison, and the pipeline-launch action. Everything shown originates from the
existing Planner Agent / Engineering Memory via BackendController — no backend
logic here, and long Planner calls run on a worker thread.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vds.gui.controller import BackendController
from vds.gui.notifications import NotificationSystem
from vds.gui.pages.base import Page
from vds.gui.planner_view import PlanControls, PlanView, diff_plans
from vds.gui.threads import ThreadManager
from vds.gui.widgets.common import Card, label

_DECISION_COLS = ["Decision", "Value", "Reason", "Confidence", "Expected Impact",
                  "Trade-offs", "Validation"]
_MEMORY_COLS = ["Dataset", "Similarity", "Previous Strategy", "Review", "Runtime", "Analyst Rec"]


class PlannerPage(Page):
    name = "Planner"

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
        self._original: PlanView | None = None
        self._current: PlanView | None = None
        self._busy = False

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

        self._root.addWidget(label("Planner Workspace", "H1"))
        self._root.addWidget(label("Understand, validate, modify, and approve the AI "
                                   "annotation plan before running the pipeline.", "Muted"))
        self._root.addLayout(self._action_bar())

        grid = QGridLayout()
        grid.setSpacing(12)
        grid.addWidget(self._summary_card(), 0, 0)
        grid.addWidget(self._eval_card(), 0, 1)
        holder = QWidget()
        holder.setLayout(grid)
        self._root.addWidget(holder)

        self._root.addWidget(self._decisions_card())
        self._root.addWidget(self._memory_card())
        self._root.addWidget(self._controls_card())
        self._root.addWidget(self._comparison_card())
        self._root.addStretch(1)

    # --- top bar ---
    def _action_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addWidget(label("Dataset", "Muted"))
        self._dataset = QComboBox()
        self._dataset.setMinimumWidth(220)
        bar.addWidget(self._dataset)
        self._btn_generate = QPushButton("Generate Plan")
        self._btn_generate.setObjectName("Primary")
        self._btn_generate.clicked.connect(self._generate)
        self._btn_accept = QPushButton("Accept Plan")
        self._btn_accept.clicked.connect(self._accept)
        self._btn_restore = QPushButton("Restore AI Recommendation")
        self._btn_restore.clicked.connect(self._restore)
        self._btn_run = QPushButton("Run Annotation Pipeline")
        self._btn_run.clicked.connect(self._run_pipeline)
        for b in (self._btn_generate, self._btn_accept, self._btn_restore, self._btn_run):
            bar.addWidget(b)
        bar.addStretch(1)
        return bar

    # --- Panel 1: Dataset Summary ---
    def _summary_card(self) -> Card:
        card = Card("Dataset Summary")
        self._summary = QGridLayout()
        self._summary.setColumnStretch(1, 1)
        card.body.addLayout(self._summary)
        self._set_grid(self._summary, [("Status", "Select a dataset and Generate Plan.")])
        return card

    # --- Panel 4: Plan Evaluation ---
    def _eval_card(self) -> Card:
        card = Card("Plan Evaluation")
        self._eval = QGridLayout()
        self._eval.setColumnStretch(1, 1)
        card.body.addLayout(self._eval)
        self._eval_notes = QVBoxLayout()
        card.body.addLayout(self._eval_notes)
        return card

    # --- Panel 2: AI Planning ---
    def _decisions_card(self) -> Card:
        card = Card("AI Planning — every Planner decision")
        self._decisions = QTableWidget(0, len(_DECISION_COLS))
        self._decisions.setHorizontalHeaderLabels(_DECISION_COLS)
        self._decisions.verticalHeader().setVisible(False)
        self._decisions.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._decisions.setWordWrap(True)
        h = self._decisions.horizontalHeader()
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        card.add(self._decisions)
        return card

    # --- Panel 3: Engineering Memory ---
    def _memory_card(self) -> Card:
        card = Card("Engineering Memory")
        self._memory_note = label("Query runs automatically when a plan is generated.", "Muted", wrap=True)
        card.add(self._memory_note)
        self._memory = QTableWidget(0, len(_MEMORY_COLS))
        self._memory.setHorizontalHeaderLabels(_MEMORY_COLS)
        self._memory.verticalHeader().setVisible(False)
        self._memory.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._memory.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        card.add(self._memory)
        return card

    # --- Interactive Controls ---
    def _controls_card(self) -> Card:
        card = Card("Interactive Controls — re-runs only the Planner (no annotation)")
        form = QGridLayout()
        form.setHorizontalSpacing(14)
        self._c_detector = QComboBox()
        self._c_detector.addItems(self._controller.detector_options())
        self._c_seg = QCheckBox("Enabled")
        self._c_conf = QDoubleSpinBox()
        self._c_conf.setRange(0.0, 1.0)
        self._c_conf.setSingleStep(0.05)
        self._c_batch = QSpinBox()
        self._c_batch.setRange(1, 1024)
        self._c_workers = QSpinBox()
        self._c_workers.setRange(1, 64)
        self._c_export = QComboBox()
        self._c_export.addItems(self._controller.export_options())
        fields = [
            ("Detector", self._c_detector), ("Segmentation", self._c_seg),
            ("Confidence", self._c_conf), ("Batch Size", self._c_batch),
            ("Worker Count", self._c_workers), ("Export Format", self._c_export),
        ]
        for i, (name, widget) in enumerate(fields):
            form.addWidget(label(name, "Muted"), i // 3, (i % 3) * 2)
            form.addWidget(widget, i // 3, (i % 3) * 2 + 1)
        card.body.addLayout(form)
        self._btn_update = QPushButton("Update Plan")
        self._btn_update.clicked.connect(self._update)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._btn_update)
        card.body.addLayout(row)
        return card

    # --- Plan Comparison ---
    def _comparison_card(self) -> Card:
        self._comparison = Card("Plan Comparison — Original AI Plan vs Modified")
        self._compare_table = QTableWidget(0, 4)
        self._compare_table.setHorizontalHeaderLabels(["Field", "Original", "Modified", "Δ"])
        self._compare_table.verticalHeader().setVisible(False)
        self._compare_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._compare_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._comparison.add(self._compare_table)
        self._comparison.setVisible(False)
        return self._comparison

    # --- lifecycle ---
    def on_show(self) -> None:
        current = self._dataset.currentData()
        self._dataset.blockSignals(True)
        self._dataset.clear()
        for d in self._controller.list_datasets():
            self._dataset.addItem(f"{d.name}  ({d.image_count} imgs)", d.project_id)
        idx = self._dataset.findData(current)
        if idx >= 0:
            self._dataset.setCurrentIndex(idx)
        self._dataset.blockSignals(False)

    # --- planner runs (threaded) ---
    def _selected_project(self) -> str | None:
        return self._dataset.currentData()

    def _run_plan(self, overrides: PlanControls | None, done) -> None:
        pid = self._selected_project()
        if pid is None:
            self._notify.warning("Import a dataset first, then generate a plan.")
            return
        self._set_busy(True)
        self._notify.info("Running the Planner…")
        self._threads.submit(
            self._controller.plan_dataset, pid, overrides,
            on_finished=lambda view: self._on_planned(view, done),
            on_error=self._on_error,
        )

    def _on_planned(self, view: PlanView, done) -> None:
        self._set_busy(False)
        self._current = view
        done(view)
        self._populate(view)
        self._notify.success(
            f"Plan ready ({view.source}). "
            + (view.memory_note if view.memory_used else "No historical match found."))

    def _on_error(self, message: str) -> None:
        self._set_busy(False)
        self._notify.error(f"Planner failed — {message}")

    def _generate(self) -> None:
        def done(view: PlanView) -> None:
            self._original = view
            self._sync_controls(view)
            self._comparison.setVisible(False)
        self._run_plan(None, done)

    def _update(self) -> None:
        if self._original is None:
            self._notify.warning("Generate a plan first.")
            return
        overrides = PlanControls(
            detector=self._c_detector.currentText(),
            segmentation=self._c_seg.isChecked(),
            confidence_threshold=round(self._c_conf.value(), 4),
            batch_size=self._c_batch.value(),
            worker_count=self._c_workers.value(),
            export_format=self._c_export.currentText(),
        )

        def done(view: PlanView) -> None:
            self._show_comparison(self._original, view)
        self._run_plan(overrides, done)

    def _restore(self) -> None:
        def done(view: PlanView) -> None:
            self._original = view
            self._sync_controls(view)
            self._comparison.setVisible(False)
            self._notify.info("Restored the Planner's original recommendation.")
        self._run_plan(None, done)

    def _accept(self) -> None:
        if self._current is None:
            self._notify.warning("Generate a plan before accepting.")
            return
        c = self._current.effective_controls
        self._notify.success(
            f"Plan accepted: detector={c.detector}, conf={c.confidence_threshold}, "
            f"export={c.export_format}. Ready to run the pipeline.")

    def _run_pipeline(self) -> None:
        if self._busy:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select the source image folder to process")
        if not folder:
            return
        name = (self._current.profile.name if self._current else "planned") + "-run"
        self._set_busy(True)
        self._notify.info(f"Launching the annotation pipeline for '{name}'…")
        self._threads.submit(
            self._controller.import_dataset, folder, name,
            wants_progress=True,
            on_progress=lambda pct, msg: self._notify.info(msg),
            on_finished=self._on_pipeline_done,
            on_error=self._on_error,
        )

    def _on_pipeline_done(self, report) -> None:
        self._set_busy(False)
        self._notify.success(
            f"Pipeline finished: {report.imported} images, {report.detections} annotations, "
            f"{report.verified_approved} approved.")
        self.on_show()

    # --- populate panels ---
    def _populate(self, view: PlanView) -> None:
        p = view.profile
        self._set_grid(self._summary, [
            ("Dataset Name", p.name), ("Dataset Size", f"{p.storage_mb} MB"),
            ("Image Count", str(p.image_count)), ("Average Resolution", p.avg_resolution),
            ("Class Distribution", ", ".join(f"{k}: {v}" for k, v in p.class_distribution.items())),
            ("Small Object %", "—" if p.small_object_pct is None else f"{p.small_object_pct}%"),
            ("Duplicate %", "—" if p.duplicate_pct is None else f"{p.duplicate_pct}%"),
            ("Estimated Difficulty", p.difficulty),
            ("Dataset Fingerprint", p.fingerprint), ("Import Date", p.import_date),
            ("Dataset Version", str(p.version)),
        ])
        self._fill_decisions(view)
        self._fill_memory(view)
        self._fill_eval(view)

    def _fill_decisions(self, view: PlanView) -> None:
        self._decisions.setRowCount(len(view.decisions))
        for r, d in enumerate(view.decisions):
            for c, val in enumerate([d.name, d.value, d.reason, d.confidence,
                                     d.expected_impact, d.trade_offs, d.validation]):
                self._decisions.setItem(r, c, QTableWidgetItem(val))
        self._decisions.resizeRowsToContents()

    def _fill_memory(self, view: PlanView) -> None:
        if view.memory_used:
            self._memory_note.setText(
                f"✓ {view.memory_note}  Previous experience influenced this plan.")
        else:
            self._memory_note.setText("No similar dataset found in Engineering Memory — "
                                      "this plan is generated from the dataset alone.")
        self._memory.setRowCount(len(view.memory_matches))
        for r, m in enumerate(view.memory_matches):
            cells = [m.dataset, f"{m.similarity:.2f}", m.strategy, f"{m.review_rate:.0%}",
                     m.runtime, m.analyst_recommendation]
            for c, val in enumerate(cells):
                self._memory.setItem(r, c, QTableWidgetItem(val))

    def _fill_eval(self, view: PlanView) -> None:
        e = view.evaluation
        rows = [
            ("Estimated Runtime", f"{e.runtime_s} s"),
            ("Estimated Throughput", f"{e.throughput_ips} img/s"),
            ("Estimated Review Rate", f"{e.review_rate:.0%}"),
            ("Expected Annotation Quality", f"{e.quality:.0%}"),
            ("Expected GPU Utilization", "—" if e.gpu_util_pct is None else f"{e.gpu_util_pct}%"),
            ("Estimated Memory Usage", "—" if e.memory_mb is None else f"{e.memory_mb} MB"),
            ("Estimated Cost", "local (no cost)" if e.cost_usd is None else f"${e.cost_usd}"),
            ("Planner Confidence", "—" if e.confidence is None else str(e.confidence)),
        ]
        self._set_grid(self._eval, rows)
        while self._eval_notes.count():
            item = self._eval_notes.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for w in view.warnings:
            self._eval_notes.addWidget(label(f"⚠ {w}", "Muted", wrap=True))
        for rk in view.risks:
            self._eval_notes.addWidget(label(f"● Risk: {rk}", "Muted", wrap=True))

    def _show_comparison(self, original: PlanView, modified: PlanView) -> None:
        rows = diff_plans(original, modified)
        self._comparison.setVisible(True)
        self._compare_table.setRowCount(len(rows) or 1)
        if not rows:
            self._compare_table.setItem(0, 0, QTableWidgetItem("No differences"))
            self._compare_table.setSpan(0, 0, 1, 4)
            self._notify.info("Modified plan is identical to the original.")
            return
        for r, row in enumerate(rows):
            for c, val in enumerate([row.field, row.original, row.modified, row.delta]):
                self._compare_table.setItem(r, c, QTableWidgetItem(val))
        self._notify.info(f"Plan updated — {len(rows)} field(s) changed.")

    # --- helpers ---
    def _sync_controls(self, view: PlanView) -> None:
        c = view.effective_controls
        i = self._c_detector.findText(c.detector)
        if i >= 0:
            self._c_detector.setCurrentIndex(i)
        self._c_seg.setChecked(bool(c.segmentation))
        self._c_conf.setValue(c.confidence_threshold or 0.0)
        self._c_batch.setValue(c.batch_size or 1)
        self._c_workers.setValue(c.worker_count or 1)
        j = self._c_export.findText(c.export_format or "coco")
        if j >= 0:
            self._c_export.setCurrentIndex(j)

    def _set_grid(self, grid: QGridLayout, rows: list[tuple[str, str]]) -> None:
        while grid.count():
            item = grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, (k, v) in enumerate(rows):
            grid.addWidget(label(k, "Muted"), i, 0, Qt.AlignmentFlag.AlignTop)
            val = QLabel(v)
            val.setWordWrap(True)
            grid.addWidget(val, i, 1)

    def _set_busy(self, value: bool) -> None:
        self._busy = value
        for b in (self._btn_generate, self._btn_update, self._btn_restore,
                  self._btn_run, self._btn_accept):
            b.setEnabled(not value)
        self.busy.emit(value)

    def context(self) -> tuple[str, list[tuple[str, str]]]:
        if self._current is None:
            return ("Planner", [("Status", "No plan generated")])
        v = self._current
        return ("Plan Summary", [
            ("Source", v.source),
            ("Memory", "used" if v.memory_used else "no match"),
            ("Confidence", "—" if v.evaluation.confidence is None else str(v.evaluation.confidence)),
            ("Review rate", f"{v.evaluation.review_rate:.0%}"),
            ("Warnings", str(len(v.warnings))),
        ])
