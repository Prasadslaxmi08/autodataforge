"""AI Verification Workspace (Phase 14) — inspect and act on verification results.

Five areas: image inspection, verification results, AI evidence, historical
comparison, and statistics — plus filtering, human-review actions, and a decision
timeline. It VISUALIZES verification: verdicts are reproduced deterministically by
the existing verifier through BackendController, evidence scores come from measured
outputs, and anything the backend can't supply (timestamps, original runtime) is
shown as unavailable — never invented.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vds.gui.controller import BackendController
from vds.gui.notifications import NotificationSystem
from vds.gui.pages.base import Page
from vds.gui.threads import ThreadManager
from vds.gui.verification_view import STATUS_COLOR
from vds.gui.widgets.common import Card, label
from vds.gui.widgets.image_preview import ImagePreview

_RESULT_COLS = ["Object ID", "Class", "Det. Conf", "Verification Status",
                "Ver. Conf", "Suggested Action"]
_STATUSES = ["All", "Verified", "Needs Review", "Rejected", "Uncertain"]
_REVIEW = [("Approve", "approve"), ("Reject", "reject"), ("Mark for Review", "mark_review"),
           ("Accept Detection", "accept_detection"), ("Reject Detection", "reject_detection")]


def _stars(value) -> str:
    if value is None:
        return "unavailable"
    return "★" * value + "☆" * (5 - value)


class VerificationPage(Page):
    name = "VLM Verification"

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
        self._verdicts: list = []
        self._selected = None
        self._project_id: str | None = None

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

        self._root.addWidget(label("AI Verification Workspace", "H1"))
        self._root.addWidget(label("Understand why each object was verified, flagged, "
                                   "or rejected — through measured evidence.", "Muted"))
        self._root.addLayout(self._top_bar())
        self._root.addLayout(self._filter_bar())

        grid = QGridLayout()
        grid.setSpacing(12)
        grid.addWidget(self._inspection_card(), 0, 0)
        grid.addWidget(self._evidence_card(), 0, 1)
        holder = QWidget()
        holder.setLayout(grid)
        self._root.addWidget(holder)

        self._root.addWidget(self._results_card())
        self._root.addLayout(self._review_bar())

        grid2 = QGridLayout()
        grid2.setSpacing(12)
        grid2.addWidget(self._history_card(), 0, 0)
        grid2.addWidget(self._stats_card(), 0, 1)
        holder2 = QWidget()
        holder2.setLayout(grid2)
        self._root.addWidget(holder2)

        self._root.addWidget(self._timeline_card())
        self._root.addStretch(1)

    # --- bars ---
    def _top_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addWidget(label("Dataset", "Muted"))
        self._dataset = QComboBox()
        self._dataset.setMinimumWidth(220)
        bar.addWidget(self._dataset)
        load = QPushButton("Load Verification")
        load.setObjectName("Primary")
        load.clicked.connect(self._load)
        bar.addWidget(load)
        report = QPushButton("Generate Verification Report")
        report.clicked.connect(self._generate_report)
        bar.addWidget(report)
        prev = QPushButton("View Previous Cases")
        prev.clicked.connect(self._view_previous)
        bar.addWidget(prev)
        bar.addStretch(1)
        return bar

    def _filter_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self._f_status = QComboBox()
        self._f_status.addItems(_STATUSES)
        self._f_class = QComboBox()
        self._f_class.addItem("All classes")
        self._f_detconf = QDoubleSpinBox()
        self._f_detconf.setRange(0.0, 1.0)
        self._f_detconf.setSingleStep(0.05)
        self._f_verconf = QDoubleSpinBox()
        self._f_verconf.setRange(0.0, 1.0)
        self._f_verconf.setSingleStep(0.05)
        self._f_image = QLineEdit()
        self._f_image.setPlaceholderText("Search image…")
        self._f_label = QLineEdit()
        self._f_label.setPlaceholderText("Search label…")
        for w in (self._f_status, self._f_class):
            w.currentTextChanged.connect(lambda _t: self._render_table())
        for w in (self._f_detconf, self._f_verconf):
            w.valueChanged.connect(lambda _v: self._render_table())
        for w in (self._f_image, self._f_label):
            w.textChanged.connect(lambda _t: self._render_table())
        for name, w in [("Status", self._f_status), ("Class", self._f_class),
                        ("Min Det.Conf", self._f_detconf), ("Min Ver.Conf", self._f_verconf),
                        ("Image", self._f_image), ("Label", self._f_label)]:
            bar.addWidget(label(name, "Muted"))
            bar.addWidget(w)
        bar.addStretch(1)
        return bar

    def _review_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addWidget(label("Human Review:", "Muted"))
        self._review_buttons = []
        for text, action in _REVIEW:
            b = QPushButton(text)
            b.clicked.connect(lambda _checked=False, a=action: self._review(a))
            b.setEnabled(False)
            self._review_buttons.append(b)
            bar.addWidget(b)
        bar.addStretch(1)
        return bar

    # --- area 1: image inspection ---
    def _inspection_card(self) -> Card:
        card = Card("Image Inspection")
        self._img_meta = label("Select an object to inspect.", "Muted")
        card.add(self._img_meta)
        toggles = QHBoxLayout()
        self._t_boxes = QCheckBox("Boxes")
        self._t_boxes.setChecked(True)
        self._t_labels = QCheckBox("Labels")
        self._t_labels.setChecked(True)
        self._t_conf = QCheckBox("Confidence")
        self._t_conf.setChecked(True)
        self._t_masks = QCheckBox("Masks")
        self._t_masks.setEnabled(False)  # masks-overlay from RLE is unavailable in this view
        for t in (self._t_boxes, self._t_labels, self._t_conf, self._t_masks):
            t.stateChanged.connect(lambda _s: self._show_selected_image())
            toggles.addWidget(t)
        toggles.addStretch(1)
        card.body.addLayout(toggles)
        row = QHBoxLayout()
        self._orig = ImagePreview()
        self._orig.setMinimumHeight(220)
        self._overlay = ImagePreview()
        self._overlay.setMinimumHeight(220)
        row.addWidget(self._orig)
        row.addWidget(self._overlay)
        card.body.addLayout(row)
        nav = QHBoxLayout()
        for text, slot in [("Zoom +", self._zoom_in), ("Zoom −", self._zoom_out),
                           ("Fit", self._fit)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            nav.addWidget(b)
        nav.addStretch(1)
        card.body.addLayout(nav)
        return card

    # --- area 3: evidence ---
    def _evidence_card(self) -> Card:
        card = Card("AI Evidence")
        self._evidence = QVBoxLayout()
        card.body.addLayout(self._evidence)
        self._set_evidence(None)
        return card

    # --- area 2: results ---
    def _results_card(self) -> Card:
        card = Card("Verification Results")
        self._table = QTableWidget(0, len(_RESULT_COLS))
        self._table.setHorizontalHeaderLabels(_RESULT_COLS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        card.add(self._table)
        return card

    # --- area 4: history ---
    def _history_card(self) -> Card:
        card = Card("Historical Comparison")
        self._history_note = label("Load a dataset to query Engineering Memory.", "Muted", wrap=True)
        card.add(self._history_note)
        self._history = QTableWidget(0, 6)
        self._history.setHorizontalHeaderLabels(
            ["Dataset", "Similarity", "Historical Verification", "Past Corrections",
             "Agreement", "Prev Decision"])
        self._history.verticalHeader().setVisible(False)
        self._history.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._history.horizontalHeader().setStretchLastSection(True)
        card.add(self._history)
        return card

    # --- area 5: statistics ---
    def _stats_card(self) -> Card:
        card = Card("Verification Statistics")
        self._stats = QGridLayout()
        self._stats.setColumnStretch(1, 1)
        card.body.addLayout(self._stats)
        return card

    # --- timeline ---
    def _timeline_card(self) -> Card:
        card = Card("Verification Timeline")
        self._timeline = QHBoxLayout()
        card.body.addLayout(self._timeline)
        self._set_timeline(None)
        return card

    # =================== lifecycle ===================
    def on_show(self) -> None:
        cur = self._dataset.currentData()
        self._dataset.blockSignals(True)
        self._dataset.clear()
        for d in self._controller.list_datasets():
            self._dataset.addItem(f"{d.name}  ({d.annotation_count} objs)", d.project_id)
        idx = self._dataset.findData(cur)
        if idx >= 0:
            self._dataset.setCurrentIndex(idx)
        self._dataset.blockSignals(False)

    # =================== load (threaded) ===================
    def _load(self) -> None:
        pid = self._dataset.currentData()
        if pid is None:
            self._notify.warning("Import a dataset first.")
            return
        self._project_id = pid
        self.busy.emit(True)
        self._notify.info("Reproducing verification verdicts…")
        self._threads.submit(
            self._controller.object_verdicts, pid,
            on_finished=self._on_loaded, on_error=self._on_error)

    def _on_loaded(self, verdicts) -> None:
        self.busy.emit(False)
        self._verdicts = verdicts
        classes = sorted({v.label for v in verdicts})
        self._f_class.blockSignals(True)
        self._f_class.clear()
        self._f_class.addItem("All classes")
        self._f_class.addItems(classes)
        self._f_class.blockSignals(False)
        self._render_table()
        self._render_stats()
        self._render_history()
        self._notify.success(f"Loaded {len(verdicts)} verification results.")

    def _on_error(self, message: str) -> None:
        self.busy.emit(False)
        self._notify.error(f"Verification load failed — {message}")

    # =================== table + filtering ===================
    def _filtered(self) -> list:
        status = self._f_status.currentText()
        cls = self._f_class.currentText()
        min_det = self._f_detconf.value()
        min_ver = self._f_verconf.value()
        img_q = self._f_image.text().lower()
        lbl_q = self._f_label.text().lower()
        out = []
        for v in self._verdicts:
            if status != "All" and v.status != status:
                continue
            if cls != "All classes" and v.label != cls:
                continue
            if v.detection_confidence < min_det or v.verification_confidence < min_ver:
                continue
            if img_q and img_q not in v.image_name.lower():
                continue
            if lbl_q and lbl_q not in v.label.lower():
                continue
            out.append(v)
        return out

    def _render_table(self) -> None:
        rows = self._filtered()
        self._table.setRowCount(len(rows))
        for r, v in enumerate(rows):
            cells = [v.object_id[:8], v.label, f"{v.detection_confidence:.2f}",
                     v.status, f"{v.verification_confidence:.2f}", v.suggested_action]
            for c, val in enumerate(cells):
                item = QTableWidgetItem(val)
                if c == 3:
                    item.setForeground(QColor(STATUS_COLOR.get(v.status, "#d7dae0")))
                item.setData(Qt.ItemDataRole.UserRole, v.object_id)
                self._table.setItem(r, c, item)
        self._table.resizeColumnsToContents()

    def _on_row_selected(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            return
        item = self._table.item(row, 0)
        oid = item.data(Qt.ItemDataRole.UserRole) if item else None
        self._selected = next((v for v in self._verdicts if v.object_id == oid), None)
        for b in self._review_buttons:
            b.setEnabled(self._selected is not None)
        if self._selected is not None:
            self._set_evidence(self._controller.object_evidence(self._selected))
            self._set_timeline(self._controller.verification_timeline(self._selected))
            self._show_selected_image()

    # =================== image inspection ===================
    def _show_selected_image(self) -> None:
        if self._selected is None:
            return
        path = self._controller.image_path(self._selected.image_id)
        meta = self._controller.image_meta(self._selected.image_id)
        if not path:
            self._img_meta.setText("Image unavailable.")
            return
        position = next((i for i, v in enumerate(self._verdicts)
                         if v.image_id == self._selected.image_id), 0)
        self._img_meta.setText(
            f"{meta['name']}  ·  {meta['resolution']}  ·  object {self._selected.object_id[:8]}"
            f"  ·  dataset position {position + 1}/{len(self._verdicts)}")
        boxes = []
        if self._t_boxes.isChecked() and self._selected.box is not None:
            x, y, w, h = self._selected.box
            boxes = [(x, y, w, h, self._selected.label, self._selected.detection_confidence)]
        self._orig.set_image(path, [])
        self._overlay.set_image(path, boxes, show_labels=self._t_labels.isChecked(),
                                show_conf=self._t_conf.isChecked())

    def _zoom_in(self) -> None:
        self._orig.zoom_in()
        self._overlay.zoom_in()

    def _zoom_out(self) -> None:
        self._orig.zoom_out()
        self._overlay.zoom_out()

    def _fit(self) -> None:
        self._orig.fit()
        self._overlay.fit()

    # =================== evidence ===================
    def _set_evidence(self, evidence) -> None:
        while self._evidence.count():
            item = self._evidence.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if evidence is None:
            self._evidence.addWidget(label("Select an object to see its evidence.", "Muted"))
            return
        self._evidence.addWidget(label(evidence.summary, "H2"))
        rows = [("Reason", evidence.reason),
                ("Verification Confidence", f"{evidence.verification_confidence:.0%}"),
                ("Risk", evidence.risk), ("Recommendation", evidence.recommendation)]
        for k, v in rows:
            line = QWidget()
            h = QHBoxLayout(line)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(label(k, "Muted"))
            h.addStretch(1)
            val = QLabel(v)
            val.setWordWrap(True)
            h.addWidget(val, 2)
            self._evidence.addWidget(line)
        self._evidence.addWidget(label("Evidence", "Muted"))
        for e in evidence.evidence:
            self._evidence.addWidget(label(f"• {e}", wrap=True))
        self._evidence.addWidget(label("Evidence Visualization", "Muted"))
        for s in evidence.stars:
            line = QWidget()
            h = QHBoxLayout(line)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(label(s.label, "Muted"))
            h.addStretch(1)
            h.addWidget(QLabel(f"{_stars(s.value)}  {s.detail}"))
            self._evidence.addWidget(line)

    # =================== stats ===================
    def _render_stats(self) -> None:
        s = self._controller.verification_stats(self._verdicts)
        rows = [
            ("Verified Objects", str(s.verified)), ("Rejected Objects", str(s.rejected)),
            ("Needs Review", str(s.needs_review)),
            ("Avg Verification Confidence", f"{s.avg_verification_confidence:.0%}"),
            ("Avg Detection Confidence", f"{s.avg_detection_confidence:.0%}"),
            ("Agreement Rate", f"{s.agreement_rate:.0%}"),
            ("Disagreement Rate", f"{s.disagreement_rate:.0%}"),
            ("Review Percentage", f"{s.review_percentage:.0%}"),
            ("Verification Runtime", s.verification_runtime),
        ]
        while self._stats.count():
            item = self._stats.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, (k, v) in enumerate(rows):
            self._stats.addWidget(label(k, "Muted"), i, 0)
            self._stats.addWidget(QLabel(v), i, 1)

    # =================== history ===================
    def _render_history(self) -> None:
        if self._project_id is None:
            return
        h = self._controller.verification_history(self._project_id)
        prefix = "✓ " if h.influenced else ""
        self._history_note.setText(prefix + h.note)
        self._history.setRowCount(len(h.matches))
        for r, m in enumerate(h.matches):
            cells = [m["dataset"], str(m["similarity"]), m["historical_verification"],
                     m["past_corrections"], m["agreement_rate"], m["previous_review_decision"]]
            for c, val in enumerate(cells):
                self._history.setItem(r, c, QTableWidgetItem(str(val)))

    # =================== timeline ===================
    def _set_timeline(self, steps) -> None:
        while self._timeline.count():
            item = self._timeline.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if steps is None:
            self._timeline.addWidget(label("Select an object to see its decision history.", "Muted"))
            return
        for i, step in enumerate(steps):
            box = QWidget()
            v = QVBoxLayout(box)
            v.setContentsMargins(8, 6, 8, 6)
            v.addWidget(label(step.stage, "H2"))
            v.addWidget(label(step.status, "Muted"))
            v.addWidget(label(f"ts: {step.timestamp}", "Badge"))
            self._timeline.addWidget(box)
            if i < len(steps) - 1:
                self._timeline.addWidget(label("→", "Muted"))
        self._timeline.addStretch(1)

    # =================== human review ===================
    def _review(self, action: str) -> None:
        if self._selected is None:
            return
        ok, msg = self._controller.apply_review(self._selected.object_id, action)
        if ok:
            self._notify.success(msg)
            self._load()  # refresh states/stats via the existing pipeline data
        else:
            self._notify.warning(msg)

    def _view_previous(self) -> None:
        if self._project_id is None:
            self._notify.warning("Load a dataset first.")
            return
        self._render_history()
        h = self._controller.verification_history(self._project_id)
        self._notify.info(h.note)

    def _generate_report(self) -> None:
        if self._project_id is None:
            self._notify.warning("Load a dataset first.")
            return
        markdown = self._controller.verification_report_markdown(self._project_id)
        path, _ = QFileDialog.getSaveFileName(self, "Save verification report",
                                              "verification_report.md", "Markdown (*.md)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(markdown)
        self._notify.success(f"Verification report saved: {path}")

    def context(self) -> tuple[str, list[tuple[str, str]]]:
        return ("Verification", [
            ("Objects", str(len(self._verdicts))),
            ("Selected", self._selected.object_id[:8] if self._selected else "none"),
            ("Dataset", self._dataset.currentText() or "none"),
        ])
