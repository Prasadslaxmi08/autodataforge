"""AI Dataset Intelligence Workspace (Phase 15) — dataset quality at a glance.

Six sections: executive summary (with a health gauge), health dashboard, root-cause
analysis, prioritized recommendations, historical comparison, and dataset readiness.
It SUMMARIZES the existing AI Dataset Analyst through BackendController — the Analyst
runs on the cached ExecutionReport off the UI thread; every value shown is a measured
metric or a validated Analyst recommendation, and unavailable data is labelled, not
invented.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
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
from vds.gui.widgets.common import Card, label
from vds.gui.widgets.gauge import HealthGauge
from vds.gui.widgets.sparkline import Sparkline

_PRIORITY_COLOR = {"HIGH": "#e0605e", "MEDIUM": "#e0a458", "LOW": "#4caf82"}
_PRIORITIES = ["All", "HIGH", "MEDIUM", "LOW"]
_EXPORT_SECTIONS = {
    "Executive Summary": "executive", "Engineering Report": "engineering",
    "Recommendations": "recommendations", "Dataset Health Report": "health",
    "Full Intelligence": "all",
}


class IntelligencePage(Page):
    name = "AI Dataset Analyst"

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
        self._intel = None

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

        self._root.addWidget(label("AI Dataset Intelligence Workspace", "H1"))
        self._root.addWidget(label("Understand dataset quality, prioritize fixes, and decide "
                                   "if it is ready for training.", "Muted"))
        self._root.addLayout(self._top_bar())

        grid = QGridLayout()
        grid.setSpacing(12)
        grid.addWidget(self._summary_card(), 0, 0)
        grid.addWidget(self._health_card(), 0, 1)
        holder = QWidget()
        holder.setLayout(grid)
        self._root.addWidget(holder)

        self._root.addWidget(self._issues_card())
        self._root.addLayout(self._filter_bar())
        self._root.addWidget(self._recommendations_card())

        grid2 = QGridLayout()
        grid2.setSpacing(12)
        grid2.addWidget(self._historical_card(), 0, 0)
        grid2.addWidget(self._readiness_card(), 0, 1)
        holder2 = QWidget()
        holder2.setLayout(grid2)
        self._root.addWidget(holder2)
        self._root.addStretch(1)

    # --- bars ---
    def _top_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addWidget(label("Dataset", "Muted"))
        self._dataset = QComboBox()
        self._dataset.setMinimumWidth(220)
        bar.addWidget(self._dataset)
        analyze = QPushButton("Analyze Dataset")
        analyze.setObjectName("Primary")
        analyze.clicked.connect(self._analyze)
        bar.addWidget(analyze)
        bar.addWidget(label("Export", "Muted"))
        self._export_section = QComboBox()
        self._export_section.addItems(list(_EXPORT_SECTIONS))
        bar.addWidget(self._export_section)
        md = QPushButton("Markdown")
        md.clicked.connect(lambda: self._export("md"))
        pdf = QPushButton("PDF")
        pdf.clicked.connect(lambda: self._export("pdf"))
        bar.addWidget(md)
        bar.addWidget(pdf)
        bar.addStretch(1)
        return bar

    def _filter_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addWidget(label("Recommendations — Priority", "Muted"))
        self._f_priority = QComboBox()
        self._f_priority.addItems(_PRIORITIES)
        self._f_priority.currentTextChanged.connect(lambda _t: self._render_recommendations())
        bar.addWidget(self._f_priority)
        bar.addWidget(label("Min Confidence", "Muted"))
        self._f_conf = QDoubleSpinBox()
        self._f_conf.setRange(0.0, 1.0)
        self._f_conf.setSingleStep(0.05)
        self._f_conf.valueChanged.connect(lambda _v: self._render_recommendations())
        bar.addWidget(self._f_conf)
        self._f_search = QLineEdit()
        self._f_search.setPlaceholderText("Search recommendations…")
        self._f_search.textChanged.connect(lambda _t: self._render_recommendations())
        bar.addWidget(self._f_search, 1)
        return bar

    # --- section 1: executive summary ---
    def _summary_card(self) -> Card:
        card = Card("Executive Summary")
        row = QHBoxLayout()
        self._gauge = HealthGauge("Overall Health")
        row.addWidget(self._gauge)
        self._summary_grid = QGridLayout()
        self._summary_grid.setColumnStretch(1, 1)
        row.addLayout(self._summary_grid, 1)
        card.body.addLayout(row)
        self._rec_badge = label("Analyze a dataset to begin.", "Badge")
        card.add(self._rec_badge)
        return card

    # --- section 2: health dashboard ---
    def _health_card(self) -> Card:
        card = Card("Dataset Health Dashboard")
        self._health = QVBoxLayout()
        card.body.addLayout(self._health)
        return card

    # --- section 3: root cause ---
    def _issues_card(self) -> Card:
        card = Card("Root Cause Analysis")
        self._issues = QVBoxLayout()
        card.body.addLayout(self._issues)
        return card

    # --- section 4: recommendations ---
    def _recommendations_card(self) -> Card:
        card = Card("Prioritized Recommendations")
        self._recommendations = QVBoxLayout()
        card.body.addLayout(self._recommendations)
        return card

    # --- section 5: historical ---
    def _historical_card(self) -> Card:
        card = Card("Historical Comparison")
        self._historical_note = label("Analyze to compare with previous datasets.", "Muted", wrap=True)
        card.add(self._historical_note)
        self._trends = QVBoxLayout()
        card.body.addLayout(self._trends)
        self._matches = QTableWidget(0, 4)
        self._matches.setHorizontalHeaderLabels(["Dataset", "Similarity", "Review", "Quality"])
        self._matches.verticalHeader().setVisible(False)
        self._matches.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._matches.horizontalHeader().setStretchLastSection(True)
        card.add(self._matches)
        return card

    # --- section 6: readiness ---
    def _readiness_card(self) -> Card:
        card = Card("Dataset Readiness")
        self._readiness = QVBoxLayout()
        card.body.addLayout(self._readiness)
        return card

    # =================== lifecycle ===================
    def on_show(self) -> None:
        cur = self._dataset.currentData()
        self._dataset.blockSignals(True)
        self._dataset.clear()
        for d in self._controller.list_datasets():
            self._dataset.addItem(f"{d.name}  ({d.image_count} imgs)", d.project_id)
        idx = self._dataset.findData(cur)
        if idx >= 0:
            self._dataset.setCurrentIndex(idx)
        self._dataset.blockSignals(False)

    # =================== analyze (threaded) ===================
    def _analyze(self) -> None:
        pid = self._dataset.currentData()
        if pid is None:
            self._notify.warning("Import a dataset first.")
            return
        self.busy.emit(True)
        self._notify.info("Running the AI Dataset Analyst…")
        created = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._threads.submit(
            self._controller.analyze_dataset, pid, created,
            on_finished=self._on_analyzed, on_error=self._on_error)

    def _on_analyzed(self, intel) -> None:
        self.busy.emit(False)
        if intel is None:
            self._notify.warning("No pipeline run is cached for this dataset — run the "
                                 "Annotation Pipeline first, then analyze.")
            return
        self._intel = intel
        self._render_summary()
        self._render_health()
        self._render_issues()
        self._render_recommendations()
        self._render_historical()
        self._render_readiness()
        self._notify.success(f"Analysis complete ({intel.summary.source}). "
                             f"Recommendation: {intel.summary.overall_recommendation}.")

    def _on_error(self, message: str) -> None:
        self.busy.emit(False)
        self._notify.error(f"Analysis failed — {message}")

    # =================== rendering ===================
    def _render_summary(self) -> None:
        s = self._intel.summary
        self._gauge.set_value(s.overall_health)
        rows = [
            ("Dataset Name", s.dataset), ("Dataset Version", str(s.version)),
            ("Dataset Size", f"{s.size_mb} MB"), ("Image Count", str(s.image_count)),
            ("Overall Dataset Health", f"{s.overall_health}/100"),
            ("Annotation Quality", f"{s.annotation_quality:.0%}"),
            ("Verification Confidence", f"{s.verification_confidence:.0%}"),
            ("Production Readiness", s.production_readiness),
            ("Historical Improvement", s.historical_improvement),
            (f"Analyst ({s.source})", s.analyst_summary),
        ]
        self._fill_grid(self._summary_grid, rows)
        self._rec_badge.setText(f"Overall Recommendation:  {s.overall_recommendation}")

    def _render_health(self) -> None:
        self._clear(self._health)
        for k in self._intel.kpis:
            line = QWidget()
            h = QHBoxLayout(line)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(label(k.name, "Muted"), 2)
            if k.score is None:
                h.addWidget(QLabel(k.value), 3)
            else:
                bar = QProgressBar()
                bar.setRange(0, 100)
                bar.setValue(k.score)
                bar.setFormat(k.value)
                bar.setMaximumHeight(16)
                h.addWidget(bar, 3)
            self._health.addWidget(line)

    def _render_issues(self) -> None:
        self._clear(self._issues)
        if not self._intel.issues:
            self._issues.addWidget(label("No dominant issues detected in the measured metrics.", "Muted"))
            return
        for i in self._intel.issues:
            card = Card(i.title)
            card.add(label(i.description, wrap=True))
            card.add(label("Evidence: " + "; ".join(i.evidence), "Muted", wrap=True))
            card.add(label(f"Impact: {i.impact}", "Muted", wrap=True))
            card.add(label(f"Recommendation: {i.recommendation}", wrap=True))
            card.add(label(f"Expected improvement: {i.expected_improvement}  ·  "
                           f"confidence {i.confidence}", "Muted"))
            self._issues.addWidget(card)

    def _render_recommendations(self) -> None:
        self._clear(self._recommendations)
        if self._intel is None:
            return
        pri = self._f_priority.currentText()
        min_conf = self._f_conf.value()
        query = self._f_search.text().lower()
        shown = 0
        for r in self._intel.recommendations:
            if pri != "All" and r.priority != pri:
                continue
            if r.confidence < min_conf:
                continue
            if query and query not in (r.recommendation + r.problem).lower():
                continue
            shown += 1
            card = Card()
            head = QWidget()
            hh = QHBoxLayout(head)
            hh.setContentsMargins(0, 0, 0, 0)
            badge = label(r.priority, "Badge")
            badge.setStyleSheet(f"color: {_PRIORITY_COLOR.get(r.priority, '#8b9096')}")
            hh.addWidget(badge)
            hh.addWidget(label(r.recommendation, "H2"), 1)
            card.body.addWidget(head)
            card.add(label(f"Problem: {r.problem}", "Muted", wrap=True))
            card.add(label(f"Expected Gain: {r.expected_gain}", wrap=True))
            grid = QGridLayout()
            for c, (k, v) in enumerate([("Estimated Effort", r.estimated_effort),
                                        ("Expected Review Reduction", r.expected_review_reduction),
                                        ("Expected Runtime Impact", r.expected_runtime_impact)]):
                grid.addWidget(label(k, "Muted"), 0, c)
                grid.addWidget(QLabel(v), 1, c)
            card.body.addLayout(grid)
            card.add(label(f"Engineering Rationale: {r.rationale}  ·  confidence {r.confidence}",
                           "Muted", wrap=True))
            self._recommendations.addWidget(card)
        if shown == 0:
            self._recommendations.addWidget(label("No recommendations match the filters.", "Muted"))

    def _render_historical(self) -> None:
        self._clear(self._trends)
        h = self._intel.historical
        self._historical_note.setText(h.note)
        self._matches.setRowCount(len(h.matches))
        for r, m in enumerate(h.matches):
            for c, val in enumerate([m["dataset"], str(m["similarity"]), m["review_rate"],
                                     str(m["quality"])]):
                self._matches.setItem(r, c, QTableWidgetItem(str(val)))
        if not h.available:
            return
        for t in h.trends:
            if len(t.series) < 2:
                self._trends.addWidget(label(
                    f"{t.metric}: {t.last} (need ≥2 runs for a trend)", "Muted"))
                continue
            spark = Sparkline(f"{t.metric} {'↑' if t.improved else '↓'} {t.delta:+}",
                              color="#4caf82" if t.improved else "#e0605e")
            spark.set_max(max(t.series) or 1.0)
            for v in t.series:
                spark.push(v)
            self._trends.addWidget(spark)

    def _render_readiness(self) -> None:
        self._clear(self._readiness)
        for c in self._intel.readiness:
            line = QWidget()
            h = QHBoxLayout(line)
            h.setContentsMargins(0, 0, 0, 0)
            mark = label("✓" if c.met else "✗")
            mark.setStyleSheet(f"color: {'#4caf82' if c.met else '#8b9096'}; font-weight: 600")
            h.addWidget(mark)
            h.addWidget(label(c.name), 1)
            v = QLabel(c.reasoning)
            v.setWordWrap(True)
            h.addWidget(v, 3)
            self._readiness.addWidget(line)

    # =================== export ===================
    def _export(self, fmt: str) -> None:
        if self._intel is None:
            self._notify.warning("Analyze a dataset before exporting.")
            return
        section = _EXPORT_SECTIONS[self._export_section.currentText()]
        markdown = self._controller.intelligence_markdown(self._intel, section)
        default = f"intelligence_{section}.{fmt}"
        flt = "Markdown (*.md)" if fmt == "md" else "PDF (*.pdf)"
        path, _ = QFileDialog.getSaveFileName(self, "Export intelligence", default, flt)
        if not path:
            return
        if fmt == "md":
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(markdown)
        else:
            _write_pdf(markdown, path)
        self._notify.success(f"Exported: {path}")

    # =================== helpers ===================
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

    @staticmethod
    def _clear(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def context(self) -> tuple[str, list[tuple[str, str]]]:
        if self._intel is None:
            return ("Intelligence", [("Status", "Not analyzed")])
        s = self._intel.summary
        return ("Intelligence", [
            ("Health", f"{s.overall_health}/100"),
            ("Recommendation", s.overall_recommendation),
            ("Recommendations", str(len(self._intel.recommendations))),
            ("Issues", str(len(self._intel.issues))),
        ])


def _write_pdf(markdown: str, path: str) -> None:
    from PySide6.QtGui import QPageSize, QPdfWriter, QTextDocument

    writer = QPdfWriter(path)
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    doc = QTextDocument()
    doc.setMarkdown(markdown)
    doc.print_(writer)
