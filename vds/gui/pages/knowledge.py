"""Knowledge Center (Phase 16) — the searchable engineering knowledge base.

Six sections: knowledge search, dataset history, reusable knowledge cards, an
engineering-evolution timeline, historical comparison, and lessons learned. It
VISUALIZES the existing Engineering Memory through BackendController — every value is
a stored measured metric or a validated Analyst recommendation. When no matching
knowledge exists, it says so explicitly rather than inventing anything.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
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
from vds.gui.pages.intelligence import _write_pdf
from vds.gui.threads import ThreadManager
from vds.gui.widgets.common import Card, label

_HISTORY_COLS = ["Dataset", "Date", "Version", "Images", "Planner Strategy",
                 "Review Rate", "Runtime", "Health", "Status"]
_PRIORITIES = ["All", "High (≥75)", "Medium (50–74)", "Low (<50)"]
_EXPORT_SECTIONS = {
    "Knowledge Report": "knowledge_report", "Historical Comparison": "comparison",
    "Lessons Learned": "lessons", "Engineering Summary": "engineering_summary",
    "Full Knowledge Base": "full",
}


def _priority_ok(health: int, choice: str) -> bool:
    if choice.startswith("High"):
        return health >= 75
    if choice.startswith("Medium"):
        return 50 <= health < 75
    if choice.startswith("Low"):
        return health < 50
    return True


class KnowledgePage(Page):
    name = "Engineering Memory"

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
        self._records = []

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

        self._root.addWidget(label("Knowledge Center", "H1"))
        self._root.addWidget(label("Explore historical engineering knowledge — past datasets, "
                                   "strategies, recommendations, and lessons learned.", "Muted"))
        self._root.addLayout(self._search_bar())
        self._root.addLayout(self._filter_bar())

        self._root.addWidget(self._history_card())
        grid = QGridLayout()
        grid.setSpacing(12)
        grid.addWidget(self._cards_card(), 0, 0)
        grid.addWidget(self._timeline_card(), 0, 1)
        holder = QWidget()
        holder.setLayout(grid)
        self._root.addWidget(holder)
        self._root.addWidget(self._comparison_card())
        self._root.addWidget(self._lessons_card())
        self._root.addStretch(1)

    # --- bars ---
    def _search_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addWidget(label("Search", "Muted"))
        self._search_field = QComboBox()
        self._search_field.addItems(list(self._controller.knowledge_search_fields()))
        bar.addWidget(self._search_field)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search historical engineering records…")
        self._search.returnPressed.connect(self._do_search)
        self._search.textChanged.connect(lambda _t: self._render_history())
        bar.addWidget(self._search, 1)
        find = QPushButton("Search")
        find.setObjectName("Primary")
        find.clicked.connect(self._do_search)
        bar.addWidget(find)
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
        return bar

    def _filter_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self._f_dataset = QComboBox()
        self._f_version = QComboBox()
        self._f_scene = QComboBox()
        self._f_detector = QComboBox()
        self._f_priority = QComboBox()
        self._f_priority.addItems(_PRIORITIES)
        self._f_date = QLineEdit()
        self._f_date.setPlaceholderText("Date…")
        self._f_date.setMaximumWidth(120)
        for combo in (self._f_dataset, self._f_version, self._f_scene, self._f_detector,
                      self._f_priority):
            combo.currentTextChanged.connect(lambda _t: self._render_history())
        self._f_date.textChanged.connect(lambda _t: self._render_history())
        for name, w in [("Dataset", self._f_dataset), ("Version", self._f_version),
                        ("Scene", self._f_scene), ("Detector", self._f_detector),
                        ("Priority", self._f_priority), ("Date", self._f_date)]:
            bar.addWidget(label(name, "Muted"))
            bar.addWidget(w)
        bar.addStretch(1)
        return bar

    # --- section 2: dataset history (+ search results) ---
    def _history_card(self) -> Card:
        card = Card("Dataset History")
        self._history = QTableWidget(0, len(_HISTORY_COLS))
        self._history.setHorizontalHeaderLabels(_HISTORY_COLS)
        self._history.verticalHeader().setVisible(False)
        self._history.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._history.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._history.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._history.setSortingEnabled(True)
        self._history.horizontalHeader().setStretchLastSection(True)
        card.add(self._history)
        card.add(label("Select multiple rows, then Compare Selected.", "Muted"))
        return card

    # --- section 3: knowledge cards ---
    def _cards_card(self) -> Card:
        card = Card("Knowledge Cards")
        self._cards = QVBoxLayout()
        card.body.addLayout(self._cards)
        return card

    # --- section 4: timeline ---
    def _timeline_card(self) -> Card:
        card = Card("Engineering Timeline")
        self._timeline = QVBoxLayout()
        card.body.addLayout(self._timeline)
        return card

    # --- section 5: historical comparison ---
    def _comparison_card(self) -> Card:
        card = Card("Historical Comparison")
        btn = QPushButton("Compare Selected")
        btn.clicked.connect(self._compare)
        card.add(btn)
        self._comparison = QTableWidget(0, 0)
        self._comparison.verticalHeader().setVisible(False)
        self._comparison.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        card.add(self._comparison)
        return card

    # --- section 6: lessons learned ---
    def _lessons_card(self) -> Card:
        card = Card("Lessons Learned")
        self._lessons = QVBoxLayout()
        card.body.addLayout(self._lessons)
        return card

    # =================== lifecycle ===================
    def on_show(self) -> None:
        self.busy.emit(True)
        self._threads.submit(self._load, on_finished=self._on_loaded, on_error=self._on_error)

    def _load(self):
        c = self._controller
        return (c.knowledge_records(), c.knowledge_cards(), c.knowledge_timeline(),
                c.lessons_learned(), c.knowledge_filter_options())

    def _on_loaded(self, data) -> None:
        self.busy.emit(False)
        self._records, cards, timeline, lessons, opts = data
        self._populate_filters(opts)
        self._render_history()
        self._render_cards(cards)
        self._render_timeline(timeline)
        self._render_lessons(lessons)
        if not self._records:
            self._notify.info("Engineering Memory is empty — process datasets to build knowledge.")

    def _on_error(self, message: str) -> None:
        self.busy.emit(False)
        self._notify.error(f"Knowledge load failed — {message}")

    def _populate_filters(self, opts) -> None:
        combos = [(self._f_dataset, ["All datasets"] + opts.datasets),
                  (self._f_version, ["All"] + [str(v) for v in opts.versions]),
                  (self._f_scene, ["All"] + opts.scene_types),
                  (self._f_detector, ["All"] + opts.detectors)]
        for combo, items in combos:
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            idx = combo.findText(cur)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

    # =================== search + history ===================
    def _do_search(self) -> None:
        self._records = self._controller.search_knowledge(
            self._search.text(), self._search_field.currentText())
        self._render_history()
        self._notify.info(f"{len(self._records)} matching record(s).")

    def _filtered(self) -> list:
        ds = self._f_dataset.currentText()
        ver = self._f_version.currentText()
        scene = self._f_scene.currentText()
        det = self._f_detector.currentText()
        pri = self._f_priority.currentText()
        date = self._f_date.text().strip()
        out = []
        for r in self._records:
            if ds not in ("All datasets", "") and r.dataset != ds:
                continue
            if ver not in ("All", "") and str(r.version) != ver:
                continue
            if scene not in ("All", "") and r.scene_type != scene:
                continue
            if det not in ("All", "") and r.detector != det:
                continue
            if not _priority_ok(r.health, pri):
                continue
            if date and date not in r.created_at:
                continue
            out.append(r)
        return out

    def _render_history(self) -> None:
        rows = self._filtered()
        self._history.setSortingEnabled(False)
        self._history.setRowCount(len(rows))
        for i, r in enumerate(rows):
            cells = [r.dataset, r.created_at[:19], str(r.version), str(r.image_count),
                     r.planner_strategy, f"{r.review_rate:.0%}", f"{r.runtime_seconds:.1f}s",
                     f"{r.health}", r.status]
            for c, val in enumerate(cells):
                item = QTableWidgetItem(val)
                item.setData(Qt.ItemDataRole.UserRole, r.id)
                self._history.setItem(i, c, item)
        self._history.setSortingEnabled(True)
        self._history.resizeColumnsToContents()

    def _selected_ids(self) -> list[str]:
        ids, seen = [], set()
        for item in self._history.selectedItems():
            rid = item.data(Qt.ItemDataRole.UserRole)
            if rid and rid not in seen:
                seen.add(rid)
                ids.append(rid)
        return ids

    # =================== knowledge cards ===================
    def _render_cards(self, cards) -> None:
        self._clear(self._cards)
        if not cards:
            self._cards.addWidget(label("No reusable knowledge exists yet.", "Muted"))
            return
        for card in cards:
            box = Card(card.title)
            box.add(label(f"Occurrences: {card.occurrences}  ·  "
                          f"Success Rate: {card.success_rate:.0%}  ·  "
                          f"Confidence: {card.confidence}", "Muted"))
            box.add(label(f"Best Strategy: {card.best_strategy}", wrap=True))
            box.add(label(f"Expected Improvement: {card.expected_improvement}", wrap=True))
            box.add(label(f"Supporting Datasets: {', '.join(card.supporting_datasets)}",
                          "Muted", wrap=True))
            self._cards.addWidget(box)

    # =================== timeline ===================
    def _render_timeline(self, events) -> None:
        self._clear(self._timeline)
        if not events:
            self._timeline.addWidget(label("No engineering history yet.", "Muted"))
            return
        for e in events:
            line = QWidget()
            h = QHBoxLayout(line)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(label(e.date[:10], "Badge"))
            h.addWidget(label(e.kind, "H2"))
            detail = QLabel(e.detail)
            detail.setWordWrap(True)
            h.addWidget(detail, 1)
            self._timeline.addWidget(line)

    # =================== comparison ===================
    def _compare(self) -> None:
        ids = self._selected_ids()
        if not ids:
            self._notify.warning("Select one or more datasets in Dataset History first.")
            return
        cmp = self._controller.compare_knowledge(ids)
        self._comparison.clear()
        self._comparison.setColumnCount(len(cmp.datasets) + 1)
        self._comparison.setHorizontalHeaderLabels(["Metric"] + cmp.datasets)
        self._comparison.setRowCount(len(cmp.rows))
        for i, row in enumerate(cmp.rows):
            metric = row.metric + (f"  ({row.trend})" if row.trend else "")
            self._comparison.setItem(i, 0, QTableWidgetItem(metric))
            for c, val in enumerate(row.values, start=1):
                self._comparison.setItem(i, c, QTableWidgetItem(val))
        self._comparison.resizeColumnsToContents()
        self._notify.success(f"Comparing {len(cmp.datasets)} dataset(s).")

    # =================== lessons ===================
    def _render_lessons(self, lessons) -> None:
        self._clear(self._lessons)
        if not lessons:
            self._lessons.addWidget(label("No validated lessons recorded yet.", "Muted"))
            return
        for lsn in lessons:
            box = Card(lsn.solution)
            box.add(label(f"Problem: {lsn.problem}", wrap=True))
            box.add(label(f"Root Cause: {lsn.root_cause}", "Muted", wrap=True))
            box.add(label(f"Supporting Evidence: {', '.join(lsn.evidence) or 'none'}",
                          "Muted", wrap=True))
            box.add(label(f"Expected Benefit: {lsn.expected_benefit}", wrap=True))
            box.add(label(f"Occurrences: {lsn.occurrences}  ·  Confidence: {lsn.confidence}  ·  "
                          f"Datasets: {', '.join(lsn.reference_datasets)}", "Muted", wrap=True))
            self._lessons.addWidget(box)

    # =================== export ===================
    def _export(self, fmt: str) -> None:
        section = _EXPORT_SECTIONS[self._export_section.currentText()]
        ids = self._selected_ids() if section == "comparison" else None
        if section == "comparison" and not ids:
            self._notify.warning("Select datasets to compare before exporting.")
            return
        markdown = self._controller.knowledge_markdown(section, ids)
        default = f"knowledge_{section}.{fmt}"
        flt = "Markdown (*.md)" if fmt == "md" else "PDF (*.pdf)"
        path, _ = QFileDialog.getSaveFileName(self, "Export knowledge", default, flt)
        if not path:
            return
        if fmt == "md":
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(markdown)
        else:
            _write_pdf(markdown, path)
        self._notify.success(f"Exported: {path}")

    # =================== helpers ===================
    @staticmethod
    def _clear(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def context(self) -> tuple[str, list[tuple[str, str]]]:
        return ("Knowledge", [
            ("Records", str(len(self._records))),
            ("Selected", str(len(self._selected_ids()))),
        ])
