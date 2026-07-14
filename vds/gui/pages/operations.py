"""Operations & Performance Center (Phase 17) — the engineering operations dashboard.

Seven sections: executive KPI overview, system performance, benchmark explorer,
performance comparison, historical trends, platform health, and report export. It is
an operations dashboard, not an AI workspace: every number is measured execution data
read through BackendController (historical benchmark runs from Engineering Memory,
current totals from the store, live CPU/RAM/disk from psutil). Unmeasured metrics show
**Unavailable** — never estimated.
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
from vds.gui.widgets.sparkline import Sparkline

_STATUS_COLOR = {"ok": "#4caf82", "warn": "#e0a458", "crit": "#e0605e", "na": "#8b9096"}
_PLATFORM_COLOR = {"Healthy": "#4caf82", "Warning": "#e0a458", "Critical": "#e0605e",
                   "Unknown": "#8b9096"}
_BENCH_COLS = ["Run ID", "Dataset", "Detector", "Seg", "Planner Strategy", "Runtime",
               "Img/s", "Review", "Verification", "Export", "Peak RAM", "Peak GPU"]
_EXPORT_SECTIONS = {
    "Operations Report": "operations", "Benchmark Report": "benchmark",
    "Historical Trends": "trends", "Performance Summary": "performance_summary",
    "Full Report": "full",
}


class OperationsPage(Page):
    name = "Benchmark Center"

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
        self._live: dict = {"running_jobs": 0}
        self._benchmarks: list = []

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

        self._root.addWidget(label("Operations & Performance Center", "H1"))
        self._root.addWidget(label("Platform health, benchmark performance, and system "
                                   "utilization — from measured execution data.", "Muted"))
        self._root.addLayout(self._top_bar())

        self._root.addWidget(self._overview_card())
        grid = QGridLayout()
        grid.setSpacing(12)
        grid.addWidget(self._system_card(), 0, 0)
        grid.addWidget(self._health_card(), 0, 1)
        holder = QWidget()
        holder.setLayout(grid)
        self._root.addWidget(holder)

        self._root.addLayout(self._filter_bar())
        self._root.addWidget(self._benchmark_card())
        self._root.addWidget(self._comparison_card())
        self._root.addWidget(self._trends_card())
        self._root.addStretch(1)

    # --- bars ---
    def _top_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.setObjectName("Primary")
        refresh.clicked.connect(self._refresh)
        bar.addWidget(refresh)
        bar.addStretch(1)
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
        self._f_detector = QComboBox()
        self._f_seg = QComboBox()
        self._f_seg.addItems(["All", "On", "Off"])
        self._f_date = QLineEdit()
        self._f_date.setPlaceholderText("Date…")
        self._f_date.setMaximumWidth(110)
        self._f_search = QLineEdit()
        self._f_search.setPlaceholderText("Search run / strategy…")
        for combo in (self._f_dataset, self._f_detector, self._f_seg):
            combo.currentTextChanged.connect(lambda _t: self._render_benchmarks())
        self._f_date.textChanged.connect(lambda _t: self._render_benchmarks())
        self._f_search.textChanged.connect(lambda _t: self._render_benchmarks())
        for name, w in [("Dataset", self._f_dataset), ("Detector", self._f_detector),
                        ("Segmentation", self._f_seg), ("Date", self._f_date)]:
            bar.addWidget(label(name, "Muted"))
            bar.addWidget(w)
        bar.addWidget(self._f_search, 1)
        return bar

    # --- section 1: executive overview ---
    def _overview_card(self) -> Card:
        card = Card("Executive Operations Overview")
        self._overview = QGridLayout()
        self._overview.setSpacing(10)
        card.body.addLayout(self._overview)
        return card

    # --- section 2: system performance ---
    def _system_card(self) -> Card:
        card = Card("System Performance")
        self._system = QVBoxLayout()
        card.body.addLayout(self._system)
        return card

    # --- section 6: platform health ---
    def _health_card(self) -> Card:
        card = Card("Platform Health")
        self._health_status = label("—", "H1")
        card.add(self._health_status)
        self._health = QVBoxLayout()
        card.body.addLayout(self._health)
        return card

    # --- section 3: benchmark explorer ---
    def _benchmark_card(self) -> Card:
        card = Card("Benchmark Explorer")
        self._benchmark = QTableWidget(0, len(_BENCH_COLS))
        self._benchmark.setHorizontalHeaderLabels(_BENCH_COLS)
        self._benchmark.verticalHeader().setVisible(False)
        self._benchmark.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._benchmark.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._benchmark.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._benchmark.setSortingEnabled(True)
        self._benchmark.horizontalHeader().setStretchLastSection(True)
        card.add(self._benchmark)
        card.add(label("Select multiple runs, then Compare Selected.", "Muted"))
        return card

    # --- section 4: performance comparison ---
    def _comparison_card(self) -> Card:
        card = Card("Performance Comparison")
        btn = QPushButton("Compare Selected")
        btn.clicked.connect(self._compare)
        card.add(btn)
        self._comparison = QTableWidget(0, 0)
        self._comparison.verticalHeader().setVisible(False)
        self._comparison.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        card.add(self._comparison)
        return card

    # --- section 5: historical trends ---
    def _trends_card(self) -> Card:
        card = Card("Historical Trends")
        self._trends = QGridLayout()
        self._trends.setSpacing(10)
        card.body.addLayout(self._trends)
        return card

    # =================== lifecycle ===================
    def on_show(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        running = max(0, self._threads.active)
        # Load synchronously on the UI thread, like the Dashboard. The work is light
        # (a psutil snapshot plus a few aggregate reads); running it on a QThreadPool
        # worker instead raced Qt's renderer on Windows and segfaulted. psutil in
        # particular must be sampled on the UI thread.
        live = self._controller.ops_snapshot(running)
        self.busy.emit(True)
        try:
            self._on_loaded(self._load(live))
        except Exception as exc:  # keep the page alive on any backend hiccup
            self._on_error(f"{type(exc).__name__}: {exc}")

    def _load(self, live: dict):
        c = self._controller
        return {
            "live": live, "overview": c.ops_overview(live), "system": c.ops_system(live),
            "benchmarks": c.ops_benchmarks(), "trends": c.ops_trends(),
            "health": c.ops_health(live), "options": c.ops_filter_options(),
        }

    def _on_loaded(self, data) -> None:
        self.busy.emit(False)
        self._live = data["live"]
        self._benchmarks = data["benchmarks"]
        self._populate_filters(data["options"])
        self._render_overview(data["overview"])
        self._render_system(data["system"])
        self._render_health(data["health"])
        self._render_benchmarks()
        self._render_trends(data["trends"])

    def _on_error(self, message: str) -> None:
        self.busy.emit(False)
        self._notify.error(f"Operations refresh failed — {message}")

    def _populate_filters(self, opts) -> None:
        for combo, items in [(self._f_dataset, ["All datasets"] + opts.datasets),
                             (self._f_detector, ["All"] + opts.detectors)]:
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            idx = combo.findText(cur)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

    # =================== section 1 ===================
    def _render_overview(self, kpis) -> None:
        self._clear_grid(self._overview)
        for i, k in enumerate(kpis):
            tile = Card()
            value = label(k.value, "Metric")
            value.setAlignment(Qt.AlignmentFlag.AlignLeft)
            tile.add(value)
            tile.add(label(k.label, "Muted"))
            if k.sub:
                tile.add(label(k.sub, "Badge"))
            self._overview.addWidget(tile, i // 4, i % 4)

    # =================== section 2 ===================
    def _render_system(self, stats) -> None:
        self._clear(self._system)
        for s in stats:
            line = QWidget()
            h = QHBoxLayout(line)
            h.setContentsMargins(0, 0, 0, 0)
            dot = label("●")
            dot.setStyleSheet(f"color: {_STATUS_COLOR.get(s.status, '#8b9096')}")
            h.addWidget(dot)
            h.addWidget(label(s.name, "Muted"), 2)
            val = QLabel(s.value)
            val.setWordWrap(True)
            h.addWidget(val, 3)
            self._system.addWidget(line)

    # =================== section 6 ===================
    def _render_health(self, health) -> None:
        self._clear(self._health)
        self._health_status.setText(health.status)
        self._health_status.setStyleSheet(
            f"color: {_PLATFORM_COLOR.get(health.status, '#8b9096')}")
        for ind in health.indicators:
            line = QWidget()
            h = QHBoxLayout(line)
            h.setContentsMargins(0, 0, 0, 0)
            dot = label("●")
            dot.setStyleSheet(f"color: {_STATUS_COLOR.get(ind.status, '#8b9096')}")
            h.addWidget(dot)
            h.addWidget(label(ind.name, "Muted"), 2)
            h.addWidget(QLabel(ind.detail), 3)
            self._health.addWidget(line)
        if health.root_causes:
            self._health.addWidget(label("Root Causes", "Muted"))
            for rc in health.root_causes:
                self._health.addWidget(label(f"• {rc}", wrap=True))

    # =================== section 3 ===================
    def _filtered(self) -> list:
        ds = self._f_dataset.currentText()
        det = self._f_detector.currentText()
        seg = self._f_seg.currentText()
        date = self._f_date.text().strip()
        q = self._f_search.text().lower()
        out = []
        for b in self._benchmarks:
            if ds not in ("All datasets", "") and b.dataset != ds:
                continue
            if det not in ("All", "") and b.detector != det:
                continue
            if seg != "All" and b.segmentation != seg:
                continue
            if date and date not in b.created_at:
                continue
            if q and q not in (b.run_id + b.strategy + b.dataset).lower():
                continue
            out.append(b)
        return out

    def _render_benchmarks(self) -> None:
        rows = self._filtered()
        self._benchmark.setSortingEnabled(False)
        self._benchmark.setRowCount(len(rows))
        for i, b in enumerate(rows):
            cells = [b.run_id, b.dataset, b.detector, b.segmentation, b.strategy, b.runtime,
                     b.ips, b.review_rate, b.verification, b.export_success, b.peak_ram, b.peak_gpu]
            for c, val in enumerate(cells):
                item = QTableWidgetItem(val)
                item.setData(Qt.ItemDataRole.UserRole, b.run_id)
                self._benchmark.setItem(i, c, item)
        self._benchmark.setSortingEnabled(True)
        self._benchmark.resizeColumnsToContents()

    def _selected_ids(self) -> list[str]:
        ids, seen = [], set()
        for item in self._benchmark.selectedItems():
            rid = item.data(Qt.ItemDataRole.UserRole)
            if rid and rid not in seen:
                seen.add(rid)
                ids.append(rid)
        return ids

    # =================== section 4 ===================
    def _compare(self) -> None:
        ids = self._selected_ids()
        if not ids:
            self._notify.warning("Select one or more benchmark runs first.")
            return
        cmp = self._controller.ops_compare(ids)
        self._comparison.clear()
        self._comparison.setColumnCount(len(cmp.runs) + 1)
        self._comparison.setHorizontalHeaderLabels(["Metric"] + cmp.runs)
        self._comparison.setRowCount(len(cmp.rows))
        for i, row in enumerate(cmp.rows):
            metric = row.metric + (f"  ({row.trend})" if row.trend else "")
            self._comparison.setItem(i, 0, QTableWidgetItem(metric))
            for c, val in enumerate(row.values, start=1):
                self._comparison.setItem(i, c, QTableWidgetItem(val))
        self._comparison.resizeColumnsToContents()
        self._notify.success(f"Comparing {len(cmp.runs)} run(s).")

    # =================== section 5 ===================
    def _render_trends(self, trends) -> None:
        self._clear_grid(self._trends)
        if not trends:
            self._trends.addWidget(label("No benchmark runs recorded yet.", "Muted"), 0, 0)
            return
        for i, t in enumerate(trends):
            spark = Sparkline(f"{t.metric}: {t.last} (Δ{t.delta:+})",
                              color="#4caf82" if t.improved else "#e0605e")
            spark.set_max(max(t.series) or 1.0)
            for v in t.series:
                spark.push(v)
            self._trends.addWidget(spark, i // 3, i % 3)

    # =================== export ===================
    def _export(self, fmt: str) -> None:
        section = _EXPORT_SECTIONS[self._export_section.currentText()]
        markdown = self._controller.ops_markdown(section, self._live)
        default = f"operations_{section}.{fmt}"
        flt = "Markdown (*.md)" if fmt == "md" else "PDF (*.pdf)"
        path, _ = QFileDialog.getSaveFileName(self, "Export operations report", default, flt)
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

    def _clear_grid(self, layout) -> None:
        self._clear(layout)

    def context(self) -> tuple[str, list[tuple[str, str]]]:
        return ("Operations", [
            ("Benchmark runs", str(len(self._benchmarks))),
            ("CPU", f"{self._live.get('cpu')}%" if self._live.get("cpu") is not None else "n/a"),
            ("Selected", str(len(self._selected_ids()))),
        ])
