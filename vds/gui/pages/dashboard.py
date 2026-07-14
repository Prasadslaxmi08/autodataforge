"""Dashboard (Phase 11) — the functional landing page.

Reads a single snapshot from the BackendController and renders it: KPI tiles plus
cards for recent projects/datasets, latest benchmarks, environment (model /
provider / GPU), recent engineering memory, and pipeline status. Refreshes every
time it is shown, so it always reflects the live backend.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from vds.gui.controller import BackendController
from vds.gui.pages.base import Page
from vds.gui.widgets.common import Card, MetricTile, label


class DashboardPage(Page):
    name = "Dashboard"

    def __init__(self, controller: BackendController) -> None:
        super().__init__()
        self._controller = controller

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll)

        self._content = QWidget()
        self._root = QVBoxLayout(self._content)
        self._root.setContentsMargins(20, 18, 20, 18)
        self._root.setSpacing(14)
        scroll.setWidget(self._content)

        self._root.addWidget(label("Dashboard", "H1"))
        self._root.addWidget(label("Platform overview — AutoDataForge", "Muted"))

        # KPI tiles
        self._tiles = {
            "datasets": MetricTile("Datasets"),
            "memory": MetricTile("Memory Records"),
            "benchmarks": MetricTile("Benchmark Files"),
            "status": MetricTile("Pipeline"),
        }
        row = QHBoxLayout()
        for tile in self._tiles.values():
            row.addWidget(tile)
        self._root.addLayout(row)

        # dynamic cards grid
        self._grid = QGridLayout()
        self._grid.setSpacing(12)
        grid_holder = QWidget()
        grid_holder.setLayout(self._grid)
        self._root.addWidget(grid_holder)
        self._root.addStretch(1)

    def on_show(self) -> None:
        snap = self._controller.dashboard_snapshot()
        self._tiles["datasets"].set_value(str(snap["dataset_count"]))
        self._tiles["memory"].set_value(str(snap["memory_count"]))
        self._tiles["benchmarks"].set_value(str(len(snap["latest_benchmarks"])))
        self._tiles["status"].set_value(snap["pipeline_status"].title())
        self._rebuild_cards(snap)

    def _rebuild_cards(self, snap: dict) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        cards = [
            self._list_card("Recent Projects",
                            [f"{name}  ·  {phase}" for _id, name, phase in snap["recent_projects"]]
                            or ["No projects yet — import a dataset to begin."]),
            self._kv_card("Environment", [
                ("Current Model", snap["current_model"]),
                ("Current Provider", snap["current_provider"]),
                ("GPU Device", snap["gpu_device"]),
                ("VRAM Budget", f"{snap['vram_budget_mb']} MB"),
            ]),
            self._list_card("Latest Benchmarks",
                            [f"{n}  ·  {size}" for n, size in snap["latest_benchmarks"]]
                            or ["No benchmarks recorded yet."]),
            self._list_card("Recent Engineering Memory",
                            [f"{mid}  ·  quality {q}  ·  review {r}"
                             for mid, _ts, q, r in snap["recent_memory"]]
                            or ["No engineering memory recorded yet."]),
            self._kv_card("Reports & Jobs", [
                ("Recent Analyst Report", snap["recent_analyst_report"] or "none"),
                ("Recent Planner Recommendation", "Planner page (future phase)"),
                ("Recent Processing Jobs", "none"),
                ("Pipeline Status", snap["pipeline_status"]),
            ]),
        ]
        for i, card in enumerate(cards):
            self._grid.addWidget(card, i // 2, i % 2)

    @staticmethod
    def _list_card(title: str, lines: list[str]) -> Card:
        card = Card(title)
        for ln in lines[:8]:
            card.add(label(ln, "Muted", wrap=True))
        return card

    @staticmethod
    def _kv_card(title: str, rows: list[tuple[str, str]]) -> Card:
        card = Card(title)
        for k, v in rows:
            card.add(label(k, "Muted"))
            card.add(label(v, wrap=True))
        return card

    def context(self) -> tuple[str, list[tuple[str, str]]]:
        snap = self._controller.dashboard_snapshot()
        return ("Overview", [
            ("Datasets", str(snap["dataset_count"])),
            ("Memory records", str(snap["memory_count"])),
            ("Model", snap["current_model"]),
            ("Provider", snap["current_provider"]),
        ])
