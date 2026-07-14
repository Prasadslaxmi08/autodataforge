"""ContextSidebar — the right panel (Phase 11).

Context-sensitive info for the active page: planner decisions, pipeline progress,
memory matches, analyst recommendations, benchmark metrics. Pages push content via
`set_context(title, rows)`; the sidebar owns no page logic.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from vds.gui.widgets.common import Card, label


class ContextSidebar(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedWidth(260)
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(8, 8, 8, 8)
        self._root.setSpacing(8)

        self._card = Card("Context")
        self._body = QVBoxLayout()
        self._card.body.addLayout(self._body)
        self._root.addWidget(self._card)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        self._root.addWidget(self._progress)
        self._root.addStretch(1)
        self.set_context("Context", [("Select a module", "")])

    def set_context(self, title: str, rows: list[tuple[str, str]]) -> None:
        while self._body.count():
            item = self._body.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for key, value in rows:
            self._body.addWidget(label(key, "Muted"))
            if value:
                v = QLabel(value)
                v.setWordWrap(True)
                self._body.addWidget(v)

    # --- progress (pipeline execution) ---
    def show_progress(self, visible: bool) -> None:
        self._progress.setVisible(visible)

    def set_progress(self, pct: int, _msg: str = "") -> None:
        self._progress.setVisible(True)
        self._progress.setValue(pct)
