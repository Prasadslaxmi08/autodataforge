"""StatusBar (Phase 11) — transient messages + permanent indicators.

Left: the latest notification (color-coded, auto-clears). Right: running-task
count, a live CPU/MEM summary, and the active theme. It is a display only.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QStatusBar

_COLORS = {"info": "#8b9096", "success": "#4caf82", "warning": "#e0a458", "error": "#e0605e"}


class StatusBar(QStatusBar):
    def __init__(self) -> None:
        super().__init__()
        self.setSizeGripEnabled(False)
        self._tasks = QLabel("Tasks: 0")
        self._res = QLabel("CPU — · MEM —")
        self._theme = QLabel("dark")
        for w in (self._tasks, self._res, self._theme):
            w.setObjectName("Muted")
            self.addPermanentWidget(w)

    def set_message(self, level: str, message: str) -> None:
        color = _COLORS.get(level, _COLORS["info"])
        self.showMessage(f"  {message}", 6000)
        self.setStyleSheet(f"QStatusBar {{ color: {color}; }}")

    def set_tasks(self, count: int) -> None:
        self._tasks.setText(f"Tasks: {count}")

    def set_resources(self, cpu: float, ram_pct: float) -> None:
        self._res.setText(f"CPU {cpu:.0f}% · MEM {ram_pct:.0f}%")

    def set_theme(self, name: str) -> None:
        self._theme.setText(name)
