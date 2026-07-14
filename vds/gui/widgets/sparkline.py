"""Sparkline — a tiny live line chart drawn with QPainter (Phase 13).

PySide6-Essentials ships no QtCharts, and a full chart library would be overkill
for a rolling CPU/RAM trace. This is a fixed-capacity ring of recent values drawn
as a smooth polyline — cheap enough to update every tick without touching pipeline
execution.
"""

from __future__ import annotations

from collections import deque

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget


class Sparkline(QWidget):
    def __init__(self, caption: str = "", capacity: int = 60,
                 color: str = "#3d7eff", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._values: deque[float] = deque(maxlen=capacity)
        self._caption = caption
        self._color = QColor(color)
        self._max = 100.0  # values are percentages by default
        self.setMinimumHeight(52)

    def set_max(self, value: float) -> None:
        self._max = max(1.0, value)

    def push(self, value: float) -> None:
        self._values.append(value)
        self.update()

    def clear(self) -> None:
        self._values.clear()
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(0, 0, 0, 24))
        if self._caption:
            p.setPen(QColor("#8b9096"))
            p.drawText(6, 14, f"{self._caption}"
                       + (f"  {self._values[-1]:.0f}" if self._values else ""))
        if len(self._values) < 2:
            return
        pad = 6
        n = len(self._values)
        step = (w - 2 * pad) / max(1, n - 1)
        poly = QPolygonF()
        for i, v in enumerate(self._values):
            y = h - pad - (min(v, self._max) / self._max) * (h - pad - 18)
            poly.append(QPointF(pad + i * step, y))
        p.setPen(QPen(self._color, 1.6, Qt.PenStyle.SolidLine))
        p.drawPolyline(poly)
