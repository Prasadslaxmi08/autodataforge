"""HealthGauge — a 0..100 arc gauge drawn with QPainter (Phase 15).

A clean engineering gauge for overall dataset health; colour shifts red→amber→green
with the value. No animation, no chart dependency.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


def _color(value: int) -> QColor:
    if value >= 75:
        return QColor("#4caf82")
    if value >= 50:
        return QColor("#e0a458")
    return QColor("#e0605e")


class HealthGauge(QWidget):
    def __init__(self, caption: str = "Overall Health", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value = 0
        self._caption = caption
        self.setMinimumSize(160, 130)

    def set_value(self, value: int) -> None:
        self._value = max(0, min(100, int(value)))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height() - 20)
        rect = QRectF((self.width() - side) / 2 + 10, 8, side - 20, side - 20)
        # track
        p.setPen(QPen(QColor(0, 0, 0, 40), 12, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawArc(rect, 225 * 16, -270 * 16)
        # value arc (270° sweep from 225°)
        p.setPen(QPen(_color(self._value), 12, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawArc(rect, 225 * 16, int(-270 * 16 * self._value / 100))
        # number
        p.setPen(_color(self._value))
        f = QFont()
        f.setPointSize(22)
        f.setBold(True)
        p.setFont(f)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(self._value))
        # caption
        p.setPen(QColor("#8b9096"))
        p.setFont(QFont())
        p.drawText(0, self.height() - 6, self.width(), 16,
                   Qt.AlignmentFlag.AlignHCenter, self._caption)
