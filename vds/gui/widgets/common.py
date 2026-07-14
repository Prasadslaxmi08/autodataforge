"""Small reusable widgets (Phase 11) — cards and metric tiles used across pages."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


def label(text: str, object_name: str = "", *, wrap: bool = False) -> QLabel:
    lbl = QLabel(text)
    if object_name:
        lbl.setObjectName(object_name)
    lbl.setWordWrap(wrap)
    return lbl


class Card(QFrame):
    """A titled surface panel. `body` layout is exposed for callers to fill."""

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._v = QVBoxLayout(self)
        self._v.setContentsMargins(14, 12, 14, 12)
        self._v.setSpacing(8)
        if title:
            self._v.addWidget(label(title, "H2"))

    @property
    def body(self) -> QVBoxLayout:
        return self._v

    def add(self, widget: QWidget) -> None:
        self._v.addWidget(widget)


class MetricTile(Card):
    """A big number with a caption — the dashboard KPI unit."""

    def __init__(self, caption: str, value: str = "—", parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._value = label(value, "Metric")
        self._value.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.add(self._value)
        self.add(label(caption, "Muted"))

    def set_value(self, value: str) -> None:
        self._value.setText(value)
