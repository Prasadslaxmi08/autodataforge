"""ImagePreview — an image with optional detection/mask overlays (Phase 13).

Draws a CAS image scaled to a zoom factor, optionally overlaying bounding boxes
with labels + confidence. Used twice side by side (Original vs Annotated). Zoom and
Fit-to-Window are pure view operations — no backend involvement.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QScrollArea, QWidget

Box = tuple[float, float, float, float, str, float]  # x, y, w, h, label, confidence


class ImagePreview(QScrollArea):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label = QLabel("No image")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWidget(self._label)
        self._pixmap: QPixmap | None = None
        self._boxes: list[Box] = []
        self._zoom = 1.0
        self._show_labels = True
        self._show_conf = True

    def set_image(self, path: str, boxes: list[Box] | None = None,
                  show_labels: bool = True, show_conf: bool = True) -> None:
        self._show_labels = show_labels
        self._show_conf = show_conf
        self._pixmap = QPixmap(path)
        self._boxes = boxes or []
        if self._pixmap.isNull():
            self._label.setText("Preview unavailable")
            self._pixmap = None
            return
        self.fit()

    def clear_image(self) -> None:
        self._pixmap = None
        self._boxes = []
        self._label.setText("No image")

    def zoom_in(self) -> None:
        self._set_zoom(self._zoom * 1.25)

    def zoom_out(self) -> None:
        self._set_zoom(self._zoom / 1.25)

    def fit(self) -> None:
        if self._pixmap is None:
            return
        vw = max(1, self.viewport().width() - 4)
        scale = vw / self._pixmap.width()
        self._set_zoom(min(scale, 4.0))

    def _set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.1, min(zoom, 8.0))
        self._render()

    def _render(self) -> None:
        if self._pixmap is None:
            return
        w = int(self._pixmap.width() * self._zoom)
        h = int(self._pixmap.height() * self._zoom)
        scaled = self._pixmap.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
        if self._boxes:
            scaled = self._draw_boxes(scaled)
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())

    def _draw_boxes(self, pixmap: QPixmap) -> QPixmap:
        out = QPixmap(pixmap)
        p = QPainter(out)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont()
        font.setPointSize(8)
        p.setFont(font)
        for x, y, w, h, label, conf in self._boxes:
            p.setPen(QPen(QColor("#3d7eff"), 2))
            p.drawRect(int(x * self._zoom), int(y * self._zoom),
                       int(w * self._zoom), int(h * self._zoom))
            parts = []
            if self._show_labels:
                parts.append(label)
            if self._show_conf:
                parts.append(f"{conf:.2f}")
            if parts:
                p.setPen(QColor("#d7dae0"))
                p.drawText(int(x * self._zoom) + 2, int(y * self._zoom) + 12, " ".join(parts))
        p.end()
        return out
