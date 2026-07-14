"""BoxCanvas — the interactive annotation canvas (Phase 19).

A QGraphicsView over the image at 1:1 scene scale (scene units = image pixels), with
movable/resizable box items. QGraphicsView gives smooth zoom/pan and hit-testing for
free, so the editor doesn't reinvent any of it.

Interaction: drag on empty image = draw a new box; drag a box body = move; drag near an
edge/corner = resize; wheel = zoom to cursor; middle-mouse (or Space-drag) = pan.
Masks are drawn as a read-only translucent overlay.

The canvas is persistence-agnostic: it holds boxes as plain dicts
``{id, x, y, w, h, label}`` and reports edits via signals. The page owns undo/redo and
Save; the canvas only edits pixels on screen.
"""

from __future__ import annotations

import json

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

_HANDLE_PX = 8.0  # edge/corner grab tolerance, in view pixels
_MIN_SIZE = 3.0   # discard boxes smaller than this (accidental clicks)


def rle_to_qimage(rle: str, width: int, height: int, color: QColor) -> QImage:
    """Decode a COCO-style column-major RLE (JSON counts, leading zero-run) into a
    translucent QImage the same size as the image — matches builtin `_rle_encode`."""
    counts = json.loads(rle)
    flat = np.zeros(width * height, dtype=np.uint8)
    idx, val = 0, 0  # RLE starts counting zeros (background)
    for run in counts:
        if val:
            flat[idx:idx + run] = 1
        idx += run
        val ^= 1
    mask = flat.reshape((height, width), order="F")  # column-major, as encoded
    rgba = (140 << 24) | (color.red() << 16) | (color.green() << 8) | color.blue()
    # Build the ARGB buffer with numpy (fast) instead of per-pixel setPixel.
    argb = np.zeros((height, width), dtype=np.uint32)
    argb[mask.astype(bool)] = rgba
    return QImage(argb.tobytes(), width, height, QImage.Format.Format_ARGB32).copy()


class BoxItem(QGraphicsRectItem):
    """A selectable box that moves and resizes by direct manipulation."""

    def __init__(self, box_id: str, rect: QRectF, label: str, color: QColor,
                 confidence: float = 1.0, canvas: BoxCanvas | None = None) -> None:
        super().__init__(rect)
        self.box_id = box_id
        self.label = label
        self.confidence = confidence
        self.has_mask = False
        self.mask: dict | None = None  # pending re-segment result, saved with the box
        self._canvas = canvas
        self._color = QColor(color)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(10)
        self._edge = ""
        self._start_scene: QPointF | None = None
        self._start_rect: QRectF | None = None

    # --- appearance ---
    def paint(self, painter, option, widget=None) -> None:
        r = self.rect()
        pen = QPen(self._color, 2 if self.isSelected() else 1.5)
        pen.setCosmetic(True)  # constant width regardless of zoom
        painter.setPen(pen)
        painter.drawRect(r)
        painter.fillRect(QRectF(r.left(), r.top() - 14, max(24.0, len(self.label) * 7), 13),
                         self._color)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QPointF(r.left() + 2, r.top() - 3), self.label)
        if self.isSelected():
            painter.setBrush(self._color)
            t = self._tol()
            for cx, cy in self._corners(r):
                painter.drawRect(QRectF(cx - t / 2, cy - t / 2, t, t))

    def set_color(self, color: QColor) -> None:
        self._color = QColor(color)
        self.update()

    @staticmethod
    def _corners(r: QRectF):
        return [(r.left(), r.top()), (r.right(), r.top()),
                (r.left(), r.bottom()), (r.right(), r.bottom())]

    def _tol(self) -> float:
        views = self.scene().views() if self.scene() else []
        scale = views[0].transform().m11() if views else 1.0
        return _HANDLE_PX / max(scale, 0.01)

    def _edge_at(self, pos: QPointF) -> str:
        r, t = self.rect(), self._tol()
        within_x = r.left() - t <= pos.x() <= r.right() + t
        within_y = r.top() - t <= pos.y() <= r.bottom() + t
        e = ""
        if abs(pos.y() - r.top()) <= t and within_x:
            e += "t"
        if abs(pos.y() - r.bottom()) <= t and within_x:
            e += "b"
        if abs(pos.x() - r.left()) <= t and within_y:
            e += "l"
        if abs(pos.x() - r.right()) <= t and within_y:
            e += "r"
        return e

    # --- interaction ---
    def hoverMoveEvent(self, event) -> None:  # noqa: N802
        cursors = {"t": Qt.CursorShape.SizeVerCursor, "b": Qt.CursorShape.SizeVerCursor,
                   "l": Qt.CursorShape.SizeHorCursor, "r": Qt.CursorShape.SizeHorCursor,
                   "tl": Qt.CursorShape.SizeFDiagCursor, "br": Qt.CursorShape.SizeFDiagCursor,
                   "tr": Qt.CursorShape.SizeBDiagCursor, "bl": Qt.CursorShape.SizeBDiagCursor}
        self.setCursor(cursors.get(self._edge_at(event.pos()), Qt.CursorShape.SizeAllCursor))
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.setSelected(True)
        self._edge = self._edge_at(event.pos())
        self._start_scene = event.scenePos()
        self._start_rect = QRectF(self.rect())
        if self._canvas:
            self._canvas._begin_edit()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._start_scene is None:
            return
        d = event.scenePos() - self._start_scene
        r = QRectF(self._start_rect)
        if not self._edge:
            r.translate(d)
        else:
            if "l" in self._edge:
                r.setLeft(r.left() + d.x())
            if "r" in self._edge:
                r.setRight(r.right() + d.x())
            if "t" in self._edge:
                r.setTop(r.top() + d.y())
            if "b" in self._edge:
                r.setBottom(r.bottom() + d.y())
            r = r.normalized()
        self.prepareGeometryChange()
        self.setRect(r)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._start_scene = None
        if self._canvas:
            self._canvas._end_edit()
        event.accept()


class BoxCanvas(QGraphicsView):
    edit_committed = Signal()          # a create/move/resize actually changed something
    selection_changed = Signal(str)    # selected box id ("" = none)

    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(self.renderHints())
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._mask_item: QGraphicsPixmapItem | None = None
        self._pending: list[dict] | None = None  # undo snapshot captured at edit start
        self._color_for = lambda _label: QColor("#3d7eff")
        self._hidden: set[str] = set()  # labels hidden by the class filter
        self._filter = None  # confidence/review predicate
        self._draw_item: BoxItem | None = None
        self._draw_start: QPointF | None = None
        self._pan_start = None
        self._user_view = False  # True once the user zooms/pans — stop auto-fitting
        self._scene.selectionChanged.connect(self._emit_selection)

    def _emit_selection(self) -> None:
        # Guard the teardown race: a queued selectionChanged can fire after the
        # scene/view C++ objects are gone during GC.
        try:
            self.selection_changed.emit(self.selected_id())
        except (RuntimeError, TypeError):
            pass

    # --- content ---
    def load_image(self, path: str) -> None:
        self._scene.clear()
        self._pixmap_item = self._mask_item = None
        pix = QPixmap(path)
        self._pixmap_item = self._scene.addPixmap(pix)
        self._pixmap_item.setZValue(0)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self._user_view = False
        self.fit()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if not self._user_view:  # keep the image fitted until the user takes control
            self.fit()

    def set_color_resolver(self, fn) -> None:
        self._color_for = fn

    def set_boxes(self, boxes: list[dict]) -> None:
        for it in list(self._scene.items()):
            if isinstance(it, BoxItem):
                self._scene.removeItem(it)
        for b in boxes:
            self._add_item(b)

    def _add_item(self, b: dict) -> BoxItem:
        item = BoxItem(b.get("id", ""), QRectF(b["x"], b["y"], b["w"], b["h"]),
                       b.get("label", "object"), self._color_for(b.get("label", "object")),
                       b.get("confidence", 1.0), canvas=self)
        item.has_mask = b.get("has_mask", False)
        item.mask = b.get("mask")
        self._scene.addItem(item)
        self._apply_one(item)
        return item

    def boxes(self) -> list[dict]:
        out = []
        for it in self._scene.items():
            if isinstance(it, BoxItem):
                r = it.rect()
                d = {"id": it.box_id, "x": r.x(), "y": r.y(), "w": r.width(),
                     "h": r.height(), "label": it.label, "confidence": it.confidence,
                     "has_mask": it.has_mask}
                if it.mask is not None:
                    d["mask"] = it.mask
                out.append(d)
        return list(reversed(out))  # scene items() is top-first; keep insertion order

    # --- selection-driven ops (called by the page, which owns undo) ---
    def selected(self) -> BoxItem | None:
        try:
            items = self._scene.selectedItems()
        except RuntimeError:  # scene torn down (e.g. during teardown) — no selection
            return None
        for it in items:
            if isinstance(it, BoxItem):
                return it
        return None

    def selected_id(self) -> str:
        it = self.selected()
        return it.box_id if it else ""

    def add_box(self, b: dict) -> None:
        item = self._add_item(b)
        self._scene.clearSelection()
        item.setSelected(True)
        self.edit_committed.emit()

    def delete_selected(self) -> None:
        removed = False
        for it in self._scene.selectedItems():
            if isinstance(it, BoxItem):
                self._scene.removeItem(it)
                removed = True
        if removed:
            self.edit_committed.emit()

    def duplicate_selected(self) -> None:
        it = self.selected()
        if it is None:
            return
        r = it.rect()
        self.add_box({"id": "", "x": r.x() + 10, "y": r.y() + 10, "w": r.width(),
                      "h": r.height(), "label": it.label, "confidence": it.confidence})

    def relabel_selected(self, label: str) -> None:
        it = self.selected()
        if it is None:
            return
        it.label = label
        it.set_color(self._color_for(label))
        self.edit_committed.emit()

    def select_all(self) -> None:
        for it in self._scene.items():
            if isinstance(it, BoxItem) and it.isVisible():
                it.setSelected(True)

    # --- class filter + mask overlay ---
    def set_hidden_labels(self, hidden: set[str]) -> None:
        self._hidden = set(hidden)
        self._apply_visibility()

    def set_filter(self, predicate) -> None:
        """predicate(box_dict) -> bool; boxes failing it are hidden (with the class
        filter). None shows everything."""
        self._filter = predicate
        self._apply_visibility()

    def _apply_one(self, it: BoxItem) -> None:
        r = it.rect()
        box = {"id": it.box_id, "x": r.x(), "y": r.y(), "w": r.width(), "h": r.height(),
               "label": it.label, "confidence": it.confidence, "has_mask": it.has_mask}
        ok = it.label not in self._hidden and (self._filter is None or self._filter(box))
        it.setVisible(ok)

    def _apply_visibility(self) -> None:
        for it in self._scene.items():
            if isinstance(it, BoxItem):
                self._apply_one(it)

    def show_mask(self, image: QImage | None) -> None:
        if self._mask_item is not None:
            self._scene.removeItem(self._mask_item)
            self._mask_item = None
        if image is not None:
            self._mask_item = self._scene.addPixmap(QPixmap.fromImage(image))
            self._mask_item.setZValue(5)

    # --- undo bracketing (interactive edits) ---
    def _begin_edit(self) -> None:
        self._pending = self.boxes()

    def _end_edit(self) -> None:
        if self._pending is not None and self._pending != self.boxes():
            self._push_undo(self._pending)
            self.edit_committed.emit()
        self._pending = None

    def _push_undo(self, snapshot: list[dict]) -> None:
        # set by the page so interactive edits feed the same undo stack as button ops
        pass

    # --- view: zoom / pan ---
    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        self._user_view = True
        self.scale(factor, factor)

    def fit(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._user_view = False  # fitted is the auto-view; resizes keep fitting

    def reset_zoom(self) -> None:
        self.resetTransform()

    def center_selected(self) -> None:
        it = self.selected() or self._pixmap_item
        if it is not None:
            self.centerOn(it)

    def zoom(self, factor: float) -> None:
        self._user_view = True
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = event.position()
            self._user_view = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if isinstance(item, BoxItem):
                super().mousePressEvent(event)
                return
            # empty image → start drawing a new box
            self._draw_start = self.mapToScene(event.position().toPoint())
            self._begin_edit()
            self._draw_item = BoxItem("", QRectF(self._draw_start, self._draw_start),
                                      "object", self._color_for("object"), canvas=self)
            self._scene.addItem(self._draw_item)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._pan_start is not None:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))
            event.accept()
            return
        if self._draw_item is not None:
            cur = self.mapToScene(event.position().toPoint())
            self._draw_item.prepareGeometryChange()
            self._draw_item.setRect(QRectF(self._draw_start, cur).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if self._draw_item is not None:
            r = self._draw_item.rect().normalized()
            done, self._draw_item = self._draw_item, None
            if r.width() >= _MIN_SIZE and r.height() >= _MIN_SIZE:
                done.setRect(r)
                self._end_edit()  # commits the create as one undo step
                self._scene.clearSelection()
                done.setSelected(True)
            else:
                self._scene.removeItem(done)
                self._pending = None
            event.accept()
            return
        super().mouseReleaseEvent(event)
