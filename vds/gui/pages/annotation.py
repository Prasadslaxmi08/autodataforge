"""Annotation Workspace — the manual editor (Phase 19).

Five regions: a compact always-on toolbar, a Class Manager (left), the interactive
BoxCanvas (center), a Properties panel (right), and a thumbnail filmstrip (bottom),
with a confidence/review filter bar. Editing is in-memory with undo/redo; **Save**
diffs the session against what was loaded and commits via `controller.save_edits`
(which only uses the existing add/set_state — the backend is frozen).

Everything backend goes through BackendController; no page reimplements pipeline logic.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
)

from vds.gui.controller import BackendController
from vds.gui.notifications import NotificationSystem
from vds.gui.pages.base import Page
from vds.gui.threads import ThreadManager
from vds.gui.widgets.box_canvas import BoxCanvas, rle_to_qimage
from vds.gui.widgets.common import Card, label

_PALETTE = ["#3d7eff", "#4caf82", "#e0a458", "#e0605e", "#9b7ede", "#4bb6c9", "#d98cb3", "#8bbf5a"]
# (label, predicate over a box dict). None = show all.
_FILTERS: list[tuple[str, object]] = [
    ("All annotations", None),
    ("Confidence < 30%", lambda b: b["confidence"] < 0.30),
    ("Confidence < 50%", lambda b: b["confidence"] < 0.50),
    ("Needs review", lambda b: b.get("state") == "needs_review"),
    ("Missing masks", lambda b: not b.get("has_mask")),
    ("Duplicate objects", "dup"),  # handled specially (needs cross-box IoU)
]


def _color_for(lbl: str) -> str:
    return _PALETTE[sum(map(ord, lbl)) % len(_PALETTE)]


def _iou(a: dict, b: dict) -> float:
    ax2, ay2, bx2, by2 = a["x"] + a["w"], a["y"] + a["h"], b["x"] + b["w"], b["y"] + b["h"]
    ix, iy = max(a["x"], b["x"]), max(a["y"], b["y"])
    iw, ih = max(0.0, min(ax2, bx2) - ix), max(0.0, min(ay2, by2) - iy)
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


class AnnotationPage(Page):
    name = "Annotation"

    busy = Signal(bool)
    export_requested = Signal(str)  # project_id → shell opens the Export page

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
        self._pid: str | None = None
        self._images: list = []
        self._idx = 0
        self._original: list[dict] = []
        self._undo: list[list[dict]] = []
        self._redo: list[list[dict]] = []
        self._hidden: set[str] = set()
        self._thumb_cache: dict[str, QIcon] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)
        root.addLayout(self._toolbar())
        root.addLayout(self._filter_bar())

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._class_manager())
        self._canvas = BoxCanvas()
        self._canvas.set_color_resolver(lambda lbl: QColor(_color_for(lbl)))
        self._canvas._push_undo = self._record_undo  # interactive edits feed our stack
        self._canvas.edit_committed.connect(self._on_changed)
        self._canvas.selection_changed.connect(self._on_selection)
        split.addWidget(self._canvas)
        split.addWidget(self._properties())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([220, 780, 240])
        root.addWidget(split, 1)

        root.addWidget(self._filmstrip_widget())
        self._install_shortcuts()
        self._refresh_toolbar()

    # --- toolbar ---
    def _toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self._btn = {}
        specs = [
            ("undo", "↶ Undo", self.undo), ("redo", "↷ Redo", self.redo),
            ("save", "💾 Save", self.save), ("|", "", None),
            ("prev", "◀ Prev", self.prev_image), ("next", "Next ▶", self.next_image),
            ("|", "", None),
            ("zin", "🔍+", lambda: self._canvas.zoom(1.25)),
            ("zout", "🔍−", lambda: self._canvas.zoom(0.8)),
            ("fit", "Fit", self._canvas_fit), ("center", "Center", lambda: self._canvas.center_selected()),
            ("|", "", None),
            ("ai", "✨ AI Annotate", self.ai_annotate), ("export", "⇩ Export", self._export),
        ]
        for key, text, cb in specs:
            if key == "|":
                line = QLabel("│")
                line.setObjectName("Muted")
                bar.addWidget(line)
                continue
            b = QPushButton(text)
            if key in ("save", "ai"):
                b.setObjectName("Primary")
            b.clicked.connect(cb)
            self._btn[key] = b
            bar.addWidget(b)
        bar.addStretch(1)
        self._dirty_lbl = label("", "Muted")
        bar.addWidget(self._dirty_lbl)
        return bar

    def _filter_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addWidget(label("Show", "Muted"))
        self._filter = QComboBox()
        self._filter.addItems([name for name, _ in _FILTERS])
        self._filter.currentIndexChanged.connect(self._apply_filter)
        bar.addWidget(self._filter)
        bar.addStretch(1)
        self._img_lbl = label("", "Muted")
        bar.addWidget(self._img_lbl)
        return bar

    # --- class manager (left) ---
    def _class_manager(self) -> Card:
        card = Card("Classes")
        self._class_search = QComboBox()
        self._class_search.setEditable(True)
        self._class_search.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._class_search.lineEdit().setPlaceholderText("Search classes…")
        self._class_search.lineEdit().textChanged.connect(self._render_classes)
        card.add(self._class_search)
        self._class_list = QListWidget()
        self._class_list.itemChanged.connect(self._on_class_toggled)
        card.add(self._class_list)
        row = QHBoxLayout()
        for text, cb in [("Rename", self._rename_class), ("Merge", self._merge_class),
                         ("Delete", self._delete_class)]:
            b = QPushButton(text)
            b.clicked.connect(cb)
            row.addWidget(b)
        card.body.addLayout(row)
        return card

    # --- properties (right) ---
    def _properties(self) -> Card:
        card = Card("Properties")
        self._prop_label = QComboBox()
        self._prop_label.setEditable(True)
        # Relabel on commit (Enter / focus-out / dropdown pick), not per keystroke.
        self._prop_label.activated.connect(lambda _i: self._on_relabel(self._prop_label.currentText()))
        self._prop_label.lineEdit().editingFinished.connect(
            lambda: self._on_relabel(self._prop_label.currentText()))
        card.add(label("Label", "Muted"))
        card.add(self._prop_label)
        self._prop_info = label("No selection.", "Muted", wrap=True)
        card.add(self._prop_info)
        self._reseg_btn = QPushButton("Re-segment mask")
        self._reseg_btn.clicked.connect(self._resegment)
        card.add(self._reseg_btn)
        self._dup_btn = QPushButton("Duplicate (Ctrl+D)")
        self._dup_btn.clicked.connect(self.duplicate)
        card.add(self._dup_btn)
        self._del_btn = QPushButton("Delete (Del)")
        self._del_btn.clicked.connect(self.delete)
        card.add(self._del_btn)
        card.body.addStretch(1)
        return card

    def _filmstrip_widget(self) -> QListWidget:
        self._filmstrip = QListWidget()
        self._filmstrip.setViewMode(QListWidget.ViewMode.IconMode)
        self._filmstrip.setFlow(QListWidget.Flow.LeftToRight)
        self._filmstrip.setWrapping(False)
        self._filmstrip.setFixedHeight(104)
        self._filmstrip.setIconSize(QSize(120, 76))
        self._filmstrip.setMovement(QListWidget.Movement.Static)
        self._filmstrip.itemClicked.connect(self._on_filmstrip_click)
        return self._filmstrip

    # =================== project / image loading ===================
    def set_project(self, project_id: str) -> None:
        self._pid = project_id
        self._images = self._controller.project_images(project_id)
        self._idx = 0
        self._thumb_cache.clear()
        self._build_filmstrip()
        self._render_classes()
        if self._images:
            self._load_image(0)
        else:
            self._canvas.load_image("")
            self._notify.info("This project has no images yet.")

    def on_show(self) -> None:
        if self._pid and not self._images:
            self.set_project(self._pid)
        elif self._pid is None:  # opened straight from the nav — default to newest project
            datasets = self._controller.list_datasets()
            if datasets:
                self.set_project(datasets[-1].project_id)

    def _load_image(self, idx: int, *, keep_edits: bool = False) -> None:
        if not (0 <= idx < len(self._images)):
            return
        self._idx = idx
        ref = self._images[idx]
        self._canvas.load_image(ref.path)
        self._original = [self._box_dict(b) for b in self._controller.image_boxes(ref.image_id)]
        self._canvas.set_boxes([dict(b) for b in self._original])
        self._undo.clear()
        self._redo.clear()
        self._apply_filter()
        self._img_lbl.setText(f"Image {idx + 1} / {len(self._images)}")
        self._highlight_filmstrip()
        self._on_selection("")
        self._refresh_toolbar()

    @staticmethod
    def _box_dict(b) -> dict:
        return {"id": b.id, "x": b.x, "y": b.y, "w": b.w, "h": b.h, "label": b.label,
                "confidence": b.confidence, "state": b.state, "has_mask": b.has_mask}

    def _current_image_id(self) -> str | None:
        return self._images[self._idx].image_id if self._images else None

    # =================== undo / redo ===================
    def _record_undo(self, snapshot: list[dict]) -> None:
        self._undo.append(snapshot)
        self._redo.clear()
        self._refresh_toolbar()

    def _snapshot_undo(self) -> None:
        self._record_undo(self._canvas.boxes())

    def undo(self) -> None:
        if not self._undo:
            return
        self._redo.append(self._canvas.boxes())
        self._canvas.set_boxes(self._undo.pop())
        self._after_mutation()

    def redo(self) -> None:
        if not self._redo:
            return
        self._undo.append(self._canvas.boxes())
        self._canvas.set_boxes(self._redo.pop())
        self._after_mutation()

    def _on_changed(self) -> None:
        self._after_mutation()

    def _after_mutation(self) -> None:
        self._apply_filter()
        self._refresh_toolbar()
        self._on_selection(self._canvas.selected_id())

    # =================== edit actions ===================
    def delete(self) -> None:
        if self._canvas.selected() is None:
            return
        self._snapshot_undo()
        self._canvas.delete_selected()

    def duplicate(self) -> None:
        if self._canvas.selected() is None:
            return
        self._snapshot_undo()
        self._canvas.duplicate_selected()

    def ai_annotate(self) -> None:
        iid = self._current_image_id()
        if iid is None:
            return
        self.busy.emit(True)
        self._notify.info("Running AI detection on this image…")
        self._threads.submit(
            self._controller.ai_annotate, iid,
            on_finished=self._on_ai_done,
            on_error=lambda m: (self.busy.emit(False), self._notify.error(f"AI failed — {m}")),
        )

    def _on_ai_done(self, proposals) -> None:
        self.busy.emit(False)
        if not proposals:
            self._notify.info("AI found no objects.")
            return
        self._snapshot_undo()
        for p in proposals:
            self._canvas.add_box({"id": "", "x": p.x, "y": p.y, "w": p.w, "h": p.h,
                                  "label": p.label, "confidence": p.confidence})
        self._notify.success(f"Added {len(proposals)} AI boxes — review and Save.")
        self._render_classes()

    def _resegment(self) -> None:
        it = self._canvas.selected()
        iid = self._current_image_id()
        if it is None or iid is None:
            return
        r = it.rect()
        box = {"x": r.x(), "y": r.y(), "w": r.width(), "h": r.height()}
        self.busy.emit(True)
        self._threads.submit(
            self._controller.resegment, iid, box,
            on_finished=lambda mask, item=it: self._on_reseg_done(item, mask),
            on_error=lambda m: (self.busy.emit(False), self._notify.error(f"Segment failed — {m}")),
        )

    def _on_reseg_done(self, item, mask: dict) -> None:
        self.busy.emit(False)
        self._snapshot_undo()
        item.mask = mask
        item.has_mask = True
        self._canvas.show_mask(rle_to_qimage(mask["rle"], mask["width"], mask["height"],
                                             QColor(_color_for(item.label))))
        self._refresh_toolbar()
        self._notify.success("Mask regenerated — Save to keep it.")

    # =================== save ===================
    def _compute_ops(self) -> list[dict]:
        current = self._canvas.boxes()
        orig = {b["id"]: b for b in self._original if b["id"]}
        seen, ops = set(), []
        for c in current:
            box = {"x": c["x"], "y": c["y"], "w": c["w"], "h": c["h"]}
            payload = {"box": box, "label": c["label"], "confidence": c["confidence"]}
            if c.get("mask"):
                payload["mask"] = c["mask"]
            if not c["id"]:
                ops.append({"op": "create", **payload})
                continue
            seen.add(c["id"])
            o = orig.get(c["id"])
            if o is None:
                continue
            moved = any(abs(c[k] - o[k]) > 0.5 for k in ("x", "y", "w", "h"))
            if moved or c["label"] != o["label"] or c.get("mask"):
                ops.append({"op": "edit", "id": c["id"], **payload})
        for oid in orig:
            if oid not in seen:
                ops.append({"op": "delete", "id": oid})
        return ops

    def save(self) -> None:
        iid = self._current_image_id()
        if iid is None:
            return
        ops = self._compute_ops()
        if not ops:
            self._notify.info("Nothing to save.")
            return
        self.busy.emit(True)
        self._threads.submit(
            self._controller.save_edits, iid, ops,
            on_finished=lambda n: self._on_saved(n),
            on_error=lambda m: (self.busy.emit(False), self._notify.error(f"Save failed — {m}")),
        )

    def _on_saved(self, n: int) -> None:
        self.busy.emit(False)
        self._notify.success(f"Saved {n} change(s).")
        self._load_image(self._idx)  # reload with fresh ids/state
        self._render_classes()
        self._build_filmstrip()

    # =================== navigation ===================
    def prev_image(self) -> None:
        self._guarded_switch(self._idx - 1)

    def next_image(self) -> None:
        self._guarded_switch(self._idx + 1)

    def _guarded_switch(self, idx: int) -> None:
        if not (0 <= idx < len(self._images)):
            return
        if self._compute_ops():
            resp = QMessageBox.question(self, "Unsaved changes",
                                        "Discard unsaved edits on this image?")
            if resp != QMessageBox.StandardButton.Yes:
                return
        self._load_image(idx)

    # =================== filters ===================
    def _apply_filter(self) -> None:
        name, pred = _FILTERS[self._filter.currentIndex()]
        if pred == "dup":
            boxes = self._canvas.boxes()
            dup_ids = set()
            for i, a in enumerate(boxes):
                for b in boxes[i + 1:]:
                    if _iou(a, b) > 0.85:
                        dup_ids.add(id(a))
                        dup_ids.add(id(b))
            keys = {(round(a["x"]), round(a["y"]), round(a["w"]), round(a["h"]))
                    for a in boxes if id(a) in dup_ids}
            self._canvas.set_filter(
                lambda box: (round(box["x"]), round(box["y"]), round(box["w"]),
                             round(box["h"])) in keys)
        else:
            self._canvas.set_filter(pred)

    # =================== class manager ===================
    def _render_classes(self) -> None:
        if self._pid is None:
            return
        query = self._class_search.currentText().strip().lower()
        counts = self._controller.project_classes(self._pid)
        self._class_list.blockSignals(True)
        self._class_list.clear()
        for lbl in sorted(counts):
            if query and query not in lbl.lower():
                continue
            item = QListWidgetItem(f"  {lbl}  ({counts[lbl]})")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked if lbl in self._hidden
                               else Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, lbl)
            pix = QPixmap(12, 12)
            pix.fill(QColor(_color_for(lbl)))
            item.setIcon(QIcon(pix))
            self._class_list.addItem(item)
        self._class_list.blockSignals(False)
        # keep the relabel dropdown in sync
        cur = self._prop_label.currentText()
        self._prop_label.blockSignals(True)
        self._prop_label.clear()
        self._prop_label.addItems(sorted(counts))
        if cur:
            self._prop_label.setEditText(cur)
        self._prop_label.blockSignals(False)

    def _on_class_toggled(self, item: QListWidgetItem) -> None:
        lbl = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Unchecked:
            self._hidden.add(lbl)
        else:
            self._hidden.discard(lbl)
        self._canvas.set_hidden_labels(self._hidden)

    def _selected_class(self) -> str | None:
        item = self._class_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _rename_class(self) -> None:
        old = self._selected_class()
        if old is None or self._pid is None:
            self._notify.warning("Select a class to rename.")
            return
        new, ok = QInputDialog.getText(self, "Rename class", f"Rename '{old}' to:", text=old)
        if not ok or not new.strip() or new.strip() == old:
            return
        self._run_class_op(self._controller.rename_class, self._pid, old, new.strip(),
                           done=f"Renamed '{old}' → '{new.strip()}'.")

    def _merge_class(self) -> None:
        src = self._selected_class()
        if src is None or self._pid is None:
            self._notify.warning("Select the class to merge FROM.")
            return
        others = [c for c in self._controller.project_classes(self._pid) if c != src]
        if not others:
            self._notify.warning("No other class to merge into.")
            return
        target, ok = QInputDialog.getItem(self, "Merge class", f"Merge '{src}' into:", others, 0, False)
        if not ok or not target:
            return
        self._run_class_op(self._controller.merge_classes, self._pid, [src], target,
                           done=f"Merged '{src}' into '{target}'.")

    def _delete_class(self) -> None:
        lbl = self._selected_class()
        if lbl is None or self._pid is None:
            self._notify.warning("Select a class to delete.")
            return
        if QMessageBox.question(self, "Delete class",
                                f"Reject every '{lbl}' annotation? This can't be undone.") \
                != QMessageBox.StandardButton.Yes:
            return
        self._run_class_op(self._controller.delete_class, self._pid, lbl,
                           done=f"Deleted class '{lbl}'.")

    def _run_class_op(self, fn, *args, done: str) -> None:
        self.busy.emit(True)
        self._threads.submit(
            fn, *args,
            on_finished=lambda _n: self._on_class_op_done(done),
            on_error=lambda m: (self.busy.emit(False), self._notify.error(f"Failed — {m}")),
        )

    def _on_class_op_done(self, message: str) -> None:
        self.busy.emit(False)
        self._notify.success(message)
        self._hidden.clear()
        self._render_classes()
        self._load_image(self._idx)

    # =================== properties / selection ===================
    def _on_selection(self, box_id: str) -> None:
        it = self._canvas.selected()
        has = it is not None
        self._reseg_btn.setEnabled(has)
        self._dup_btn.setEnabled(has)
        self._del_btn.setEnabled(has)
        self._prop_label.setEnabled(has)
        if not has:
            self._prop_info.setText("No selection.")
            self._canvas.show_mask(None)
            return
        self._prop_label.blockSignals(True)
        self._prop_label.setEditText(it.label)
        self._prop_label.blockSignals(False)
        r = it.rect()
        self._prop_info.setText(
            f"conf {it.confidence:.2f}  ·  {int(r.width())}×{int(r.height())} px  ·  "
            f"mask: {'yes' if (it.mask or it.has_mask) else 'no'}")
        # read-only mask overlay
        if it.mask:
            self._canvas.show_mask(rle_to_qimage(it.mask["rle"], it.mask["width"],
                                                 it.mask["height"], QColor(_color_for(it.label))))
        elif it.has_mask and it.box_id:
            m = self._controller.box_mask(it.box_id)
            self._canvas.show_mask(
                rle_to_qimage(m["rle"], m["width"], m["height"], QColor(_color_for(it.label)))
                if m else None)
        else:
            self._canvas.show_mask(None)

    def _on_relabel(self, text: str) -> None:
        it = self._canvas.selected()
        if it is None or not text.strip() or text == it.label:
            return
        self._snapshot_undo()
        self._canvas.relabel_selected(text.strip())
        self._render_classes()

    # =================== filmstrip ===================
    def _build_filmstrip(self) -> None:
        self._filmstrip.clear()
        for i, ref in enumerate(self._images):
            icon = self._thumb_cache.get(ref.image_id)
            if icon is None:
                icon = QIcon(QPixmap(ref.path).scaled(120, 76, Qt.AspectRatioMode.KeepAspectRatio,
                                                      Qt.TransformationMode.SmoothTransformation))
                self._thumb_cache[ref.image_id] = icon
            item = QListWidgetItem(icon, str(i + 1))
            item.setData(Qt.ItemDataRole.UserRole, i)
            self._filmstrip.addItem(item)
        self._highlight_filmstrip()

    def _highlight_filmstrip(self) -> None:
        for row in range(self._filmstrip.count()):
            item = self._filmstrip.item(row)
            item.setForeground(QColor("#3d7eff") if row == self._idx else QColor("#8b9096"))
        if 0 <= self._idx < self._filmstrip.count():
            self._filmstrip.setCurrentRow(self._idx)

    def _on_filmstrip_click(self, item: QListWidgetItem) -> None:
        self._guarded_switch(item.data(Qt.ItemDataRole.UserRole))

    # =================== misc ===================
    def _canvas_fit(self) -> None:
        self._canvas.fit()

    def _export(self) -> None:
        if self._pid:
            self.export_requested.emit(self._pid)

    def _refresh_toolbar(self) -> None:
        self._btn["undo"].setEnabled(bool(self._undo))
        self._btn["redo"].setEnabled(bool(self._redo))
        dirty = bool(self._images) and bool(self._compute_ops())
        self._btn["save"].setEnabled(dirty)
        self._dirty_lbl.setText("● unsaved" if dirty else "")
        for k in ("prev",):
            self._btn[k].setEnabled(self._idx > 0)
        self._btn["next"].setEnabled(self._idx < len(self._images) - 1)

    def _install_shortcuts(self) -> None:
        for seq, cb in [
            (QKeySequence(Qt.Key.Key_Delete), self.delete),
            (QKeySequence.StandardKey.Undo, self.undo),
            (QKeySequence.StandardKey.Redo, self.redo),
            (QKeySequence.StandardKey.Save, self.save),
            (QKeySequence.StandardKey.SelectAll, lambda: self._canvas.select_all()),
            (QKeySequence("Ctrl+D"), self.duplicate),
            (QKeySequence(Qt.Key.Key_Right), self.next_image),
            (QKeySequence(Qt.Key.Key_Left), self.prev_image),
            (QKeySequence(Qt.Key.Key_Escape), lambda: self._canvas.scene().clearSelection()),
        ]:
            QShortcut(seq, self, activated=cb)

    def context(self) -> tuple[str, list[tuple[str, str]]]:
        return ("Annotation", [
            ("Image", f"{self._idx + 1}/{len(self._images)}" if self._images else "—"),
            ("Unsaved", "yes" if (self._images and self._compute_ops()) else "no"),
            ("Undo", str(len(self._undo))),
        ])
