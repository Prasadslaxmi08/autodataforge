"""ImportWizard — a guided, four-step import (Phase 18).

Replaces the bare "pick a folder, type a name" dialogs with a wizard:

    Source  →  Preview  →  Configure  →  Summary

It collects inputs only; the actual run stays on the shell's ThreadManager (the one
place threading + progress live), so the wizard emits a *request* rather than calling
the pipeline itself:

  * Folder / ZIP  →  ``request_folder_import(folder, name, export_format)``
                     (ZIP is unzipped to a temp folder first — a UI-only shim over
                      the existing folder import; no backend change.)
  * Video         →  ``request_video_import(path, name)`` (the shell opens the
                     existing VideoImportDialog, reused unchanged).

COCO / YOLO source tiles are shown disabled: the backend has no dataset-label ingest
and this phase must not change it.

Tests drive it without native dialogs via ``set_source(kind, path)``.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from vds.gui.controller import BackendController
from vds.gui.widgets.common import label
from vds.models.adapters.yolo_config import YOLO_PRESETS

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_VIDEO_FILTER = "Videos (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.gif *.webp *.tif *.tiff)"


def _count_images(folder: str) -> tuple[int, int]:
    """(image count, total bytes) for a folder tree — the same extensions the
    controller's source scan uses."""
    count, size = 0, 0
    root = Path(folder)
    if root.is_dir():
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                count += 1
                size += p.stat().st_size
    return count, size


def _zip_image_count(path: str) -> tuple[int, int]:
    try:
        with zipfile.ZipFile(path) as zf:
            infos = [i for i in zf.infolist()
                     if not i.is_dir() and Path(i.filename).suffix.lower() in _IMAGE_EXTS]
            return len(infos), sum(i.file_size for i in infos)
    except (zipfile.BadZipFile, OSError):
        return 0, 0


def _human_size(nbytes: int) -> str:
    val = float(nbytes)
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024 or unit == "GB":
            return f"{val:.0f} {unit}" if unit == "B" else f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} GB"


class _SourcePage(QWizardPage):
    def __init__(self, wizard: ImportWizard) -> None:
        super().__init__()
        self._wiz = wizard
        self.setTitle("Choose a source")
        self.setSubTitle("Where should the dataset come from?")
        root = QVBoxLayout(self)
        for kind, text, enabled in [
            ("folder", "📁  Folder of images", True),
            ("zip", "🗜  ZIP archive of images", True),
            ("video", "🎞  Video file", True),
            ("coco", "COCO dataset  ·  coming soon", False),
            ("yolo", "YOLO dataset  ·  coming soon", False),
        ]:
            btn = QPushButton(text)
            btn.setEnabled(enabled)
            if enabled:
                btn.clicked.connect(lambda _c=False, k=kind: self._pick(k))
            root.addWidget(btn)
        self._chosen = label("No source selected.", "Muted")
        root.addWidget(self._chosen)
        root.addStretch(1)

    def _pick(self, kind: str) -> None:
        if kind == "folder":
            path = QFileDialog.getExistingDirectory(self, "Select image folder")
        elif kind == "zip":
            path, _ = QFileDialog.getOpenFileName(self, "Select ZIP archive", "", "ZIP (*.zip)")
        else:  # video — delegated out of the wizard entirely
            path, _ = QFileDialog.getOpenFileName(self, "Select a video", "", _VIDEO_FILTER)
            if path:
                self._wiz.delegate_video(path)
            return
        if path:
            self._wiz.set_source(kind, path)
            self._chosen.setText(f"{kind.upper()}: {path}")
            self.completeChanged.emit()
            self._wiz.next()

    def isComplete(self) -> bool:  # noqa: N802 (Qt override)
        return bool(self._wiz.source_path) and self._wiz.kind in ("folder", "zip")


class _PreviewPage(QWizardPage):
    def __init__(self, wizard: ImportWizard) -> None:
        super().__init__()
        self._wiz = wizard
        self.setTitle("Preview")
        self.setSubTitle("What will be imported")
        self._form = QFormLayout(self)

    def initializePage(self) -> None:  # noqa: N802 (Qt override)
        while self._form.rowCount():
            self._form.removeRow(0)
        count, size = self._wiz.scan()
        self._wiz.estimated_images = count
        self._form.addRow("Source", QLabel(self._wiz.source_path))
        self._form.addRow("Images found", QLabel(str(count)))
        self._form.addRow("Estimated storage", QLabel(_human_size(size)))
        # ponytail: exact dup detection is the ingest average-hash pass (heavy); show
        # a cheap heuristic here and let the real pipeline do the authoritative dedup.
        self._form.addRow("Duplicate estimate", QLabel("computed during import (approx.)"))


class _ConfigurePage(QWizardPage):
    def __init__(self, wizard: ImportWizard) -> None:
        super().__init__()
        self._wiz = wizard
        d = wizard.detector_prefs
        self.setTitle("Configure")
        self.setSubTitle("Name the dataset, pick the model and export format")
        form = QFormLayout(self)
        self._name = QLineEdit("dataset")
        self._model = QComboBox()
        self._model.addItems([*YOLO_PRESETS, "Custom .pt…"])
        if d["model"] in YOLO_PRESETS:
            self._model.setCurrentText(d["model"])
        self._custom_path = d["model"] if d["model"] not in YOLO_PRESETS else ""
        self._model.currentTextChanged.connect(self._maybe_browse)
        self._export = QComboBox()
        self._export.addItems(wizard.controller.export_options())
        form.addRow("Dataset name", self._name)
        form.addRow("Model", self._model)
        form.addRow("Export format", self._export)

        # Advanced — collapsed by default (checkable group). The normal user never
        # touches these; defaults come from remembered preferences.
        adv = QGroupBox("Advanced")
        adv.setCheckable(True)
        adv.setChecked(False)
        af = QFormLayout(adv)
        self._conf = QDoubleSpinBox()
        self._conf.setRange(0.01, 1.0)
        self._conf.setSingleStep(0.05)
        self._conf.setValue(d["conf"])
        self._iou = QDoubleSpinBox()
        self._iou.setRange(0.1, 1.0)
        self._iou.setSingleStep(0.05)
        self._iou.setValue(d["iou"])
        self._imgsz = QSpinBox()
        self._imgsz.setRange(320, 1920)
        self._imgsz.setSingleStep(32)
        self._imgsz.setValue(d["imgsz"])
        self._segment = QCheckBox("Instance segmentation (needs a -seg model)")
        self._segment.setChecked(d["segment"])
        af.addRow("Detection confidence", self._conf)
        af.addRow("IoU threshold", self._iou)
        af.addRow("Max image size", self._imgsz)
        af.addRow("", self._segment)
        form.addRow(adv)
        self._name.textChanged.connect(self.completeChanged)

    def _maybe_browse(self, text: str) -> None:
        if text == "Custom .pt…":
            path, _ = QFileDialog.getOpenFileName(self, "Select a YOLO .pt model", "",
                                                  "YOLO weights (*.pt)")
            if path:
                self._custom_path = path
                self.setToolTip(path)

    def initializePage(self) -> None:  # noqa: N802
        if self._wiz.kind == "zip" and self._name.text() == "dataset":
            self._name.setText(Path(self._wiz.source_path).stem or "dataset")

    def isComplete(self) -> bool:  # noqa: N802
        return bool(self._name.text().strip())

    def commit(self) -> None:
        self._wiz.name = self._name.text().strip()
        self._wiz.export_format = self._export.currentText()
        chosen = self._model.currentText()
        self._wiz.model = self._custom_path if chosen == "Custom .pt…" else chosen
        self._wiz.conf = self._conf.value()
        self._wiz.iou = self._iou.value()
        self._wiz.imgsz = self._imgsz.value()
        self._wiz.segment = self._segment.isChecked()


class _SummaryPage(QWizardPage):
    def __init__(self, wizard: ImportWizard) -> None:
        super().__init__()
        self._wiz = wizard
        self.setTitle("Ready to import")
        self.setSubTitle("Review, then Begin Import")
        self._form = QFormLayout(self)

    def initializePage(self) -> None:  # noqa: N802
        self._wiz.config_page.commit()
        while self._form.rowCount():
            self._form.removeRow(0)
        self._form.addRow("Dataset", QLabel(self._wiz.name))
        self._form.addRow("Estimated images", QLabel(str(self._wiz.estimated_images)))
        self._form.addRow("Model", QLabel(Path(self._wiz.model).name))
        self._form.addRow("Confidence / IoU", QLabel(f"{self._wiz.conf:.2f} / {self._wiz.iou:.2f}"))
        self._form.addRow("Segmentation", QLabel("on" if self._wiz.segment else "off"))
        self._form.addRow("Export format", QLabel(self._wiz.export_format))


class ImportWizard(QWizard):
    #: folder/zip → the shell runs the pipeline off-thread
    request_folder_import = Signal(str, str, str)  # folder, name, export_format
    #: video → the shell opens the existing VideoImportDialog
    request_video_import = Signal(str, str)  # path, name

    def __init__(self, controller: BackendController, gui=None, preset_path: str = "",
                 parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.gui = gui  # GuiSettings, for remembering the model/detection prefs
        self.detector_prefs = gui.detector if gui is not None else {
            "model": "yolo11n.pt", "conf": 0.25, "iou": 0.7, "imgsz": 640, "segment": False}
        self.setWindowTitle("Import Dataset")
        self.resize(560, 500)
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)

        # collected state
        self.kind: str = ""
        self.source_path: str = ""
        self.name: str = "dataset"
        self.export_format: str = "coco"
        self.estimated_images: int = 0
        self.model: str = self.detector_prefs["model"]
        self.conf: float = self.detector_prefs["conf"]
        self.iou: float = self.detector_prefs["iou"]
        self.imgsz: int = self.detector_prefs["imgsz"]
        self.segment: bool = self.detector_prefs["segment"]
        self._tempdir: str | None = None

        self.addPage(_SourcePage(self))
        self.addPage(_PreviewPage(self))
        self.config_page = _ConfigurePage(self)
        self.addPage(self.config_page)
        self.addPage(_SummaryPage(self))
        self.setButtonText(QWizard.WizardButton.FinishButton, "Begin Import")
        self.finished.connect(self._on_finished)

        if preset_path:
            self._apply_preset(preset_path)

    # --- programmatic entry (drag&drop preset + tests) ---
    def set_source(self, kind: str, path: str) -> None:
        self.kind = kind
        self.source_path = path

    def _apply_preset(self, path: str) -> None:
        # Only folder/zip presets are applied here (no signals emitted at construction
        # time — the shell connects them afterwards). Video presets are detected and
        # routed by the shell before the wizard is built.
        p = Path(path)
        if p.is_dir():
            self.set_source("folder", path)
        elif p.suffix.lower() == ".zip":
            self.set_source("zip", path)
        else:
            return  # stay on the Source page
        self.setStartId(1)  # skip the Source page — jump to Preview

    def delegate_video(self, path: str) -> None:
        name = Path(path).stem or "video-dataset"
        self._apply_detector_config()  # video reuses the remembered model prefs
        self.request_video_import.emit(path, name)
        self.reject()  # close the wizard; the video dialog takes over

    def _apply_detector_config(self) -> None:
        """Remember the choices and make them active before the pipeline runs."""
        if self.gui is not None:
            self.gui.set_detector(self.model, self.conf, self.iou, self.imgsz, self.segment)
        self.controller.set_detector_config(self.model, self.conf, self.iou,
                                            self.imgsz, self.segment)

    # --- scanning ---
    def scan(self) -> tuple[int, int]:
        return _zip_image_count(self.source_path) if self.kind == "zip" \
            else _count_images(self.source_path)

    def resolved_folder(self) -> str:
        """The folder to hand the folder-import path. Unzips a ZIP to a temp dir."""
        if self.kind != "zip":
            return self.source_path
        self._tempdir = tempfile.mkdtemp(prefix="vds_zip_")
        with zipfile.ZipFile(self.source_path) as zf:
            zf.extractall(self._tempdir)
        return self._tempdir

    def _on_finished(self, result: int) -> None:
        if result != QWizard.DialogCode.Accepted:
            return
        if self.kind in ("folder", "zip"):
            self._apply_detector_config()
            self.request_folder_import.emit(self.resolved_folder(), self.name, self.export_format)
