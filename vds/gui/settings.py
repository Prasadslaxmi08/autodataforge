"""GuiSettings — persisted UI preferences and window state (Phase 11).

A thin typed wrapper over QSettings (native registry/plist/ini per OS). Persists
the theme, the last-selected page, the last import directory, and the main-window
geometry/state so the app restores exactly where the user left it. This is UI
state only — it never touches the backend `Settings` (model/provider config).
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QSettings

ORG = "VisionDatasetStudio"
APP = "Studio"


class GuiSettings:
    def __init__(self, org: str = ORG, app: str = APP, settings: QSettings | None = None) -> None:
        # An injected QSettings (e.g. a temp INI file) keeps tests isolated from the
        # user's real registry/plist store.
        self._s = settings if settings is not None else QSettings(org, app)

    # --- preferences ---
    @property
    def theme(self) -> str:
        return str(self._s.value("ui/theme", "dark"))

    @theme.setter
    def theme(self, value: str) -> None:
        self._s.setValue("ui/theme", value)

    @property
    def last_page(self) -> str:
        return str(self._s.value("ui/last_page", "Projects"))

    @last_page.setter
    def last_page(self, value: str) -> None:
        self._s.setValue("ui/last_page", value)

    @property
    def last_import_dir(self) -> str:
        return str(self._s.value("ui/last_import_dir", ""))

    @last_import_dir.setter
    def last_import_dir(self, value: str) -> None:
        self._s.setValue("ui/last_import_dir", value)

    # --- detector (YOLO) preferences, remembered across runs (Phase 18.5) ---
    @property
    def detector(self) -> dict:
        return {
            "model": str(self._s.value("det/model", "yolo11n.pt")),
            "conf": float(self._s.value("det/conf", 0.25)),
            "iou": float(self._s.value("det/iou", 0.7)),
            "imgsz": int(self._s.value("det/imgsz", 640)),
            "segment": str(self._s.value("det/segment", "false")).lower() == "true",
        }

    def set_detector(self, model: str, conf: float, iou: float, imgsz: int, segment: bool) -> None:
        self._s.setValue("det/model", model)
        self._s.setValue("det/conf", conf)
        self._s.setValue("det/iou", iou)
        self._s.setValue("det/imgsz", imgsz)
        self._s.setValue("det/segment", "true" if segment else "false")

    # --- window restoration ---
    def save_window(self, geometry: QByteArray, state: QByteArray) -> None:
        self._s.setValue("win/geometry", geometry)
        self._s.setValue("win/state", state)

    def window_geometry(self) -> QByteArray | None:
        return self._s.value("win/geometry")

    def window_state(self) -> QByteArray | None:
        return self._s.value("win/state")

    def sync(self) -> None:
        self._s.sync()
