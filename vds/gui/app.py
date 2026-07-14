"""Application entry point (Phase 11).

Builds the QApplication, the BackendController (over the existing Container), the
ThemeManager (restoring the persisted theme), and the MainWindow. This is the only
place a QApplication is created; everything else is a plain widget/manager that a
test can build with `QApplication.instance()`.

Run:  python -m vds.gui   ·   vds-studio   ·   vds gui
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from vds.gui.controller import BackendController
from vds.gui.main_window import MainWindow
from vds.gui.settings import GuiSettings
from vds.gui.theme import ThemeManager


def build_window(controller: BackendController | None = None) -> MainWindow:
    """Construct the main window on the current QApplication. Injectable controller
    keeps it testable (pass a Container backed by a temp DB)."""
    settings = GuiSettings()
    app = QApplication.instance()
    theme = ThemeManager(app, initial=settings.theme)  # type: ignore[arg-type]
    return MainWindow(controller or BackendController(), theme, settings)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("AutoDataForge")
    app.setOrganizationName("VisionDatasetStudio")
    window = build_window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
