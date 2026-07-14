"""Desktop GUI (Phase 11) — a PySide6 front end over the existing platform.

Strict UI/backend separation: everything the UI needs comes through
`BackendController`; no business logic is duplicated here. The GUI is a set of
independent modules — shell, navigation, workspace pages, context sidebar, bottom
panel, status bar, thread/theme/settings/notification managers.
"""

from vds.gui.controller import BackendController

__all__ = ["BackendController", "main"]


def main() -> int:
    from vds.gui.app import main as _main

    return _main()
