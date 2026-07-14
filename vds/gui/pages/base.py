"""Page base (Phase 11).

Every workspace page is a QWidget subclass with a stable `name`, an `on_show`
refresh hook (called when navigated to), and a `context()` describing what the
right sidebar should display for it. The shell owns navigation; a page only knows
itself.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget


class Page(QWidget):
    name: str = "Page"

    def on_show(self) -> None:
        """Called each time the page becomes visible — refresh backend-derived data
        here so the page is never stale."""

    def context(self) -> tuple[str, list[tuple[str, str]]]:
        """(title, [(label, value), ...]) for the context sidebar."""
        return (self.name, [])
