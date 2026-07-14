"""NotificationSystem — one place to raise user-facing messages (Phase 11).

Any module emits a notification through here; the shell routes it to the status
bar (transient) and the bottom logging panel (persistent). Decoupling via a signal
means pages don't hold references to the status bar or log widget.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

LEVELS = ("info", "success", "warning", "error")


class NotificationSystem(QObject):
    #: level, message
    notified = Signal(str, str)

    def emit_message(self, message: str, level: str = "info") -> None:
        self.notified.emit(level if level in LEVELS else "info", message)

    def info(self, message: str) -> None:
        self.emit_message(message, "info")

    def success(self, message: str) -> None:
        self.emit_message(message, "success")

    def warning(self, message: str) -> None:
        self.emit_message(message, "warning")

    def error(self, message: str) -> None:
        self.emit_message(message, "error")
