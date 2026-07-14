"""BottomPanel — the processing log, running tasks, and resource meters (Phase 11).

Tabs: Log (all notifications, timestamped), Tasks (running/queued), Problems
(warnings + errors). A right-aligned resource strip shows live CPU / memory and the
GPU device. Purely a sink — it observes signals, drives nothing.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

_COLORS = {"info": "#8b9096", "success": "#4caf82", "warning": "#e0a458", "error": "#e0605e"}


class BottomPanel(QWidget):
    def __init__(self, gpu_device: str = "cpu", vram_budget_mb: int = 0) -> None:
        super().__init__()
        self._counter = 0
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._tabs = QTabWidget()
        self._log = self._make_view()
        self._tasks = self._make_view()
        self._problems = self._make_view()
        self._tabs.addTab(self._log, "Log")
        self._tabs.addTab(self._tasks, "Tasks")
        self._tabs.addTab(self._problems, "Problems")
        root.addWidget(self._tabs)

        strip = QWidget()
        row = QHBoxLayout(strip)
        row.setContentsMargins(10, 4, 10, 4)
        self._cpu = QLabel("CPU —")
        self._mem = QLabel("MEM —")
        self._gpu = QLabel(f"GPU {gpu_device}"
                           + (f" · {vram_budget_mb} MB" if vram_budget_mb else ""))
        for w in (self._cpu, self._mem, self._gpu):
            w.setObjectName("Muted")
        row.addStretch(1)
        row.addWidget(self._cpu)
        row.addWidget(self._sep())
        row.addWidget(self._mem)
        row.addWidget(self._sep())
        row.addWidget(self._gpu)
        root.addWidget(strip)

    @staticmethod
    def _make_view() -> QPlainTextEdit:
        v = QPlainTextEdit()
        v.setReadOnly(True)
        v.setMaximumBlockCount(2000)  # ring buffer; never unbounded
        return v

    def _sep(self) -> QLabel:
        lbl = QLabel("·")
        lbl.setObjectName("Muted")
        return lbl

    # --- slots ---
    def log(self, level: str, message: str) -> None:
        self._counter += 1
        color = _COLORS.get(level, _COLORS["info"])
        line = f'<span style="color:{color}">[{level.upper():7}]</span> {message}'
        self._log.appendHtml(line)
        if level in ("warning", "error"):
            self._problems.appendHtml(line)
            self._tabs.setTabText(2, f"Problems ({self._problem_count()})")

    def set_task(self, message: str) -> None:
        self._tasks.appendHtml(message)

    def _problem_count(self) -> int:
        return len([ln for ln in self._problems.toPlainText().splitlines() if ln.strip()])

    def update_resources(self, cpu: float, ram_pct: float, ram_mb: float) -> None:
        self._cpu.setText(f"CPU {cpu:.0f}%")
        self._mem.setText(f"MEM {ram_pct:.0f}% ({ram_mb:.0f} MB)")
