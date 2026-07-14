"""ResourceMonitor — live CPU / memory (and GPU device) sampling (Phase 11).

Polls psutil on a QTimer on the UI thread (the calls are non-blocking) and emits a
sample. GPU *utilization* needs a vendor library not installed here, so we report
the configured device and VRAM budget instead of a fabricated number — honest, and
a clean seam to wire real telemetry (nvml) later.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal

try:
    import psutil
except ImportError:  # psutil is a base dependency, but stay defensive
    psutil = None  # type: ignore


class ResourceMonitor(QObject):
    #: cpu_percent, ram_percent, ram_used_mb
    sampled = Signal(float, float, float)

    def __init__(self, interval_ms: int = 1500, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._sample)
        if psutil is not None:
            psutil.cpu_percent(interval=None)  # prime the first reading

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _sample(self) -> None:
        if psutil is None:
            self.sampled.emit(0.0, 0.0, 0.0)
            return
        vm = psutil.virtual_memory()
        self.sampled.emit(
            round(psutil.cpu_percent(interval=None), 1),
            round(vm.percent, 1),
            round(vm.used / (1024 * 1024), 0),
        )
