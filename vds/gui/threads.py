"""ThreadManager — keep the UI thread free (Phase 11).

Every long-running backend call (dataset import runs the whole pipeline) executes
on a QThreadPool worker. The worker communicates back to the UI ONLY through Qt
signals, which Qt marshals to the receiver's thread — so the backend never touches
a widget directly. This is the one rule that keeps the app responsive and
thread-safe.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot


class WorkerSignals(QObject):
    started = Signal()
    progress = Signal(int, str)  # percent, message
    finished = Signal(object)  # the return value of the callable
    error = Signal(str)  # "ExceptionType: message"


class Worker(QRunnable):
    """Runs `fn(*args, **kwargs)` off the UI thread. If `fn` accepts a `progress`
    keyword, it receives a callable that emits the progress signal."""

    def __init__(self, fn: Callable, *args, wants_progress: bool = False, **kwargs) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._wants_progress = wants_progress
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        self.signals.started.emit()
        try:
            if self._wants_progress:
                self._kwargs["progress"] = lambda pct, msg: self.signals.progress.emit(pct, msg)
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # surface, never crash the pool
            self.signals.error.emit(f"{type(exc).__name__}: {exc}")
            return
        self.signals.finished.emit(result)


class ThreadManager:
    """Thin owner of the shared thread pool; also tracks how many tasks run."""

    def __init__(self) -> None:
        self._pool = QThreadPool.globalInstance()
        self.active = 0
        # Callers discard the returned Worker, so nothing else keeps it (and its
        # WorkerSignals QObject) alive until the queued finished/error signal is
        # delivered on the UI thread. Without this the worker is GC'd the instant
        # run() returns, the signal fires from a freed sender, and Qt segfaults.
        self._workers: set[Worker] = set()

    def submit(
        self,
        fn: Callable,
        *args,
        wants_progress: bool = False,
        on_finished: Callable | None = None,
        on_error: Callable | None = None,
        on_progress: Callable | None = None,
        on_started: Callable | None = None,
        **kwargs,
    ) -> Worker:
        worker = Worker(fn, *args, wants_progress=wants_progress, **kwargs)
        self.active += 1
        self._workers.add(worker)

        def _done(_result=None):
            self.active = max(0, self.active - 1)
            # Release on the next event-loop tick, after this emission fully
            # completes, so we never drop the last ref to a live sender mid-signal.
            QTimer.singleShot(0, lambda: self._workers.discard(worker))

        worker.signals.finished.connect(lambda r: _done())
        worker.signals.error.connect(lambda e: _done())
        if on_started:
            worker.signals.started.connect(on_started)
        if on_progress:
            worker.signals.progress.connect(on_progress)
        if on_finished:
            worker.signals.finished.connect(on_finished)
        if on_error:
            worker.signals.error.connect(on_error)
        self._pool.start(worker)
        return worker

    def wait(self, ms: int = 30000) -> bool:
        return self._pool.waitForDone(ms)
