"""GPU manager (System Design §5.1, NFR-2).

Enforces the VRAM budget by loading models sequentially and evicting the
least-recently-used ones when the budget would be exceeded. Adapters request
loading through here and never call `.cuda()` themselves — VRAM policy lives in
exactly one place.

Bootstrap scope: the eviction accounting is real; actual device placement is the
adapter's job (a no-op for the FakeAdapter).
"""

from __future__ import annotations

from collections import OrderedDict

from vds.logging import get_logger
from vds.models.protocols import ModelAdapter

log = get_logger(__name__)


class GpuManager:
    def __init__(self, vram_budget_mb: int) -> None:
        self._budget = vram_budget_mb
        self._loaded: OrderedDict[str, ModelAdapter] = OrderedDict()

    @property
    def used_mb(self) -> int:
        return sum(a.vram_estimate_mb for a in self._loaded.values())

    def ensure_loaded(self, adapter: ModelAdapter) -> None:
        """Load `adapter`, evicting LRU models until it fits the budget."""
        if adapter.name in self._loaded:
            self._loaded.move_to_end(adapter.name)  # mark most-recently-used
            return

        while self._loaded and self.used_mb + adapter.vram_estimate_mb > self._budget:
            name, victim = self._loaded.popitem(last=False)  # evict LRU
            log.info("gpu.evict", model=name, freed_mb=victim.vram_estimate_mb)
            victim.unload()

        adapter.load()
        self._loaded[adapter.name] = adapter
        log.info("gpu.load", model=adapter.name, used_mb=self.used_mb, budget=self._budget)

    def unload_all(self) -> None:
        for adapter in self._loaded.values():
            adapter.unload()
        self._loaded.clear()
