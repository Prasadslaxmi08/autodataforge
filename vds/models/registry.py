"""Model plugin registry (System Design §5.1).

A model is resolved from configuration by import path — `"module.path:ClassName"`
— so adding a model is: write one adapter class, change one config line. Import
paths *are* the plugin-discovery mechanism; no entry-point framework, no scan.

The registry resolves lazily and caches, and routes every load/unload through
the GpuManager so the VRAM budget is enforced in one place.
"""

from __future__ import annotations

import importlib
from typing import Any

from vds.config.settings import ModelSelection
from vds.core.errors import ConfigError
from vds.logging import get_logger
from vds.models.gpu import GpuManager
from vds.models.protocols import ModelAdapter

log = get_logger(__name__)


def load_adapter(import_path: str) -> ModelAdapter:
    """Import and instantiate an adapter from a `module:ClassName` string."""
    if ":" not in import_path:
        raise ConfigError(
            f"model import path must be 'module:ClassName', got {import_path!r}"
        )
    module_name, class_name = import_path.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"cannot load adapter {import_path!r}: {exc}") from exc
    adapter = cls()
    if not isinstance(adapter, ModelAdapter):
        raise ConfigError(f"{import_path!r} does not implement the ModelAdapter protocol")
    return adapter


class ModelRegistry:
    """Resolves capability -> adapter instance, caching by import path so two
    capabilities pointing at the same model share one loaded instance."""

    def __init__(self, selection: ModelSelection, gpu: GpuManager) -> None:
        self._selection = selection
        self._gpu = gpu
        self._cache: dict[str, ModelAdapter] = {}

    def _resolve(self, capability: str) -> ModelAdapter:
        import_path = getattr(self._selection, capability, None)
        if import_path is None:
            raise ConfigError(f"no model configured for capability {capability!r}")
        if import_path not in self._cache:
            log.info("registry.load_adapter", capability=capability, path=import_path)
            self._cache[import_path] = load_adapter(import_path)
        return self._cache[import_path]

    def get(self, capability: str) -> Any:
        """Return the loaded adapter for a capability (see protocols.Capability)."""
        adapter = self._resolve(capability)
        self._gpu.ensure_loaded(adapter)
        return adapter

    def describe(self) -> dict[str, str]:
        """capability -> configured import path, for diagnostics / config-check."""
        return {
            cap: getattr(self._selection, cap)
            for cap in type(self._selection).model_fields
        }
