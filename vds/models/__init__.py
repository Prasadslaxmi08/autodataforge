"""L1 — model runtime and plugin layer.

Capability protocols + import-path registry + GPU-budget manager. Every model is
a plugin resolved from configuration; nothing above this layer knows a model's
concrete type.
"""

from vds.models.gpu import GpuManager
from vds.models.protocols import Capability
from vds.models.registry import ModelRegistry, load_adapter

__all__ = ["Capability", "GpuManager", "ModelRegistry", "load_adapter"]
