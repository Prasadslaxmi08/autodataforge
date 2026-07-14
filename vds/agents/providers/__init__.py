"""LLM providers. Adapters are resolved by import path (see `load_provider`), so
this package intentionally exposes only the interface, loader, and the default
Echo provider — the real adapters (anthropic/openai/ollama) are imported lazily
by path and never at package import, keeping optional SDKs off the base install.
"""

from vds.agents.providers.base import (
    BaseProvider,
    CapabilityNotSupported,
    LLMProvider,
    ProviderError,
    ProviderNotConfigured,
    ProviderTimeout,
    load_provider,
)
from vds.agents.providers.echo import EchoProvider

__all__ = [
    "BaseProvider",
    "CapabilityNotSupported",
    "EchoProvider",
    "LLMProvider",
    "ProviderError",
    "ProviderNotConfigured",
    "ProviderTimeout",
    "load_provider",
]
