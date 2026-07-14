"""Provider interface + loader (Agent Framework).

`LLMProvider` is the single abstraction every LLM backend implements. Adding a
provider = writing one adapter and pointing config at it; no existing agent code
changes. Providers are resolved by import path exactly like the CV-model registry.

Provider errors map onto the platform's error taxonomy so the client's retry
policy (transient -> retry, config -> fail fast) works uniformly across providers.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from vds.agents.messages import CompletionRequest, CompletionResponse, StreamChunk
from vds.config.settings import LLMSettings
from vds.core.errors import ConfigError, TransientError, VDSError


class ProviderError(TransientError):
    """A recoverable provider failure (5xx, connection reset). Retried."""


class ProviderTimeout(TransientError):
    """A provider call exceeded its timeout. Retried, then surfaced."""


class ProviderNotConfigured(ConfigError):
    """A provider is selected but its SDK or credentials are missing."""


class CapabilityNotSupported(VDSError):
    """A provider was asked for a capability (streaming/tools) it lacks."""


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    supports_tools: bool
    supports_streaming: bool

    def complete(self, request: CompletionRequest) -> CompletionResponse: ...
    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]: ...


class BaseProvider(ABC):
    """Convenience base: default capability flags + a streaming stub that fails
    loudly until an adapter opts in. Adapters override what they support."""

    name: str = "base"
    supports_tools: bool = False
    supports_streaming: bool = False

    def __init__(self, config: LLMSettings | None = None) -> None:
        self.config = config or LLMSettings()

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResponse: ...

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        raise CapabilityNotSupported(f"{self.name} does not support streaming")


def load_provider(import_path: str, config: LLMSettings) -> LLMProvider:
    """Instantiate a provider from a `module:ClassName` string with its config."""
    if ":" not in import_path:
        raise ConfigError(f"provider path must be 'module:ClassName', got {import_path!r}")
    module_name, class_name = import_path.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"cannot load provider {import_path!r}: {exc}") from exc
    provider = cls(config)
    if not isinstance(provider, LLMProvider):
        raise ConfigError(f"{import_path!r} does not implement the LLMProvider protocol")
    return provider
