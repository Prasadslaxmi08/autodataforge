"""Provider-layer tests: loading, config selection, payload translation, capabilities."""

from __future__ import annotations

import pytest

from vds.agents.messages import CompletionRequest, Message, ToolSpec
from vds.agents.providers.anthropic import AnthropicProvider
from vds.agents.providers.base import (
    CapabilityNotSupported,
    load_provider,
)
from vds.agents.providers.echo import EchoProvider
from vds.agents.providers.ollama import OllamaProvider
from vds.agents.providers.openai import OpenAIProvider
from vds.config.settings import LLMSettings
from vds.core.errors import ConfigError


def _req(model="m", **kw) -> CompletionRequest:
    return CompletionRequest(
        model=model,
        messages=[Message(role="system", content="sys"), Message(role="user", content="hi")],
        **kw,
    )


# --- loading / config ---
def test_load_echo_provider():
    p = load_provider("vds.agents.providers.echo:EchoProvider", LLMSettings())
    assert p.name == "echo"


def test_load_provider_bad_path_raises():
    with pytest.raises(ConfigError):
        load_provider("not_a_path", LLMSettings())
    with pytest.raises(ConfigError):
        load_provider("vds.agents.providers.echo:Nope", LLMSettings())


def test_config_selects_provider():
    settings = LLMSettings(provider="vds.agents.providers.openai:OpenAIProvider")
    p = load_provider(settings.provider, settings)
    assert p.name == "openai"


# --- echo behaviour ---
def test_echo_completes():
    resp = EchoProvider().complete(_req())
    assert resp.text == "echo: hi"
    assert resp.provider == "echo"
    assert resp.usage.completion_tokens == 2


def test_echo_streaming():
    chunks = list(EchoProvider(reply="a b c").stream(_req()))
    assert chunks[-1].done is True
    assert "".join(c.delta for c in chunks).strip() == "a b c"


def test_unsupported_streaming_raises():
    # A provider that does not override stream() must fail loudly, not silently.
    from vds.agents.providers.base import BaseProvider

    class NoStream(BaseProvider):
        name = "nostream"

        def complete(self, request):
            raise NotImplementedError

    with pytest.raises(CapabilityNotSupported):
        NoStream().stream(_req())


# --- payload translation (pure, no network) ---
def test_anthropic_payload_splits_system():
    payload = AnthropicProvider(LLMSettings())._build_payload(_req(max_tokens=50))
    assert payload["system"] == "sys"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["max_tokens"] == 50


def test_openai_payload_keeps_system_inline():
    payload = OpenAIProvider(LLMSettings())._build_payload(_req())
    assert payload["messages"][0] == {"role": "system", "content": "sys"}


def test_openai_payload_tools():
    tools = [ToolSpec(name="search", description="d", parameters={"type": "object"})]
    payload = OpenAIProvider(LLMSettings())._build_payload(_req(tools=tools))
    assert payload["tools"][0]["function"]["name"] == "search"


def test_ollama_payload_json_format_when_schema():
    payload = OllamaProvider(LLMSettings())._build_payload(
        _req(response_schema={"type": "object"})
    )
    assert payload["format"] == "json"


def test_real_providers_need_sdk_or_key():
    # Without an API key configured, the Anthropic provider refuses to run rather
    # than emitting an obscure error mid-pipeline.
    from vds.core.errors import ConfigError as CE

    with pytest.raises(CE):
        AnthropicProvider(LLMSettings(api_key=None))._client()
