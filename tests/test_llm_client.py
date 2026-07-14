"""LLMClient tests: retries, timeout, structured output, logging hook."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from vds.agents.llm import LLMClient
from vds.agents.messages import Conversation
from vds.agents.providers.echo import EchoProvider
from vds.config.settings import LLMSettings
from vds.core.errors import AgentOutputError

# max_retries high, backoff 0 so retry tests are fast and deterministic.
FAST = LLMSettings(max_retries=3, retry_backoff_seconds=0.0, model="m")


def _convo(text="hi") -> Conversation:
    return Conversation().system("s").user(text)


def test_complete_returns_response():
    client = LLMClient(EchoProvider(), FAST)
    resp = client.complete(_convo("hello"))
    assert resp.text == "echo: hello"
    assert resp.latency_ms >= 0


def test_retry_succeeds_after_transient_failures():
    provider = EchoProvider(fail_times=2)  # fail twice, then succeed
    resp = LLMClient(provider, FAST).complete(_convo())
    assert resp.text.startswith("echo:")
    assert provider.calls == 3


def test_retry_exhausted_raises():
    provider = EchoProvider(fail_times=99)
    with pytest.raises(Exception) as exc:  # ProviderError (TransientError)
        LLMClient(provider, LLMSettings(max_retries=2, retry_backoff_seconds=0.0)).complete(
            _convo()
        )
    assert "transient" in str(exc.value)
    assert provider.calls == 3  # initial + 2 retries


def test_timeout_is_retried_then_raised():
    provider = EchoProvider(timeout=True)
    from vds.agents.providers.base import ProviderTimeout

    with pytest.raises(ProviderTimeout):
        LLMClient(provider, FAST).complete(_convo())
    assert provider.calls == 4  # initial + 3 retries


# --- structured output ---
class Ontology(BaseModel):
    classes: list[str]
    count: int


def test_structured_output_validates():
    provider = EchoProvider(reply='{"classes": ["drone", "bird"], "count": 2}')
    result = LLMClient(provider, FAST).complete_structured(_convo(), Ontology)
    assert isinstance(result, Ontology)
    assert result.classes == ["drone", "bird"] and result.count == 2


def test_structured_output_invalid_json_raises():
    provider = EchoProvider(reply="not json at all")
    with pytest.raises(AgentOutputError):
        LLMClient(provider, FAST).complete_structured(_convo(), Ontology)


def test_structured_output_schema_mismatch_raises():
    provider = EchoProvider(reply='{"classes": "should-be-a-list"}')
    with pytest.raises(AgentOutputError):
        LLMClient(provider, FAST).complete_structured(_convo(), Ontology)


def test_on_call_hook_records():
    calls = []
    client = LLMClient(EchoProvider(), FAST, on_call=lambda req, resp: calls.append((req, resp)))
    client.complete(_convo())
    assert len(calls) == 1
    assert calls[0][1].provider == "echo"
