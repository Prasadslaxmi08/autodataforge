"""Agent base + container wiring tests."""

from __future__ import annotations

from pydantic import BaseModel

from vds.agents.base import Agent
from vds.agents.llm import LLMClient
from vds.agents.providers.echo import EchoProvider
from vds.config.settings import LLMSettings, Settings
from vds.container import Container

FAST = LLMSettings(model="m", retry_backoff_seconds=0.0)


def test_agent_seeds_system_prompt():
    class Greeter(Agent):
        system_prompt = "you are a test agent"

    agent = Greeter(LLMClient(EchoProvider(), FAST))
    convo = agent.new_conversation()
    assert convo.messages[0].role == "system"
    assert convo.messages[0].content == "you are a test agent"


def test_agent_complete_delegates_to_client():
    agent = Agent(LLMClient(EchoProvider(), FAST), system_prompt="")
    resp = agent.complete("ping")
    assert resp.text == "echo: ping"


def test_agent_structured():
    class Out(BaseModel):
        ok: bool

    agent = Agent(LLMClient(EchoProvider(reply='{"ok": true}'), FAST))
    assert agent.complete_structured("x", Out).ok is True


def test_container_wires_default_echo_provider():
    from vds.agents.messages import Conversation

    c = Container(settings=Settings(), db_path=":memory:")
    assert c.llm_provider.name == "echo"
    resp = c.llm_client.complete(Conversation().user("hey"))
    assert resp.text == "echo: hey"


def test_agents_never_import_a_provider_sdk():
    # The framework must not hardcode a provider: importing the agent framework
    # pulls in no vendor SDK.
    import sys

    import vds.agents  # noqa: F401

    assert "anthropic" not in sys.modules
    assert "openai" not in sys.modules
