"""Anthropic provider adapter.

Payload translation (`_build_payload`) is pure and unit-tested; the network call
in `complete()` runs only when the `anthropic` SDK and an API key are present
(production inference is Phase 2 — the phase brief does not require it here).
Nothing about Anthropic leaks outside this module.
"""

from __future__ import annotations

from vds.agents.messages import CompletionRequest, CompletionResponse, Usage
from vds.agents.providers.base import (
    BaseProvider,
    ProviderError,
    ProviderNotConfigured,
    ProviderTimeout,
)


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    supports_tools = True
    supports_streaming = True

    def _build_payload(self, request: CompletionRequest) -> dict:
        # Anthropic takes `system` separately from the message list.
        system = "\n".join(m.content for m in request.messages if m.role == "system")
        messages = []
        for m in request.messages:
            if m.role not in ("user", "assistant"):
                continue
            if m.images:
                blocks = [
                    {"type": "image", "source": {"type": "base64",
                     "media_type": img.media_type, "data": img.data_base64}}
                    for img in m.images
                ]
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                messages.append({"role": m.role, "content": blocks})
            else:
                messages.append({"role": m.role, "content": m.content})
        payload = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,
            "temperature": request.temperature,
        }
        if system:
            payload["system"] = system
        if request.tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in request.tools
            ]
        return payload

    def _client(self):
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderNotConfigured(
                "AnthropicProvider needs the 'anthropic' package (pip install anthropic)."
            ) from exc
        if not self.config.api_key:
            raise ProviderNotConfigured("AnthropicProvider needs VDS_LLM__API_KEY set.")
        return anthropic.Anthropic(api_key=self.config.api_key)

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        client = self._client()
        try:
            resp = client.messages.create(
                timeout=request.timeout_seconds, **self._build_payload(request)
            )
        except Exception as exc:  # SDK exceptions -> taxonomy
            name = type(exc).__name__.lower()
            if "timeout" in name:
                raise ProviderTimeout(str(exc)) from exc
            raise ProviderError(str(exc)) from exc
        text = "".join(block.text for block in resp.content if block.type == "text")
        return CompletionResponse(
            text=text,
            model=request.model,
            provider=self.name,
            finish_reason=getattr(resp, "stop_reason", "stop"),
            usage=Usage(
                prompt_tokens=getattr(resp.usage, "input_tokens", 0),
                completion_tokens=getattr(resp.usage, "output_tokens", 0),
            ),
        )
