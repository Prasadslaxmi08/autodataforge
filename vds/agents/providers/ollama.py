"""Ollama provider adapter (local / air-gapped inference).

Pure payload translation is unit-tested; the network call runs only with the
`ollama` package present and a reachable local daemon. This is the on-prem path
(NFR-1) — no API key, a local base_url.
"""

from __future__ import annotations

from vds.agents.messages import CompletionRequest, CompletionResponse, Usage
from vds.agents.providers.base import (
    BaseProvider,
    ProviderError,
    ProviderNotConfigured,
    ProviderTimeout,
)


class OllamaProvider(BaseProvider):
    name = "ollama"
    supports_tools = False
    supports_streaming = True

    def _build_payload(self, request: CompletionRequest) -> dict:
        return {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "options": {"temperature": request.temperature},
            "format": "json" if request.response_schema else "",
        }

    def _client(self):
        try:
            import ollama
        except ImportError as exc:
            raise ProviderNotConfigured(
                "OllamaProvider needs the 'ollama' package (pip install ollama)."
            ) from exc
        return ollama.Client(host=self.config.base_url or "http://localhost:11434")

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        client = self._client()
        try:
            resp = client.chat(**self._build_payload(request))
        except Exception as exc:
            name = type(exc).__name__.lower()
            if "timeout" in name:
                raise ProviderTimeout(str(exc)) from exc
            raise ProviderError(str(exc)) from exc
        return CompletionResponse(
            text=resp["message"]["content"],
            model=request.model,
            provider=self.name,
            finish_reason=resp.get("done_reason", "stop"),
            usage=Usage(
                prompt_tokens=resp.get("prompt_eval_count", 0),
                completion_tokens=resp.get("eval_count", 0),
            ),
        )
