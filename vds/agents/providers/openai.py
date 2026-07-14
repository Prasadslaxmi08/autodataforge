"""OpenAI provider adapter.

Pure payload translation is unit-tested; the network call runs only with the
`openai` SDK + API key present. Nothing OpenAI-specific leaks past this module.
"""

from __future__ import annotations

from vds.agents.messages import CompletionRequest, CompletionResponse, Usage
from vds.agents.providers.base import (
    BaseProvider,
    ProviderError,
    ProviderNotConfigured,
    ProviderTimeout,
)


class OpenAIProvider(BaseProvider):
    name = "openai"
    supports_tools = True
    supports_streaming = True

    @staticmethod
    def _msg(m) -> dict:
        # Text-only stays a plain string; images become the content-parts array.
        if not m.images:
            return {"role": m.role, "content": m.content}
        parts = [
            {"type": "image_url",
             "image_url": {"url": f"data:{img.media_type};base64,{img.data_base64}"}}
            for img in m.images
        ]
        if m.content:
            parts.append({"type": "text", "text": m.content})
        return {"role": m.role, "content": parts}

    def _build_payload(self, request: CompletionRequest) -> dict:
        # OpenAI keeps the system message inline in the message list.
        payload = {
            "model": request.model,
            "messages": [self._msg(m) for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.response_schema:
            payload["response_format"] = {"type": "json_object"}
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ]
        return payload

    def _client(self):
        try:
            import openai
        except ImportError as exc:
            raise ProviderNotConfigured(
                "OpenAIProvider needs the 'openai' package (pip install openai)."
            ) from exc
        if not self.config.api_key:
            raise ProviderNotConfigured("OpenAIProvider needs VDS_LLM__API_KEY set.")
        return openai.OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        client = self._client()
        try:
            resp = client.chat.completions.create(
                timeout=request.timeout_seconds, **self._build_payload(request)
            )
        except Exception as exc:
            name = type(exc).__name__.lower()
            if "timeout" in name:
                raise ProviderTimeout(str(exc)) from exc
            raise ProviderError(str(exc)) from exc
        choice = resp.choices[0]
        return CompletionResponse(
            text=choice.message.content or "",
            model=request.model,
            provider=self.name,
            finish_reason=choice.finish_reason or "stop",
            usage=Usage(
                prompt_tokens=getattr(resp.usage, "prompt_tokens", 0),
                completion_tokens=getattr(resp.usage, "completion_tokens", 0),
            ),
        )
