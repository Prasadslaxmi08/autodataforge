"""Echo provider — the framework's runnable, no-credentials default.

It performs no real inference: it echoes the last user message (or a canned
reply) and reports token counts. It exists so the Agent Framework runs and is
fully testable with no SDK or API key, and so provider *overhead* can be
benchmarked independently of model inference. Its failure/timeout knobs let tests
exercise the client's retry and timeout logic deterministically.
"""

from __future__ import annotations

from collections.abc import Iterator

from vds.agents.messages import (
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    Usage,
)
from vds.agents.providers.base import BaseProvider, ProviderError, ProviderTimeout


class EchoProvider(BaseProvider):
    name = "echo"
    supports_tools = True
    supports_streaming = True

    def __init__(
        self,
        config=None,
        *,
        reply: str | None = None,
        fail_times: int = 0,
        timeout: bool = False,
    ) -> None:
        super().__init__(config)
        self._reply = reply
        self._fail_times = fail_times
        self._timeout = timeout
        self.calls = 0  # visible to tests

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls += 1
        if self._timeout:
            raise ProviderTimeout("echo simulated timeout")
        if self.calls <= self._fail_times:
            raise ProviderError(f"echo simulated transient failure {self.calls}")

        last_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), ""
        )
        text = self._reply if self._reply is not None else f"echo: {last_user}"
        return CompletionResponse(
            text=text,
            model=request.model,
            provider=self.name,
            finish_reason="stop",
            usage=Usage(
                prompt_tokens=sum(len(m.content.split()) for m in request.messages),
                completion_tokens=len(text.split()),
            ),
        )

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        text = self._reply if self._reply is not None else f"echo: {request.messages[-1].content}"
        for token in text.split():
            yield StreamChunk(delta=token + " ")
        yield StreamChunk(done=True)
