"""LLMClient — the provider-agnostic orchestration every agent calls.

Wraps an LLMProvider with the cross-cutting concerns that must not be duplicated
in each agent or each provider: retry-with-backoff on transient failures, timeout
surfacing, structured-output validation, and per-call logging (FR-7). Agents hand
it a Conversation and get back a validated response — they never touch a provider
or know which model answered.

No prompt engineering lives here (phase brief): the client transports messages;
it does not compose them.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from vds.agents.messages import (
    CompletionRequest,
    CompletionResponse,
    Conversation,
    ToolSpec,
)
from vds.agents.providers.base import LLMProvider
from vds.config.settings import LLMSettings
from vds.core.errors import AgentOutputError, TransientError
from vds.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class StructuredOutcome(Generic[T]):
    """A validated structured result plus the metadata callers need for metrics."""

    value: T
    response: CompletionResponse
    attempts: int  # provider calls made (>=1); attempts-1 == validation retries


class LLMClient:
    def __init__(
        self,
        provider: LLMProvider,
        config: LLMSettings,
        on_call: Callable[[CompletionRequest, CompletionResponse], None] | None = None,
    ) -> None:
        self._provider = provider
        self._config = config
        self._on_call = on_call  # sink for persistence (AgentLogRepo) — wired later

    def _request(self, conversation: Conversation, **overrides) -> CompletionRequest:
        return CompletionRequest(
            model=self._config.model,
            messages=conversation.messages,
            temperature=self._config.temperature,
            timeout_seconds=self._config.timeout_seconds,
            **overrides,
        )

    def complete(
        self,
        conversation: Conversation,
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int | None = None,
    ) -> CompletionResponse:
        request = self._request(
            conversation, tools=tools or [], max_tokens=max_tokens
        )
        return self._send(request)

    def structured(
        self,
        conversation: Conversation,
        schema: type[T],
        *,
        max_tokens: int | None = None,
    ) -> StructuredOutcome[T]:
        """Like complete_structured, but returns the outcome + metadata (response,
        attempts) so callers can record latency, tokens, and retry counts.

        Retries on invalid/unparseable output up to max_retries, then raises
        AgentOutputError. Provider-agnostic: works whether the provider returned
        JSON in `.structured` or in `.text`.
        """
        request = self._request(
            conversation,
            max_tokens=max_tokens,
            response_schema=schema.model_json_schema(),
        )
        last_error: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            response = self._send(request)
            raw = response.structured
            if raw is None:
                try:
                    raw = json.loads(response.text)
                except (json.JSONDecodeError, TypeError) as exc:
                    last_error = exc
                    log.warning("llm.structured_parse_failed", attempt=attempt)
                    continue
            try:
                value = schema.model_validate(raw)
                return StructuredOutcome(value=value, response=response, attempts=attempt + 1)
            except ValidationError as exc:
                last_error = exc
                log.warning("llm.structured_validation_failed", attempt=attempt)
        raise AgentOutputError(
            f"structured output did not match {schema.__name__} after "
            f"{self._config.max_retries + 1} attempts: {last_error}"
        )

    def complete_structured(
        self,
        conversation: Conversation,
        schema: type[T],
        *,
        max_tokens: int | None = None,
    ) -> T:
        """Return an instance of `schema`, validated from the model's output."""
        return self.structured(conversation, schema, max_tokens=max_tokens).value

    def _send(self, request: CompletionRequest) -> CompletionResponse:
        """One provider call with retry-on-transient + backoff + timing + logging."""
        last_error: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            start = time.perf_counter()
            try:
                response = self._provider.complete(request)
                response.latency_ms = round((time.perf_counter() - start) * 1000, 3)
                self._record(request, response)
                return response
            except TransientError as exc:
                last_error = exc
                log.warning(
                    "llm.transient_error", attempt=attempt, error=str(exc),
                    provider=getattr(self._provider, "name", "?"),
                )
                if attempt < self._config.max_retries:
                    self._backoff(attempt)
        raise last_error if last_error else RuntimeError("no attempts made")

    def _backoff(self, attempt: int) -> None:
        delay = self._config.retry_backoff_seconds * (2**attempt)
        if delay > 0:
            time.sleep(delay)

    def _record(self, request: CompletionRequest, response: CompletionResponse) -> None:
        log.info(
            "llm.call",
            provider=response.provider,
            model=response.model,
            latency_ms=response.latency_ms,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
        if self._on_call is not None:
            self._on_call(request, response)
