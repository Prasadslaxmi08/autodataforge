# Agent Framework (Phase 6)

The provider-agnostic LLM foundation every future agent (Planner, Analyst, …)
builds on. No agent ever imports or names a provider; provider selection is
entirely configuration.

## Layers

```
Agent (base.py)            system prompt + conversation plumbing, NO reasoning/prompts
  └─ uses ─────────────────────────────────────────────────────────────┐
LLMClient (llm.py)         retries · timeout · structured-output · logging │ provider-agnostic
  └─ calls ──────────────────────────────────────────────────────────────┘
LLMProvider (providers/)   one adapter per backend, resolved by import path
  ├─ echo        (default, runnable, no credentials — tests & overhead bench)
  ├─ anthropic   (guarded SDK)
  ├─ openai      (guarded SDK)
  └─ ollama      (guarded SDK, on-prem / air-gapped)
messages.py                Conversation · Message · CompletionRequest/Response ·
                           ToolSpec/ToolCall · StreamChunk  (tool-calling & streaming
                           types defined now → future-ready, no schema change later)
```

## Key properties

- **Provider chosen by config only.** `VDS_LLM__PROVIDER=module:ClassName` (or the
  `[llm]` block in `vds.toml`) selects the backend; nothing above `providers/`
  branches on it. Verified: `test_agents_never_import_a_provider_sdk` asserts no
  vendor SDK is imported when the framework loads.
- **One adapter to add a provider.** Implement `complete()` (+ optional `stream`),
  point config at it — no existing code changes. Same import-path plugin pattern
  as the CV-model registry.
- **Errors map to the platform taxonomy.** `ProviderError`/`ProviderTimeout` are
  `TransientError`s (retried with backoff); `ProviderNotConfigured` is a
  `ConfigError` (fail fast). So the retry policy is uniform across providers.
- **Structured output is provider-agnostic.** `complete_structured(convo, Model)`
  requests JSON, parses, and validates into a Pydantic model, retrying on bad
  output and raising `AgentOutputError` after `max_retries`.
- **Future-ready without redesign:** tool-calling (`ToolSpec`/`ToolCall`) and
  streaming (`StreamChunk`, `stream()`) types and hooks exist; MCP, multi-agent
  messaging, memory, and RAG attach at the Agent layer above this client.

## Measured framework overhead

From `scripts/provider_overhead.py` (Echo provider, ~0 inference, 2000 calls):

| Path | Overhead per call |
|---|---|
| `complete` | ~0.03 ms |
| `complete_structured` (parse + validate) | ~0.10 ms |
| structured validation delta | ~0.07 ms |

Framework overhead is ~tens of microseconds — negligible against real LLM
latency (10²–10³ ms). The comparison framework can therefore attribute future
latency to inference, not plumbing.

## Remaining work before the Planner Agent

1. **Real inference credentials/SDKs** for at least one provider (install
   `anthropic`/`openai`/`ollama`, set `VDS_LLM__API_KEY`); the adapter network
   paths exist but are unexercised here by design.
2. **Persist agent calls** — wire `LLMClient.on_call` to `AgentLogRepo` (FR-7) so
   every LLM call is auditable in the DB, not only logged.
3. **Prompt layer** — the Planner's system prompt and structured schemas
   (deliberately absent here: this phase ships no prompts).
4. **Vision input** — extend `Message.content` for image parts so the Verifier's
   VLM path can use this same client.

None of these require changing the framework's interfaces.
