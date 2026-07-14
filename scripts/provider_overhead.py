"""Benchmark the Agent Framework's overhead, independent of model inference.

Runs N completions through the Echo provider (whose "inference" is ~0), so the
measured time is pure framework cost: request construction, the retry loop,
logging, and — for the structured path — JSON parsing + Pydantic validation.
This isolates framework overhead from real LLM latency, which dominates in
production and would otherwise mask it.

Run:  python scripts/provider_overhead.py
Writes benchmarks/provider_overhead.json.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from vds.agents.llm import LLMClient
from vds.agents.messages import Conversation
from vds.agents.providers.echo import EchoProvider
from vds.config.settings import LLMSettings

N = 2000


class _Schema(BaseModel):
    classes: list[str]
    count: int


def _bench(label: str, fn, n: int) -> float:
    fn()  # warm up
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return round((time.perf_counter() - start) / n * 1000, 5)  # ms per call


def main() -> None:
    config = LLMSettings(model="echo-model")
    plain = LLMClient(EchoProvider(), config)
    structured = LLMClient(EchoProvider(reply='{"classes": ["a", "b"], "count": 2}'), config)
    convo = Conversation().system("s").user("hello world")

    plain_ms = _bench("complete", lambda: plain.complete(convo), N)
    struct_ms = _bench(
        "complete_structured",
        lambda: structured.complete_structured(convo, _Schema),
        N,
    )

    result = {
        "calls_per_measurement": N,
        "complete_overhead_ms": plain_ms,
        "structured_overhead_ms": struct_ms,
        "validation_overhead_ms": round(struct_ms - plain_ms, 5),
        "created_at": datetime.now(UTC).isoformat(),
        "note": "Echo provider ~0 inference; times are framework overhead only.",
    }
    out = Path("benchmarks") / "provider_overhead.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
