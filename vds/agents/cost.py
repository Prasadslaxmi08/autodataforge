"""LLM cost estimation (Phase 7).

A small static price table keyed by model-name substring. Returns None for local
or unknown models (Ollama/Echo are free), so cost is reported only "where
supported" as the phase asks.
"""

from __future__ import annotations

from vds.agents.messages import Usage

# USD per 1K tokens: (input, output). Public list prices; update as they change.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus": (0.015, 0.075),
    "claude-sonnet": (0.003, 0.015),
    "claude-haiku": (0.0008, 0.004),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
}


def estimate_cost(model: str, usage: Usage) -> float | None:
    key = model.lower()
    for name, (inp, out) in PRICING.items():
        if name in key:
            return round(
                usage.prompt_tokens / 1000 * inp + usage.completion_tokens / 1000 * out,
                6,
            )
    return None  # local / unknown model -> not billed
