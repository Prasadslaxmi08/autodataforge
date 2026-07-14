"""FakeAdapter — the framework's null model.

It implements every capability protocol with empty/deterministic results and
zero VRAM. It exists so that:

  * the app starts and the registry resolves with no weights installed,
  * the full pipeline is testable on CPU in CI (System Design §9),
  * the plugin mechanism is demonstrably wired end to end.

This is framework scaffolding, not an AI model — no real detection, segmentation
or reasoning happens here. Real adapters (GroundingDINO, SAM2, ...) arrive in
Phase 1 alongside the engines that use them.
"""

from __future__ import annotations

from typing import Any

from vds.core.contracts import Box2D, Detection, Mask
from vds.models.protocols import Capability


class FakeAdapter:
    name = "fake"
    capabilities = frozenset(
        {
            Capability.DETECTOR,
            Capability.SEGMENTER,
            Capability.EMBEDDER,
            Capability.CLASSIFIER,
            Capability.VISION_JUDGE,
            Capability.TEXT_LLM,
        }
    )
    vram_estimate_mb = 0

    def load(self) -> None:  # nothing to load
        pass

    def unload(self) -> None:
        pass

    # --- capability protocols (empty, deterministic) ---
    def detect(
        self, images: list[bytes], prompts: list[str], params: dict[str, Any]
    ) -> list[list[Detection]]:
        return [[] for _ in images]

    def segment(self, image: bytes, prompts: list[Box2D | tuple[float, float]]) -> Mask:
        return Mask(rle="", height=0, width=0)

    def embed(self, images: list[bytes]) -> list[list[float]]:
        return [[] for _ in images]

    def classify(self, image: bytes, labels: list[str]) -> dict[str, float]:
        return {label: 0.0 for label in labels}

    def judge(self, crop: bytes, question: str, schema: dict[str, Any]) -> dict[str, Any]:
        return {}

    def complete(
        self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None
    ) -> Any:
        return {} if schema else ""
