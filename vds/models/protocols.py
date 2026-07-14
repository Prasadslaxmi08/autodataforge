"""Model capability protocols (System Design §5.1).

These structural protocols are the *stable* surface of the plugin system. New
models implement them; no module above L1 ever changes when a model is added or
swapped (open/closed). Callers receive only these types, never model-specific
ones.

Bootstrap scope: signatures only. Real inference lands with each adapter in
Phase 1; the FakeAdapter satisfies every protocol today.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vds.core.contracts import Box2D, Detection, Mask


@runtime_checkable
class ModelAdapter(Protocol):
    """Common lifecycle every adapter exposes to the GpuManager."""

    name: str
    capabilities: frozenset[str]
    vram_estimate_mb: int

    def load(self) -> None: ...
    def unload(self) -> None: ...


class Detector(Protocol):
    def detect(
        self, images: list[bytes], prompts: list[str], params: dict[str, Any]
    ) -> list[list[Detection]]:
        """One list of structured detections (box + label + confidence) per image."""
        ...


class Segmenter(Protocol):
    def segment(self, image: bytes, prompts: list[Box2D | tuple[float, float]]) -> Mask:
        ...


class Embedder(Protocol):
    def embed(self, images: list[bytes]) -> list[list[float]]: ...


class Classifier(Protocol):
    def classify(self, image: bytes, labels: list[str]) -> dict[str, float]: ...


class VisionJudge(Protocol):
    def judge(
        self, crop: bytes, question: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        ...


class TextLLM(Protocol):
    def complete(
        self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None
    ) -> Any:
        ...


class Capability:
    """String keys mapping a capability to a config field / registry slot."""

    DETECTOR = "detector"
    SEGMENTER = "segmenter"
    EMBEDDER = "embedder"
    CLASSIFIER = "classifier"
    VISION_JUDGE = "vision_judge"
    TEXT_LLM = "text_llm"
