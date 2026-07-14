"""Tests for the VLM (multimodal) verifier and its multimodal message plumbing."""

from __future__ import annotations

import io
import json

from vds.agents.llm import LLMClient
from vds.agents.messages import CompletionResponse, Conversation, ImageContent, Usage
from vds.agents.providers.anthropic import AnthropicProvider
from vds.agents.providers.base import BaseProvider
from vds.agents.providers.echo import EchoProvider
from vds.agents.providers.openai import OpenAIProvider
from vds.agents.vlm_verifier import LLMVerifier, build_evidence
from vds.config.settings import LLMSettings
from vds.core.contracts import Annotation, Box2D, Mask, Provenance


def _png(w=80, h=60) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 200, 200)).save(buf, format="PNG")
    return buf.getvalue()


def _ann(id: str, conf: float, box: Box2D, mask: Mask | None = None) -> Annotation:
    return Annotation(id=id, image_id="i", label="object", geometry=box, confidence=conf,
                      state="labeled", provenance=Provenance(source="engine.labeling"), mask=mask)


class _StubVLM(BaseProvider):
    """Returns a valid VLMSceneVerdict; one correction cites a bogus key to prove
    evidence enforcement drops it."""

    name = "stub-vlm"

    def complete(self, request) -> CompletionResponse:
        out = {
            "annotation_verdicts": [
                {"annotation_index": 0, "decision": "approved", "object_identity_correct": True,
                 "label_correct": True, "bbox_quality": "good", "segmentation_quality": "good",
                 "confidence": 0.9, "reasoning": "clean", "cited_evidence": ["a0_conf"], "corrections": []},
                {"annotation_index": 1, "decision": "rejected", "object_identity_correct": False,
                 "label_correct": True, "bbox_quality": "good", "segmentation_quality": "good",
                 "confidence": 0.8, "reasoning": "nothing there",
                 "cited_evidence": ["a1_conf"],
                 "corrections": [
                     {"issue": "hallucination", "suggested_correction": "reject", "reasoning": "empty region",
                      "confidence": 0.8, "expected_impact": "cleaner set", "cited_evidence": ["a1_conf"]},
                     {"issue": "made up", "suggested_correction": "x", "reasoning": "y",
                      "confidence": 0.5, "expected_impact": "z", "cited_evidence": ["nonexistent_key"]},
                 ]},
            ],
            "scene_findings": [],
            "summary": "ok",
        }
        return CompletionResponse(text=json.dumps(out), model="stub-vlm", provider=self.name,
                                  usage=Usage(prompt_tokens=100, completion_tokens=40))


def _client(provider) -> LLMClient:
    return LLMClient(provider, LLMSettings(model=provider.name, max_retries=0))


def test_echo_provider_falls_back_to_deterministic():
    # Echo can't see images and echoes text -> JSON parse fails -> fallback.
    v = LLMVerifier(_client(EchoProvider(LLMSettings())))
    res = v.verify_scene(_png(), [_ann("a", 0.95, Box2D(x=0, y=0, w=10, h=10))])
    assert res.source == "deterministic"
    assert res.fallback_reason is not None
    assert len(res.verdicts) == 1
    assert res.verdicts[0].verdict == "correct"  # high conf -> approved by the fallback


def test_vlm_path_maps_verdicts_and_enforces_evidence():
    v = LLMVerifier(_client(_StubVLM(LLMSettings())))
    anns = [_ann("a0", 0.9, Box2D(x=0, y=0, w=10, h=10)),
            _ann("a1", 0.9, Box2D(x=40, y=0, w=10, h=10))]
    res = v.verify_scene(_png(), anns)
    assert res.source == "vlm"
    assert res.verdicts[0].verdict == "correct"
    assert res.verdicts[1].verdict == "hallucination"  # rejected + identity wrong
    # The bogus-citation correction was dropped; the real one survived.
    corr = res.detail.annotation_verdicts[1].corrections
    assert len(corr) == 1 and corr[0].cited_evidence == ["a1_conf"]
    assert "made up" in res.dropped_recommendations
    assert res.evidence_coverage < 1.0


def test_verify_single_annotation_protocol():
    v = LLMVerifier(_client(_StubVLM(LLMSettings())))
    verdict = v.verify(_png(), _ann("a0", 0.9, Box2D(x=0, y=0, w=10, h=10)))
    assert verdict.annotation_id == "a0"
    assert verdict.verdict == "correct"


def test_missing_verdict_defaults_to_review():
    v = LLMVerifier(_client(_StubVLM(LLMSettings())))  # stub only returns indices 0,1
    anns = [_ann("a0", 0.9, Box2D(x=0, y=0, w=10, h=10)),
            _ann("a1", 0.9, Box2D(x=40, y=0, w=10, h=10)),
            _ann("a2", 0.9, Box2D(x=60, y=0, w=10, h=10))]  # no verdict returned for #2
    res = v.verify_scene(_png(), anns)
    assert res.verdicts[2].verdict == "bad_geometry"  # NEEDS_REVIEW, never silently approved


def test_bad_image_bytes_fall_back():
    v = LLMVerifier(_client(_StubVLM(LLMSettings())))
    res = v.verify_scene(b"not a png", [_ann("a", 0.95, Box2D(x=0, y=0, w=10, h=10))])
    assert res.source == "deterministic"


def test_evidence_pack_flags_duplicates_and_small_objects():
    anns = [_ann("a0", 0.9, Box2D(x=0, y=0, w=50, h=50)),
            _ann("a1", 0.8, Box2D(x=1, y=1, w=50, h=50)),  # overlaps a0
            _ann("a2", 0.7, Box2D(x=0, y=0, w=2, h=2))]     # tiny
    keys = {f.key for f in build_evidence(anns, 200, 200)}
    assert "dup_0_1" in keys
    assert "a2_small" in keys


def test_anthropic_payload_carries_image_block():
    convo = Conversation().user("look", images=[ImageContent(data_base64="ABC")])
    from vds.agents.messages import CompletionRequest

    payload = AnthropicProvider(LLMSettings())._build_payload(
        CompletionRequest(model="m", messages=convo.messages))
    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image"
    assert content[0]["source"]["data"] == "ABC"


def test_openai_payload_carries_image_url():
    convo = Conversation().user("look", images=[ImageContent(data_base64="ABC")])
    from vds.agents.messages import CompletionRequest

    payload = OpenAIProvider(LLMSettings())._build_payload(
        CompletionRequest(model="m", messages=convo.messages))
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,ABC")
