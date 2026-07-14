"""Bootstrap smoke tests — the framework logic that has real behaviour today.

Business logic (ingest, labeling, agents) is Phase 1+ and untested here by design.
These cover: the state-machine transition table, the CAS round-trip + integrity
guard, the model-plugin loader, config loading, and app startup.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vds.api.app import create_app
from vds.config.settings import get_settings
from vds.core.enums import (
    AnnotationState,
    ProjectPhase,
    assert_transition,
    is_legal_transition,
)
from vds.core.errors import ConfigError, IllegalTransitionError, IntegrityError
from vds.models.gpu import GpuManager
from vds.models.registry import ModelRegistry, load_adapter
from vds.store.cas import LocalCas


# --- state machine ---
def test_legal_transition_allowed():
    assert is_legal_transition(AnnotationState.LABELED, AnnotationState.VERIFIED)
    assert_transition(ProjectPhase.CREATED, ProjectPhase.INGESTING)  # no raise


def test_illegal_transition_raises():
    with pytest.raises(IllegalTransitionError):
        assert_transition(AnnotationState.LABELED, AnnotationState.ACCEPTED)


def test_terminal_state_has_no_exits():
    assert not is_legal_transition(AnnotationState.REJECTED, AnnotationState.LABELED)


# --- CAS ---
def test_cas_roundtrip_and_dedup(tmp_path: Path):
    cas = LocalCas(tmp_path)
    sha = cas.put(b"hello")
    assert cas.exists(sha)
    assert cas.get(sha) == b"hello"
    assert cas.put(b"hello") == sha  # content-addressed -> same key


def test_cas_detects_corruption(tmp_path: Path):
    cas = LocalCas(tmp_path)
    sha = cas.put(b"data")
    cas.path(sha).write_bytes(b"tampered")
    with pytest.raises(IntegrityError):
        cas.get(sha)


# --- plugin loader ---
def test_load_fake_adapter():
    adapter = load_adapter("vds.models.adapters.fake:FakeAdapter")
    assert adapter.name == "fake"


def test_load_adapter_bad_path_raises():
    with pytest.raises(ConfigError):
        load_adapter("not_a_valid_path")
    with pytest.raises(ConfigError):
        load_adapter("vds.models.adapters.fake:DoesNotExist")


def test_registry_resolves_capability():
    settings = get_settings()
    registry = ModelRegistry(settings.models, GpuManager(settings.gpu.vram_budget_mb))
    detector = registry.get("detector")
    # A blank image yields a valid (empty) detection list -> the plugin resolved.
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (255, 255, 255)).save(buf, format="PNG")
    assert detector.detect([buf.getvalue()], ["cat"], {}) == [[]]


# --- gpu budget ---
def test_gpu_evicts_when_over_budget():
    class Big:
        def __init__(self, name):
            self.name, self.vram_estimate_mb = name, 6000
            self.loaded = False

        def load(self):
            self.loaded = True

        def unload(self):
            self.loaded = False

    gpu = GpuManager(vram_budget_mb=8192)
    a, b = Big("a"), Big("b")
    gpu.ensure_loaded(a)
    gpu.ensure_loaded(b)  # a+b = 12000 > 8192 -> a evicted
    assert not a.loaded and b.loaded


# --- config + app ---
def test_settings_load():
    s = get_settings(reload=True)
    assert s.gpu.vram_budget_mb == 8192


def test_app_starts_and_reports_health():
    # `with` triggers the lifespan that builds and attaches the container.
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert "models" in client.get("/info").json()
