"""Deterministic similarity engine (Phase 10).

No embeddings, no vector DB (phase rules). Similarity is a weighted average of
per-feature agreement between two DatasetFingerprints. Each numeric feature scores
1.0 for identical, decaying to 0.0 as they diverge (ratios by absolute gap;
magnitudes by relative gap). A feature that is `unknown` (sentinel) in the *query*
is skipped, so a Planner's pre-run query matches on the subset it actually knows.

Every match carries a human-readable explanation of *why* it matched — the
similarity is fully auditable, never a black-box distance.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from vds.memory.schema import DatasetFingerprint, EngineeringMemory

# feature name -> (weight, kind). kind "ratio" scores by absolute gap on a 0..1
# scale; "magnitude" scores by relative gap (scale-free).
_FEATURES: dict[str, tuple[float, str]] = {
    "resolution_mp": (1.5, "magnitude"),
    "dataset_size": (1.0, "magnitude"),
    "scene_density": (1.5, "magnitude"),
    "object_density": (1.0, "magnitude"),
    "small_object_ratio": (1.5, "ratio"),
    "duplicate_ratio": (1.0, "ratio"),
    "avg_confidence": (1.0, "ratio"),
}
_SCENE_TYPE_WEIGHT = 2.0  # categorical scene-type agreement


def _known(v: float) -> bool:
    return v is not None and v >= 0.0


def _feature_score(a: float, b: float, kind: str) -> float:
    if kind == "ratio":
        return max(0.0, 1.0 - abs(a - b))  # both already in 0..1
    denom = max(abs(a), abs(b), 1e-9)
    return max(0.0, 1.0 - abs(a - b) / denom)  # relative gap


class FeatureMatch(BaseModel):
    feature: str
    query: float | str
    candidate: float | str
    score: float
    weight: float


class MemoryMatch(BaseModel):
    memory: EngineeringMemory
    score: float  # 0..1 overall similarity
    reasons: list[FeatureMatch] = Field(default_factory=list)

    def explain(self) -> str:
        top = sorted(self.reasons, key=lambda r: r.score * r.weight, reverse=True)[:4]
        bits = [f"{r.feature}({r.query}≈{r.candidate}, {r.score:.2f})" for r in top]
        return f"score {self.score:.2f} — matched on " + ", ".join(bits)


def similarity(query: DatasetFingerprint, cand: DatasetFingerprint) -> tuple[float, list[FeatureMatch]]:
    """Weighted similarity in [0,1] plus the per-feature breakdown. Only features
    known in the query contribute (weights renormalize over what's present)."""
    reasons: list[FeatureMatch] = []
    total_w = 0.0
    acc = 0.0
    for name, (weight, kind) in _FEATURES.items():
        qv, cv = getattr(query, name), getattr(cand, name)
        if not (_known(qv) and _known(cv)):
            continue
        s = _feature_score(qv, cv, kind)
        reasons.append(FeatureMatch(feature=name, query=round(qv, 4), candidate=round(cv, 4),
                                    score=round(s, 4), weight=weight))
        acc += s * weight
        total_w += weight

    if query.scene_type != "unknown" and cand.scene_type != "unknown":
        s = 1.0 if query.scene_type == cand.scene_type else 0.0
        reasons.append(FeatureMatch(feature="scene_type", query=query.scene_type,
                                    candidate=cand.scene_type, score=s, weight=_SCENE_TYPE_WEIGHT))
        acc += s * _SCENE_TYPE_WEIGHT
        total_w += _SCENE_TYPE_WEIGHT

    score = round(acc / total_w, 4) if total_w else 0.0
    return score, reasons


class SimilarityEngine:
    def __init__(self, min_score: float = 0.5) -> None:
        self._min = min_score

    def search(
        self, query: DatasetFingerprint, memories: list[EngineeringMemory], top_k: int = 3
    ) -> list[MemoryMatch]:
        """Most-similar memories first, above the relevance floor. Deterministic:
        ties break on newest-then-id so results are stable across runs."""
        scored: list[MemoryMatch] = []
        for m in memories:
            score, reasons = similarity(query, m.dataset_fingerprint)
            if score >= self._min:
                scored.append(MemoryMatch(memory=m, score=score, reasons=reasons))
        scored.sort(key=lambda x: (x.score, x.memory.created_at, x.memory.id), reverse=True)
        return scored[:top_k]
