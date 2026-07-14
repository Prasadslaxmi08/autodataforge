"""Engineering Memory (Phase 10) — deterministic, versioned, explainable
long-term memory of engineering decisions. No RAG, no vector DB, no embeddings.

The platform learns from prior runs: the Planner recalls similar datasets, the
Analyst records validated knowledge after each execution, and trend/engineering
reports show what improved over time.
"""

from vds.memory.builder import build_memory
from vds.memory.schema import (
    AnalystConclusions,
    BenchmarkSummary,
    DatasetFingerprint,
    EngineeringMemory,
    ExecutionMetrics,
    MemoryRecommendation,
    PlannerDecisions,
    VerificationOutcomes,
)
from vds.memory.service import EngineeringMemoryService, MemoryGuidance
from vds.memory.similarity import MemoryMatch, SimilarityEngine, similarity
from vds.memory.store import MEMORY_PATH, MemoryStore
from vds.memory.trends import TrendAnalyzer

__all__ = [
    "AnalystConclusions",
    "BenchmarkSummary",
    "DatasetFingerprint",
    "EngineeringMemory",
    "EngineeringMemoryService",
    "ExecutionMetrics",
    "MEMORY_PATH",
    "MemoryGuidance",
    "MemoryMatch",
    "MemoryRecommendation",
    "MemoryStore",
    "PlannerDecisions",
    "SimilarityEngine",
    "TrendAnalyzer",
    "VerificationOutcomes",
    "build_memory",
    "similarity",
]
