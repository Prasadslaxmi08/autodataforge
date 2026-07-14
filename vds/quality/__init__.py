"""L2 service — dataset metrics (deterministic).

Responsibility: compute the numbers the Analyst interprets — class balance,
per-class verification pass rate, mixed-label embedding clusters, split leakage,
geometry stats, anomaly flags.

Bootstrap scope: the service interface. Metric computation is Phase 1.
"""

from __future__ import annotations

from typing import Protocol

from vds.core.contracts import ProjectId, QualityMetrics


class QualityService(Protocol):
    def compute(self, project_id: ProjectId) -> QualityMetrics: ...
