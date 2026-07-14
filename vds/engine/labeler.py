"""Labeling Engine (Phase 1).

Executes a ProcessingPlan mechanically: for each image, run the detector, drop
detections below the plan's confidence floor, segment each surviving detection,
and write an Annotation (box + mask + provenance). Deterministic — it invents no
strategy (System Design §2.6). Model calls are timed for benchmarking.
"""

from __future__ import annotations

import uuid

from vds.benchmark import BenchmarkCollector
from vds.core.contracts import Annotation, Detection, ImageId, ProcessingPlan, Provenance
from vds.core.enums import AnnotationState
from vds.core.geometry import overlap_iou
from vds.logging import get_logger
from vds.models.protocols import Capability
from vds.models.registry import ModelRegistry
from vds.store.cas import Cas
from vds.store.sqlite import AnnotationRepo, ImageRepo

log = get_logger(__name__)

_NMS_IOU = 0.9  # boxes overlapping more than this are treated as duplicates


def _suppress_overlaps(detections: list[Detection]) -> list[Detection]:
    """Deterministic greedy NMS: keep higher-confidence boxes, drop near-duplicates."""
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(overlap_iou(det.box, k.box) <= _NMS_IOU for k in kept):
            kept.append(det)
    return kept


class LabelingEngine:
    def __init__(
        self,
        registry: ModelRegistry,
        cas: Cas,
        images: ImageRepo,
        annotations: AnnotationRepo,
    ) -> None:
        self._registry = registry
        self._cas = cas
        self._images = images
        self._annotations = annotations

    def label(
        self, plan: ProcessingPlan, image_ids: list[ImageId], bench: BenchmarkCollector
    ) -> int:
        """Detect + segment every image; return the number of annotations written."""
        import time

        detector = self._registry.get(Capability.DETECTOR)
        segmenter = self._registry.get(Capability.SEGMENTER)
        written = 0

        for image_id in image_ids:
            record = self._images.get(image_id)
            if record is None:
                continue
            data = self._cas.get(record.sha256)

            with bench.stage("detection"):
                t0 = time.perf_counter()
                detections = detector.detect(
                    [data], plan.classes, {"confidence_threshold": plan.confidence_threshold}
                )[0]
                bench.record_inference(time.perf_counter() - t0)

            detections = _suppress_overlaps(detections)
            for det in detections:
                if det.confidence < plan.confidence_threshold:
                    continue
                with bench.stage("segmentation"):
                    t0 = time.perf_counter()
                    mask = segmenter.segment(data, [det.box])
                    bench.record_inference(time.perf_counter() - t0)

                self._annotations.add(
                    Annotation(
                        id=uuid.uuid4().hex,
                        image_id=image_id,
                        label=det.label,
                        geometry=det.box,
                        confidence=det.confidence,
                        state=AnnotationState.LABELED,
                        mask=mask,
                        provenance=Provenance(
                            source="engine.labeling",
                            model=plan.detector,
                            plan_version=plan.version,
                        ),
                    )
                )
                written += 1

        log.info("engine.labeled", images=len(image_ids), annotations=written)
        return written
