"""Deterministic dataset-quality and error analysis (Phase-5 baseline).

This is the metric layer the Analyst Agent will later interpret (Phase 3). It
computes — it does not reason. Every number is a direct count or statistic over
the produced annotations; every error category is an explicitly-labelled
heuristic proxy, because the baseline has no ground-truth labels to check against.
"""

from __future__ import annotations

from vds.core.contracts import (
    Box2D,
    DatasetQualityReport,
    ErrorAnalysis,
    ErrorCategory,
    ProjectId,
)
from vds.core.enums import AnnotationState
from vds.core.geometry import mask_is_empty, overlap_iou
from vds.store.sqlite import AnnotationRepo, ImageRepo

_LOW_CONF = 0.5
_SMALL_AREA_FRAC = 0.02  # object smaller than 2% of the image => "small object"
_CROWDED = 8  # more than this many detections => "crowded scene"
_DUP_IOU = 0.9

# Failure modes that cannot be measured without reference labels. Named
# explicitly so the baseline never pretends to measure them.
_UNMEASURABLE = [
    "missed detections (needs ground truth)",
    "incorrect segmentation (needs ground truth)",
    "difficult lighting (needs labels/metadata)",
    "occlusion (needs ground truth)",
]


class QualityAnalyzer:
    def __init__(self, images: ImageRepo, annotations: AnnotationRepo) -> None:
        self._images = images
        self._annotations = annotations

    def _gather(self, project_id: ProjectId):
        images = self._images.by_project(project_id)
        per_image = {img.id: self._annotations.by_image(img.id) for img in images}
        return images, per_image

    def quality(self, project_id: ProjectId) -> DatasetQualityReport:
        images, per_image = self._gather(project_id)
        anns = [a for anns in per_image.values() for a in anns]
        total = len(anns)

        def rate(state: AnnotationState) -> float:
            return sum(1 for a in anns if a.state == state) / total if total else 0.0

        rejected = sum(
            1
            for a in anns
            if a.state in (AnnotationState.REJECTED_AUTO, AnnotationState.REJECTED)
        )
        empty = sum(1 for a in anns if a.mask is not None and mask_is_empty(a.mask.rle))
        invalid = sum(
            1
            for a in anns
            if isinstance(a.geometry, Box2D)
            and (a.geometry.w <= 0 or a.geometry.h <= 0)
        )
        duplicates = self._count_duplicates(per_image)
        confidences = [a.confidence for a in anns]

        return DatasetQualityReport(
            project_id=project_id,
            images=len(images),
            detections=total,
            masks=sum(1 for a in anns if a.mask is not None),
            approval_rate=round(rate(AnnotationState.AUTO_ACCEPTED), 4),
            review_rate=round(rate(AnnotationState.NEEDS_REVIEW), 4),
            rejection_rate=round(rejected / total, 4) if total else 0.0,
            invalid_annotations=invalid,
            duplicate_detections=duplicates,
            empty_masks=empty,
            annotation_density=round(total / len(images), 4) if images else 0.0,
            avg_confidence=round(sum(confidences) / total, 4) if total else 0.0,
            images_with_no_detection=sum(1 for anns in per_image.values() if not anns),
        )

    def errors(self, project_id: ProjectId) -> ErrorAnalysis:
        images, per_image = self._gather(project_id)
        anns = [a for anns in per_image.values() for a in anns]

        cats = [
            ErrorCategory(
                name="low_confidence",
                count=sum(1 for a in anns if a.confidence < _LOW_CONF),
                description=f"detections with confidence < {_LOW_CONF}",
            ),
            ErrorCategory(
                name="small_objects",
                count=self._count_small(per_image),
                description=f"boxes smaller than {_SMALL_AREA_FRAC:.0%} of the image",
            ),
            ErrorCategory(
                name="duplicate_annotations",
                count=self._count_duplicates(per_image),
                description=f"box pairs with IoU > {_DUP_IOU} (post-NMS residual)",
            ),
            ErrorCategory(
                name="empty_masks",
                count=sum(1 for a in anns if a.mask is not None and mask_is_empty(a.mask.rle)),
                description="segmentation masks with no foreground",
            ),
            ErrorCategory(
                name="crowded_scenes",
                count=sum(1 for anns in per_image.values() if len(anns) > _CROWDED),
                description=f"images with more than {_CROWDED} detections",
            ),
            ErrorCategory(
                name="possible_missed_detection",
                count=sum(1 for anns in per_image.values() if not anns),
                description="images that produced zero detections (proxy only)",
            ),
        ]
        return ErrorAnalysis(
            project_id=project_id,
            total_annotations=len(anns),
            categories=cats,
            unmeasurable=_UNMEASURABLE,
        )

    def _count_small(self, per_image) -> int:
        small = 0
        for img_id, anns in per_image.items():
            rec = self._images.get(img_id)
            if rec is None:
                continue
            threshold = _SMALL_AREA_FRAC * rec.width * rec.height
            for a in anns:
                g = a.geometry
                if isinstance(g, Box2D) and g.w * g.h < threshold:
                    small += 1
        return small

    @staticmethod
    def _count_duplicates(per_image) -> int:
        dups = 0
        for anns in per_image.values():
            boxes = [a.geometry for a in anns if isinstance(a.geometry, Box2D)]
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if overlap_iou(boxes[i], boxes[j]) > _DUP_IOU:
                        dups += 1
        return dups
