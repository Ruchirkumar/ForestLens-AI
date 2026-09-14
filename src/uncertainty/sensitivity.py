"""Detection-threshold sensitivity using one already-produced prediction set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from src.detection.model import ImageInput, filter_detections
from src.geometry.crown_refinement import CrownRefinementConfig, CrownRefinementResult, refine_crowns
from src.metrics.analysis import calculate_analysis_metrics
from src.models import RasterMetadata
from src.spatial.aoi import AOIAnalysis
from src.spatial.aoi_filter import filter_trees_to_aoi

DEFAULT_THRESHOLDS = (0.30, 0.40, 0.50, 0.60)


@dataclass(frozen=True)
class RefinementSummary:
    """Composition of crown refinement outputs, not a model-accuracy metric."""

    total_detections: int
    refined: int
    bbox_fallback: int
    failed: int
    refinement_rate: float | None
    fallback_rate: float | None


@dataclass(frozen=True)
class ThresholdSensitivityResult:
    """One deterministic threshold result from the same raw detector output."""

    threshold: float
    detected_tree_count: int
    refined_count: int
    fallback_count: int
    failed_count: int
    estimated_canopy_area_m2: float | None
    estimated_canopy_cover_percent: float | None
    detected_tree_density_per_ha: float | None


def calculate_refinement_summary(results: Sequence[CrownRefinementResult]) -> RefinementSummary:
    """Summarize refinement statuses without describing them as detector accuracy."""
    total = len(results)
    refined = sum(result.refinement_status == "refined" for result in results)
    fallback = sum(result.refinement_status == "fallback_bbox" for result in results)
    failed = sum(result.refinement_status == "failed" for result in results)
    return RefinementSummary(
        total, refined, fallback, failed,
        refined / total if total else None,
        fallback / total if total else None,
    )


def run_threshold_sensitivity(
    raw_predictions: pd.DataFrame,
    image: ImageInput,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    refinement_config: CrownRefinementConfig | None = None,
    metadata: RasterMetadata | None = None,
    aoi_analysis: AOIAnalysis | None = None,
) -> tuple[ThresholdSensitivityResult, ...]:
    """Refine/filter threshold variants without ever rerunning DeepForest inference.

    ``raw_predictions`` is the sole model output used. Physical metrics are
    available only when both valid raster metadata and an effective AOI are
    supplied; otherwise their fields deliberately remain ``None``.
    """
    normalized = tuple(float(threshold) for threshold in thresholds)
    if tuple(sorted(normalized)) != normalized or len(set(normalized)) != len(normalized):
        raise ValueError("Sensitivity thresholds must be unique and sorted ascending.")
    if any(not 0.0 <= threshold <= 1.0 for threshold in normalized):
        raise ValueError("Sensitivity thresholds must be between zero and one.")
    rows: list[ThresholdSensitivityResult] = []
    for threshold in normalized:
        filtered = filter_detections(raw_predictions, threshold)
        refined = refine_crowns(image, filtered, refinement_config)
        summary = calculate_refinement_summary(refined)
        if metadata is not None and aoi_analysis is not None:
            aoi_filter = filter_trees_to_aoi(refined, metadata, aoi_analysis)
            metrics = calculate_analysis_metrics(refined, aoi_filter, aoi_analysis, metadata)
            detected_count = metrics.detected_tree_count
            canopy_area = metrics.estimated_canopy_area_m2
            canopy_cover = metrics.estimated_canopy_cover_percent
            density = metrics.detected_tree_density_per_ha
        else:
            detected_count = summary.total_detections
            canopy_area = canopy_cover = density = None
        rows.append(ThresholdSensitivityResult(
            threshold, detected_count, summary.refined, summary.bbox_fallback, summary.failed,
            canopy_area, canopy_cover, density,
        ))
    return tuple(rows)


def threshold_sensitivity(predictions: pd.DataFrame, thresholds: Sequence[float] = DEFAULT_THRESHOLDS) -> list[dict[str, float | int]]:
    """Legacy count-only sensitivity helper that performs no inference."""
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        filtered = predictions[predictions["score"] >= threshold]
        rows.append({"threshold": float(threshold), "tree_count": int(len(filtered))})
    return rows
