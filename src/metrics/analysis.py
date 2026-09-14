"""Pure, provenance-aware metrics for AOI-filtered model detections."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Literal, Sequence

from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from src.geometry.area import m2_to_hectares
from src.geometry.crown_refinement import CrownRefinementResult
from src.models import RasterMetadata
from src.spatial.aoi import AOIAnalysis, transform_geometry
from src.spatial.aoi_filter import AOIFilterResult, AOITreeResult

MeasurementBasis = Literal[
    "georeferenced_metric_raster",
    "geographic_raster_reprojected_to_metric_crs",
    "pixel_only",
    "unavailable",
]


@dataclass(frozen=True)
class MeasurementStatus:
    """How physical measurements were, or were not, obtained."""

    physical_area_available: bool
    spatial_reference: str | None
    metric_crs: str | None
    measurement_basis: MeasurementBasis
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisMetrics:
    """AOI-aware metrics with explicit model and measurement provenance.

    Counts describe DeepForest model detections, not a true forest population.
    Canopy values are estimated from RGB-refined footprints and bbox proxies,
    not semantic segmentation or ground-truth crown measurements.
    """

    total_detections_before_aoi: int
    detected_tree_count: int
    refined_tree_count: int
    bbox_fallback_count: int
    failed_tree_count: int
    refinement_rate: float | None
    mean_detection_score: float | None
    median_detection_score: float | None
    mean_refinement_confidence: float | None
    aoi_area_m2: float | None
    aoi_area_ha: float | None
    estimated_canopy_area_m2: float | None
    estimated_canopy_area_ha: float | None
    estimated_canopy_cover_percent: float | None
    detected_tree_density_per_ha: float | None
    refined_crown_area_m2: float | None
    bbox_fallback_area_m2: float | None
    sum_individual_crown_area_m2: float | None
    physical_metrics_available: bool
    measurement_warnings: tuple[str, ...]
    measurement_basis: MeasurementBasis
    measurement_status: MeasurementStatus


def _measurement_status(metadata: RasterMetadata, analysis: AOIAnalysis) -> MeasurementStatus:
    warnings = list(analysis.warnings)
    available = analysis.area_available and analysis.metric_crs is not None
    if available and metadata.geographic_crs:
        basis: MeasurementBasis = "geographic_raster_reprojected_to_metric_crs"
    elif available and metadata.projected_crs:
        basis = "georeferenced_metric_raster"
    elif metadata.crs is None:
        basis = "pixel_only"
        warnings.append("Physical metrics unavailable because raster lacks valid georeferencing.")
    else:
        basis = "unavailable"
        warnings.append("Physical metrics unavailable because no safe metric CRS is available.")
    return MeasurementStatus(available, metadata.crs, analysis.metric_crs, basis, tuple(dict.fromkeys(warnings)))


def _included_trees(
    trees: Sequence[CrownRefinementResult], aoi_filter: AOIFilterResult
) -> list[tuple[CrownRefinementResult, AOITreeResult]]:
    by_id = {tree.tree_id: tree for tree in trees}
    if len(by_id) != len(trees):
        raise ValueError("Tree IDs must be unique for metric calculation.")
    included: list[tuple[CrownRefinementResult, AOITreeResult]] = []
    for filtered in aoi_filter.tree_results:
        if not filtered.included:
            continue
        tree = by_id.get(filtered.tree_id)
        if tree is None:
            raise ValueError(f"AOI filter references unknown tree ID {filtered.tree_id}.")
        included.append((tree, filtered))
    return included


def _metric_area(geometry: BaseGeometry, metadata: RasterMetadata, metric_crs: str) -> float:
    """Measure a raster-CRS geometry only after transformation to metric CRS."""
    if metadata.crs is None:
        raise ValueError("Cannot transform geometry without a raster CRS.")
    return float(transform_geometry(geometry, metadata.crs, metric_crs).area)


def _canopy_areas(
    included: Sequence[tuple[CrownRefinementResult, AOITreeResult]], metadata: RasterMetadata,
    metric_crs: str,
) -> tuple[float, float, float, float]:
    """Return refined, bbox, individual-sum, and union canopy areas in m²."""
    refined = 0.0
    fallback = 0.0
    geometries: list[BaseGeometry] = []
    for tree, filtered in included:
        # A failed refinement may retain a detected-tree count, but its geometry
        # is deliberately excluded from estimated canopy measurement.
        if tree.refinement_status == "failed" or filtered.clipped_geometry is None:
            continue
        area = _metric_area(filtered.clipped_geometry, metadata, metric_crs)
        if tree.refinement_status == "refined":
            refined += area
        elif tree.refinement_status == "fallback_bbox":
            fallback += area
        else:
            continue
        geometries.append(transform_geometry(filtered.clipped_geometry, metadata.crs, metric_crs))  # type: ignore[arg-type]
    union_area = float(unary_union(geometries).area) if geometries else 0.0
    return refined, fallback, refined + fallback, union_area


def calculate_analysis_metrics(
    trees: Sequence[CrownRefinementResult],
    aoi_filter: AOIFilterResult,
    analysis: AOIAnalysis,
    metadata: RasterMetadata,
) -> AnalysisMetrics:
    """Calculate metrics from AOI-filtered detections without hidden conversions.

    The function is deliberately independent of inference and thresholds, so it
    can be reused by future threshold-sensitivity runs. Trees intersecting the
    AOI count once; their canopy footprint contributes only its clipped portion.
    """
    included = _included_trees(trees, aoi_filter)
    statuses = [tree.refinement_status for tree, _ in included]
    detected_count = len(included)
    refined_count = statuses.count("refined")
    fallback_count = statuses.count("fallback_bbox")
    failed_count = statuses.count("failed")
    scores = [tree.detection_score for tree, _ in included]
    confidences = [tree.refinement_confidence for tree, _ in included]
    status = _measurement_status(metadata, analysis)

    aoi_area = analysis.effective_area_m2 if status.physical_area_available else None
    if aoi_area is not None and aoi_area <= 0:
        raise ValueError("Effective AOI area must be greater than zero for physical metrics.")
    if status.physical_area_available and status.metric_crs is not None:
        refined_area, fallback_area, individual_area, canopy_area = _canopy_areas(
            included, metadata, status.metric_crs
        )
        assert aoi_area is not None
        canopy_cover = canopy_area / aoi_area * 100.0
        if canopy_cover > 100.0 + 1e-7:
            raise ValueError("Estimated canopy cover exceeds 100%; inspect AOI and crown geometry.")
        canopy_cover = min(100.0, canopy_cover)
        aoi_ha = m2_to_hectares(aoi_area)
        canopy_ha = m2_to_hectares(canopy_area)
        density = detected_count / aoi_ha
    else:
        refined_area = fallback_area = individual_area = canopy_area = None
        canopy_cover = aoi_ha = canopy_ha = density = None

    return AnalysisMetrics(
        total_detections_before_aoi=len(trees),
        detected_tree_count=detected_count,
        refined_tree_count=refined_count,
        bbox_fallback_count=fallback_count,
        failed_tree_count=failed_count,
        refinement_rate=refined_count / detected_count if detected_count else None,
        mean_detection_score=float(mean(scores)) if scores else None,
        median_detection_score=float(median(scores)) if scores else None,
        mean_refinement_confidence=float(mean(confidences)) if confidences else None,
        aoi_area_m2=aoi_area,
        aoi_area_ha=aoi_ha,
        estimated_canopy_area_m2=canopy_area,
        estimated_canopy_area_ha=canopy_ha,
        estimated_canopy_cover_percent=canopy_cover,
        detected_tree_density_per_ha=density,
        refined_crown_area_m2=refined_area,
        bbox_fallback_area_m2=fallback_area,
        sum_individual_crown_area_m2=individual_area,
        physical_metrics_available=status.physical_area_available,
        measurement_warnings=status.warnings,
        measurement_basis=status.measurement_basis,
        measurement_status=status,
    )
