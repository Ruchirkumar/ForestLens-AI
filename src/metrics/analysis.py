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
    "user_provided_gsd",
    "pixel_only",
    "unavailable",
]


@dataclass(frozen=True)
class MeasurementStatus:
    physical_area_available: bool
    spatial_reference: str | None
    metric_crs: str | None
    measurement_basis: MeasurementBasis
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisMetrics:
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


def _measurement_status(
    metadata: RasterMetadata,
    analysis: AOIAnalysis,
    manual_gsd_m: float | None = None,
) -> MeasurementStatus:
    """Determine whether physical measurements can be reported safely."""

    warnings = list(analysis.warnings)

    # ---------------------------------------------------------------
    # 1. Prefer verified CRS-based measurements.
    # ---------------------------------------------------------------
    available = (
        analysis.area_available
        and analysis.metric_crs is not None
    )

    if available and metadata.geographic_crs:
        return MeasurementStatus(
            physical_area_available=True,
            spatial_reference=metadata.crs,
            metric_crs=analysis.metric_crs,
            measurement_basis=(
                "geographic_raster_reprojected_to_metric_crs"
            ),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    if available and metadata.projected_crs:
        return MeasurementStatus(
            physical_area_available=True,
            spatial_reference=metadata.crs,
            metric_crs=analysis.metric_crs,
            measurement_basis="georeferenced_metric_raster",
            warnings=tuple(dict.fromkeys(warnings)),
        )

    # ---------------------------------------------------------------
    # 2. User-provided GSD fallback.
    #
    # This is intentionally allowed only when the raster itself is
    # the analysis AOI. A GSD gives pixel -> ground scale, but it does
    # NOT create a CRS or allow KML/image spatial alignment.
    # ---------------------------------------------------------------
    manual_gsd_available = (
        manual_gsd_m is not None
        and manual_gsd_m > 0
        and analysis.overlap_status == "raster"
        and analysis.effective_geometry is not None
    )

    if manual_gsd_available:
        # Remove older generic warnings that would incorrectly say
        # physical measurements are unavailable.
        warnings = [
            warning
            for warning in warnings
            if (
                "physical area or hectare-based measurements"
                not in warning.lower()
                and "physical area and hectare-based measurements"
                not in warning.lower()
            )
        ]

        warnings.append(
            "Physical measurements use the user-provided GSD. "
            "This does not create raster georeferencing or enable "
            "KML-to-image spatial alignment."
        )

        warnings.append(
            "Physical measurement accuracy depends on the accuracy "
            "of the supplied GSD."
        )

        return MeasurementStatus(
            physical_area_available=True,
            spatial_reference=(
                metadata.crs
                if metadata.crs is not None
                else "PIXEL_SPACE"
            ),
            metric_crs=None,
            measurement_basis="user_provided_gsd",
            warnings=tuple(dict.fromkeys(warnings)),
        )

    # ---------------------------------------------------------------
    # 3. No safe physical measurement basis.
    # ---------------------------------------------------------------
    if metadata.crs is None:
        basis: MeasurementBasis = "pixel_only"

        warnings.append(
            "Physical metrics unavailable because raster lacks valid "
            "georeferencing or a user-provided GSD."
        )
    else:
        basis = "unavailable"

        warnings.append(
            "Physical metrics unavailable because no safe metric CRS "
            "is available and no usable user-provided GSD was supplied."
        )

    return MeasurementStatus(
        physical_area_available=False,
        spatial_reference=metadata.crs,
        metric_crs=analysis.metric_crs,
        measurement_basis=basis,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _included_trees(
    trees: Sequence[CrownRefinementResult],
    aoi_filter: AOIFilterResult,
) -> list[tuple[CrownRefinementResult, AOITreeResult]]:
    """Return trees that are included by the AOI filter."""

    by_id = {
        tree.tree_id: tree
        for tree in trees
    }

    if len(by_id) != len(trees):
        raise ValueError(
            "Tree IDs must be unique for metric calculation."
        )

    included: list[
        tuple[CrownRefinementResult, AOITreeResult]
    ] = []

    for filtered in aoi_filter.tree_results:
        if not filtered.included:
            continue

        tree = by_id.get(filtered.tree_id)

        if tree is None:
            raise ValueError(
                "AOI filter references unknown tree ID "
                f"{filtered.tree_id}."
            )

        included.append(
            (tree, filtered)
        )

    return included


def _metric_area(
    geometry: BaseGeometry,
    metadata: RasterMetadata,
    metric_crs: str,
) -> float:
    """Calculate geometry area after transformation to metric CRS."""

    if metadata.crs is None:
        raise ValueError(
            "Cannot transform geometry without a raster CRS."
        )

    return float(
        transform_geometry(
            geometry,
            metadata.crs,
            metric_crs,
        ).area
    )


def _manual_gsd_area(
    geometry: BaseGeometry,
    gsd_m: float,
) -> float:
    """Convert pixel-space geometry area to square metres using GSD."""

    return float(
        geometry.area
        * gsd_m
        * gsd_m
    )


def _canopy_areas(
    included: Sequence[
        tuple[CrownRefinementResult, AOITreeResult]
    ],
    metadata: RasterMetadata,
    metric_crs: str,
) -> tuple[float, float, float, float]:
    """Calculate canopy areas using CRS-based metric geometry."""

    refined = 0.0
    fallback = 0.0
    geometries: list[BaseGeometry] = []

    for tree, filtered in included:
        if (
            tree.refinement_status == "failed"
            or filtered.clipped_geometry is None
        ):
            continue

        area = _metric_area(
            filtered.clipped_geometry,
            metadata,
            metric_crs,
        )

        if tree.refinement_status == "refined":
            refined += area

        elif tree.refinement_status == "fallback_bbox":
            fallback += area

        else:
            continue

        geometries.append(
            transform_geometry(
                filtered.clipped_geometry,
                metadata.crs,
                metric_crs,
            )
        )

    union_area = (
        float(
            unary_union(geometries).area
        )
        if geometries
        else 0.0
    )

    return (
        refined,
        fallback,
        refined + fallback,
        union_area,
    )


def _canopy_areas_from_gsd(
    included: Sequence[
        tuple[CrownRefinementResult, AOITreeResult]
    ],
    gsd_m: float,
) -> tuple[float, float, float, float]:
    """Calculate canopy areas from pixel geometry and user-provided GSD."""

    refined = 0.0
    fallback = 0.0
    geometries: list[BaseGeometry] = []

    for tree, filtered in included:
        if (
            tree.refinement_status == "failed"
            or filtered.clipped_geometry is None
        ):
            continue

        geometry = filtered.clipped_geometry

        if geometry.is_empty:
            continue

        area = _manual_gsd_area(
            geometry,
            gsd_m,
        )

        if tree.refinement_status == "refined":
            refined += area

        elif tree.refinement_status == "fallback_bbox":
            fallback += area

        else:
            continue

        geometries.append(geometry)

    union_area = (
        float(
            unary_union(geometries).area
        )
        * gsd_m
        * gsd_m
        if geometries
        else 0.0
    )

    return (
        refined,
        fallback,
        refined + fallback,
        union_area,
    )


def calculate_analysis_metrics(
    trees: Sequence[CrownRefinementResult],
    aoi_filter: AOIFilterResult,
    analysis: AOIAnalysis,
    metadata: RasterMetadata,
    manual_gsd_m: float | None = None,
) -> AnalysisMetrics:
    """Calculate detection, canopy, density and provenance metrics."""

    included = _included_trees(
        trees,
        aoi_filter,
    )

    statuses = [
        tree.refinement_status
        for tree, _ in included
    ]

    detected_count = len(included)

    refined_count = statuses.count(
        "refined"
    )

    fallback_count = statuses.count(
        "fallback_bbox"
    )

    failed_count = statuses.count(
        "failed"
    )

    scores = [
        tree.detection_score
        for tree, _ in included
    ]

    confidences = [
        tree.refinement_confidence
        for tree, _ in included
    ]

    # Validate manual GSD if supplied.
    if manual_gsd_m is not None and manual_gsd_m <= 0:
        raise ValueError(
            "manual_gsd_m must be greater than zero when supplied."
        )

    status = _measurement_status(
        metadata,
        analysis,
        manual_gsd_m=manual_gsd_m,
    )

    # ---------------------------------------------------------------
    # AOI area
    # ---------------------------------------------------------------
    aoi_area = (
        analysis.effective_area_m2
        if status.physical_area_available
        else None
    )

    # For a user-provided GSD, the raster itself is the AOI.
    # Convert its pixel-space area into square metres.
    if (
        status.measurement_basis == "user_provided_gsd"
        and manual_gsd_m is not None
        and analysis.effective_geometry is not None
    ):
        aoi_area = float(
            analysis.effective_geometry.area
            * manual_gsd_m
            * manual_gsd_m
        )

    if (
        aoi_area is not None
        and aoi_area <= 0
    ):
        raise ValueError(
            "Effective AOI area must be greater than zero "
            "for physical metrics."
        )

    # ---------------------------------------------------------------
    # Physical canopy metrics
    # ---------------------------------------------------------------
    if (
        status.physical_area_available
        and aoi_area is not None
    ):
        if status.measurement_basis == "user_provided_gsd":
            assert manual_gsd_m is not None

            (
                refined_area,
                fallback_area,
                individual_area,
                canopy_area,
            ) = _canopy_areas_from_gsd(
                included,
                manual_gsd_m,
            )

        elif status.metric_crs is not None:
            (
                refined_area,
                fallback_area,
                individual_area,
                canopy_area,
            ) = _canopy_areas(
                included,
                metadata,
                status.metric_crs,
            )

        else:
            refined_area = None
            fallback_area = None
            individual_area = None
            canopy_area = None

        if (
            refined_area is not None
            and fallback_area is not None
            and individual_area is not None
            and canopy_area is not None
        ):
            canopy_cover = (
                canopy_area
                / aoi_area
                * 100.0
            )

            if canopy_cover > 100.0 + 1e-7:
                raise ValueError(
                    "Estimated canopy cover exceeds 100%; "
                    "inspect AOI and crown geometry."
                )

            canopy_cover = min(
                100.0,
                canopy_cover,
            )

            aoi_ha = m2_to_hectares(
                aoi_area
            )

            canopy_ha = m2_to_hectares(
                canopy_area
            )

            density = (
                detected_count
                / aoi_ha
            )

        else:
            canopy_cover = None
            aoi_ha = None
            canopy_ha = None
            density = None

    else:
        refined_area = None
        fallback_area = None
        individual_area = None
        canopy_area = None

        canopy_cover = None
        aoi_ha = None
        canopy_ha = None
        density = None

    # ---------------------------------------------------------------
    # Final metrics object
    # ---------------------------------------------------------------
    return AnalysisMetrics(
        total_detections_before_aoi=len(trees),
        detected_tree_count=detected_count,
        refined_tree_count=refined_count,
        bbox_fallback_count=fallback_count,
        failed_tree_count=failed_count,
        refinement_rate=(
            refined_count / detected_count
            if detected_count
            else None
        ),
        mean_detection_score=(
            float(mean(scores))
            if scores
            else None
        ),
        median_detection_score=(
            float(median(scores))
            if scores
            else None
        ),
        mean_refinement_confidence=(
            float(mean(confidences))
            if confidences
            else None
        ),
        aoi_area_m2=aoi_area,
        aoi_area_ha=aoi_ha,
        estimated_canopy_area_m2=canopy_area,
        estimated_canopy_area_ha=canopy_ha,
        estimated_canopy_cover_percent=canopy_cover,
        detected_tree_density_per_ha=density,
        refined_crown_area_m2=refined_area,
        bbox_fallback_area_m2=fallback_area,
        sum_individual_crown_area_m2=individual_area,
        physical_metrics_available=(
            status.physical_area_available
            and aoi_area is not None
            and canopy_area is not None
        ),
        measurement_warnings=status.warnings,
        measurement_basis=status.measurement_basis,
        measurement_status=status,
    )