from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from src.geometry.crown_refinement import CrownRefinementResult
from src.spatial.aoi_filter import AOIFilterResult


@dataclass(frozen=True)
class ManualGSDMetrics:
    """Physical metrics derived from a user-supplied GSD.

    This module is intentionally independent of CRS information.

    A manual GSD means:
        metres_per_pixel = user supplied value

    Therefore:
        pixel area = GSD²

    The result must never be presented as a georeferenced CRS-based
    measurement.
    """

    gsd_m_per_pixel: float
    image_area_m2: float
    image_area_ha: float
    canopy_area_m2: float
    canopy_area_ha: float
    canopy_cover_percent: float
    tree_density_per_ha: float
    measurement_basis: str
    warnings: tuple[str, ...]


def validate_gsd(gsd_m_per_pixel: float | None) -> float | None:
    """Validate a manually supplied GSD."""

    if gsd_m_per_pixel is None:
        return None

    try:
        gsd = float(gsd_m_per_pixel)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "GSD must be a numeric value in metres per pixel."
        ) from error

    if gsd <= 0:
        raise ValueError(
            "GSD must be greater than zero."
        )

    if gsd > 100:
        raise ValueError(
            "GSD is unusually large. Enter the ground sampling "
            "distance in metres per pixel."
        )

    return gsd


def _tree_geometry(
    tree: CrownRefinementResult,
) -> BaseGeometry | None:
    """Return the best available pixel-space crown geometry."""

    if tree.refinement_status == "failed":
        return None

    geometry = getattr(tree, "geometry", None)

    if geometry is None:
        return None

    if geometry.is_empty:
        return None

    return geometry


def calculate_manual_gsd_metrics(
    trees: Sequence[CrownRefinementResult],
    aoi_filter: AOIFilterResult,
    image_width: int,
    image_height: int,
    gsd_m_per_pixel: float,
) -> ManualGSDMetrics:
    """Calculate physical metrics from a user-supplied GSD.

    This is intended for a complete image/raster used as the analysis
    region. It does not create or pretend to create a CRS.

    Tree footprints remain in image/pixel space. Their areas are scaled
    by GSD² to obtain square metres.
    """

    gsd = validate_gsd(gsd_m_per_pixel)

    if gsd is None:
        raise ValueError(
            "A valid GSD is required."
        )

    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            "Image dimensions must be greater than zero."
        )

    by_id = {
        tree.tree_id: tree
        for tree in trees
    }

    included_trees = [
        item
        for item in aoi_filter.tree_results
        if item.included
    ]

    geometries: list[BaseGeometry] = []

    for filtered in included_trees:
        tree = by_id.get(filtered.tree_id)

        if tree is None:
            continue

        geometry = _tree_geometry(tree)

        if geometry is None:
            continue

        # Use the AOI-filtered geometry when available. For the
        # whole-raster/manual-GSD pathway this remains pixel-space.
        clipped_geometry = filtered.clipped_geometry

        if clipped_geometry is not None and not clipped_geometry.is_empty:
            geometry = clipped_geometry

        geometries.append(geometry)

    image_area_pixels = float(
        image_width * image_height
    )

    pixel_area_m2 = gsd * gsd

    image_area_m2 = (
        image_area_pixels
        * pixel_area_m2
    )

    image_area_ha = (
        image_area_m2
        / 10_000.0
    )

    canopy_area_pixels = (
        float(
            unary_union(geometries).area
        )
        if geometries
        else 0.0
    )

    canopy_area_m2 = (
        canopy_area_pixels
        * pixel_area_m2
    )

    canopy_area_ha = (
        canopy_area_m2
        / 10_000.0
    )

    canopy_cover_percent = (
        canopy_area_m2
        / image_area_m2
        * 100.0
    )

    # Numerical safety guard.
    canopy_cover_percent = min(
        100.0,
        max(0.0, canopy_cover_percent),
    )

    tree_density_per_ha = (
        len(included_trees)
        / image_area_ha
        if image_area_ha > 0
        else 0.0
    )

    warnings = (
        "Physical scale is based on user-provided GSD, not raster "
        "georeferencing.",
        "The accuracy of physical measurements depends on the accuracy "
        "of the supplied GSD.",
    )

    return ManualGSDMetrics(
        gsd_m_per_pixel=gsd,
        image_area_m2=image_area_m2,
        image_area_ha=image_area_ha,
        canopy_area_m2=canopy_area_m2,
        canopy_area_ha=canopy_area_ha,
        canopy_cover_percent=canopy_cover_percent,
        tree_density_per_ha=tree_density_per_ha,
        measurement_basis="user_provided_gsd",
        warnings=warnings,
    )