"""Safe pixel, raster-footprint, and projected-polygon area helpers."""

from __future__ import annotations

from typing import Any

from shapely.geometry.base import BaseGeometry

from src.geometry.crs import coerce_crs, crs_linear_unit_to_metre
from src.models import RasterMetadata


def pixel_area_m2(
    pixel_count: int,
    pixel_width_m: float,
    pixel_height_m: float,
) -> float:
    return float(pixel_count) * float(pixel_width_m) * float(pixel_height_m)


def m2_to_hectares(area_m2: float) -> float:
    return float(area_m2) / 10_000.0


def get_pixel_area_m2(metadata: RasterMetadata) -> float | None:
    """Return physical pixel area only when a projected CRS supports it.

    Geographic degrees are never converted using a fixed fake metre factor.
    For reliable non-metre projected units, the native determinant is converted
    using the CRS-provided linear-unit conversion factor.
    """
    if not metadata.projected_crs or metadata.linear_unit_to_metre is None:
        return None
    return float(metadata.pixel_area_native * metadata.linear_unit_to_metre**2)


def calculate_raster_area_m2(metadata: RasterMetadata) -> float | None:
    """Return the full raster footprint area, never vegetation/canopy area."""
    pixel_area = get_pixel_area_m2(metadata)
    if pixel_area is None:
        return None
    return float(metadata.total_pixel_count * pixel_area)


def calculate_polygon_area_m2(geometry: BaseGeometry, source_crs: Any) -> float | None:
    """Return a planar polygon area in m², or ``None`` for unsafe CRS input."""
    if geometry.is_empty:
        return 0.0
    crs = coerce_crs(source_crs)
    _, conversion_factor = crs_linear_unit_to_metre(crs)
    if conversion_factor is None:
        return None
    return float(geometry.area * conversion_factor**2)
