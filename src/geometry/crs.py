"""CRS classification and conservative metric-CRS selection helpers."""

from __future__ import annotations

from math import floor, isclose
from typing import Any

from pyproj import CRS
from rasterio.transform import rowcol, xy
from shapely.geometry import Polygon

from src.models import RasterMetadata


def coerce_crs(value: Any) -> CRS | None:
    """Return a pyproj CRS, or ``None`` when it cannot be parsed."""
    if value is None:
        return None
    try:
        return CRS.from_user_input(value)
    except Exception:
        return None


def crs_linear_unit_to_metre(value: Any) -> tuple[str | None, float | None]:
    """Return reliable projected horizontal-unit metadata, if present."""
    crs = coerce_crs(value)
    if crs is None or not crs.is_projected or len(crs.axis_info) < 2:
        return None, None
    first, second = crs.axis_info[:2]
    first_factor = first.unit_conversion_factor
    second_factor = second.unit_conversion_factor
    if (
        first_factor is None
        or second_factor is None
        or first_factor <= 0
        or second_factor <= 0
        or not isclose(float(first_factor), float(second_factor), rel_tol=1e-12)
    ):
        return first.unit_name, None
    return first.unit_name, float(first_factor)


def is_metre_based_projected_crs(value: Any) -> bool:
    """Whether a projected CRS exposes metre-based horizontal axes."""
    units, factor = crs_linear_unit_to_metre(value)
    return (
        units is not None
        and units.lower() in {"metre", "meter", "metres", "meters"}
        and factor is not None
        and isclose(factor, 1.0, rel_tol=1e-12)
    )


def choose_metric_crs(
    source_crs: Any,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[CRS | None, str]:
    """Conservatively choose a metric CRS for later geometry measurements.

    Existing metre-based projected CRSs are preserved. For geographic CRSs, a
    WGS84 UTM zone is chosen only from a valid raster centre within UTM's normal
    latitude range (-80 to 84 degrees). This is a location-based selection, not
    a hardcoded regional assumption. All other cases return ``None`` with why.
    """
    crs = coerce_crs(source_crs)
    if crs is None:
        return None, "Source CRS is missing or invalid."
    if crs.is_projected:
        if is_metre_based_projected_crs(crs):
            return crs, "Source CRS is already projected with metre-based axes."
        return None, "Source projected CRS is not metre-based or has unreliable linear units."
    if not crs.is_geographic:
        return None, "Source CRS is neither a usable geographic nor projected CRS."
    if bounds is None:
        return None, "Geographic CRS requires valid bounds to choose a local metric CRS."
    left, bottom, right, top = bounds
    if not (-180 <= left <= 180 and -180 <= right <= 180 and -90 <= bottom <= 90 and -90 <= top <= 90):
        return None, "Geographic bounds are outside valid longitude/latitude ranges."
    longitude = (left + right) / 2.0
    latitude = (bottom + top) / 2.0
    if not -80.0 <= latitude < 84.0:
        return None, "Raster centre is outside UTM's normal latitude range."
    zone = int(floor((longitude + 180.0) / 6.0)) + 1
    zone = min(60, max(1, zone))
    epsg = (32600 if latitude >= 0 else 32700) + zone
    return CRS.from_epsg(epsg), f"Selected WGS 84 / UTM zone {zone}{'N' if latitude >= 0 else 'S'} from raster centre."


def pixel_to_map(
    metadata: RasterMetadata,
    column: float,
    row: float,
    *,
    offset: str = "center",
) -> tuple[float, float]:
    """Convert image pixel coordinates (column, row) to map coordinates."""
    x_coordinate, y_coordinate = xy(metadata.transform, row, column, offset=offset)
    return float(x_coordinate), float(y_coordinate)


def map_to_pixel(metadata: RasterMetadata, x_coordinate: float, y_coordinate: float) -> tuple[int, int]:
    """Convert map coordinates to integer image coordinates (column, row)."""
    row, column = rowcol(metadata.transform, x_coordinate, y_coordinate)
    return int(column), int(row)


def pixel_bbox_to_map_polygon(
    metadata: RasterMetadata,
    bbox: tuple[float, float, float, float],
) -> Polygon:
    """Map a pixel-space ``(xmin, ymin, xmax, ymax)`` box to a map polygon."""
    xmin, ymin, xmax, ymax = bbox
    corners = [
        pixel_to_map(metadata, xmin, ymin, offset="ul"),
        pixel_to_map(metadata, xmax, ymin, offset="ul"),
        pixel_to_map(metadata, xmax, ymax, offset="ul"),
        pixel_to_map(metadata, xmin, ymax, offset="ul"),
    ]
    return Polygon(corners)


def ensure_same_crs(*geometries_and_crs: Any) -> None:
    """Placeholder compatibility hook; CRS comparison is added with AOI support."""
    raise NotImplementedError("CRS comparison is implemented with AOI integration.")
