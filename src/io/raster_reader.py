"""Safe raster metadata and RGB extraction for detection-compatible imagery."""

from __future__ import annotations

from math import isclose
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS
from rasterio.io import DatasetReaderBase

from ..models import RasterMetadata

SUPPORTED_RASTER_EXTENSIONS = frozenset({".tif", ".tiff", ".png", ".jpg", ".jpeg"})
SUPPORTED_RGB_BAND_COUNTS = frozenset({1, 3, 4})


class RasterValidationError(ValueError):
    """Raised when a raster cannot safely be used as an RGB image source."""


def _classify_crs(crs: rasterio.crs.CRS | None) -> tuple[str, bool, bool, str | None, float | None]:
    if crs is None:
        return "missing", False, False, None, None
    try:
        pyproj_crs = CRS.from_user_input(crs)
    except Exception:
        return "unknown", False, False, None, None
    if pyproj_crs.is_geographic:
        return "geographic", True, False, "degree", None
    if not pyproj_crs.is_projected:
        return "unknown", False, False, None, None

    axes = pyproj_crs.axis_info
    if len(axes) < 2:
        return "projected", False, True, None, None
    factors = [axis.unit_conversion_factor for axis in axes[:2]]
    units = axes[0].unit_name
    if any(factor is None or factor <= 0 for factor in factors) or not isclose(
        float(factors[0]), float(factors[1]), rel_tol=1e-12
    ):
        return "projected", False, True, units, None
    return "projected", False, True, units, float(factors[0])


def _is_metre_unit(units: str | None, conversion_factor: float | None) -> bool:
    return (
        units is not None
        and units.lower() in {"metre", "meter", "metres", "meters"}
        and conversion_factor is not None
        and isclose(conversion_factor, 1.0, rel_tol=1e-12)
    )


def validate_rgb_band_count(band_count: int) -> None:
    """Allow only unambiguous grayscale/RGB/RGBA source band structures."""
    if band_count not in SUPPORTED_RGB_BAND_COUNTS:
        raise RasterValidationError(
            f"Raster has {band_count} bands. Only 1, 3, or 4 bands can be safely "
            "interpreted for RGB detection; multispectral band mapping is required."
        )


def validate_raster_metadata(metadata: RasterMetadata, max_pixels: int | None = None) -> None:
    """Reject invalid metadata while permitting non-georeferenced imagery."""
    if metadata.width <= 0 or metadata.height <= 0:
        raise RasterValidationError("Raster dimensions must both be greater than zero.")
    if max_pixels is not None and metadata.total_pixel_count > max_pixels:
        raise RasterValidationError(
            f"Raster has {metadata.total_pixel_count} pixels, exceeding configured maximum {max_pixels}."
        )
    validate_rgb_band_count(metadata.band_count)
    determinant = metadata.transform.a * metadata.transform.e - metadata.transform.b * metadata.transform.d
    if not np.isfinite(tuple(metadata.transform)).all() or determinant == 0:
        raise RasterValidationError("Raster has an invalid affine transform.")
    left, bottom, right, top = metadata.bounds
    if not np.isfinite(metadata.bounds).all() or right <= left or top <= bottom:
        raise RasterValidationError("Raster has invalid bounds.")


def _bounds_from_transform(width: int, height: int, transform: Affine) -> tuple[float, float, float, float]:
    """Return normalized bounds from all affine-transformed raster corners."""
    corners = [
        transform @ (0, 0),
        transform @ (width, 0),
        transform @ (0, height),
        transform @ (width, height),
    ]
    x_coordinates, y_coordinates = zip(*corners, strict=True)
    return min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)


def metadata_from_dataset(
    source: DatasetReaderBase,
    path: str | Path | None = None,
    *,
    max_pixels: int | None = None,
) -> RasterMetadata:
    """Create structured metadata from an open Rasterio dataset.

    Missing CRS is deliberately a warning rather than a fatal error: PNG/JPG
    and unreferenced TIFF inputs remain valid for detection but not physical
    measurements.
    """
    transform = source.transform
    if not isinstance(transform, Affine):
        raise RasterValidationError("Raster transform is unavailable.")
    crs_type, geographic, projected, units, unit_to_metre = _classify_crs(source.crs)
    georeferenced = crs_type in {"projected", "geographic"}
    if crs_type == "projected" and unit_to_metre is not None:
        physical_reason = "Projected CRS has a reliable linear-unit conversion."
        physical_available = True
    elif crs_type == "projected":
        physical_reason = "Projected CRS has no reliable linear-unit conversion."
        physical_available = False
    elif crs_type == "geographic":
        physical_reason = "Geographic CRS coordinates are angular; projection is required for metre area."
        physical_available = False
    elif crs_type == "missing":
        physical_reason = "Raster has no valid projected metric CRS."
        physical_available = False
    else:
        physical_reason = "Raster CRS could not be classified as a valid projected metric CRS."
        physical_available = False

    raster_path = str(path) if path is not None else getattr(source, "name", None)
    metadata = RasterMetadata(
        path=raster_path,
        name=Path(raster_path).name if raster_path else None,
        width=int(source.width),
        height=int(source.height),
        band_count=int(source.count),
        dtype=str(source.dtypes[0]) if source.dtypes else "unknown",
        crs=source.crs.to_string() if source.crs else None,
        crs_type=crs_type,  # type: ignore[arg-type]
        transform=transform,
        bounds=_bounds_from_transform(source.width, source.height, transform),
        pixel_width=float(abs(transform.a)),
        pixel_height=float(abs(transform.e)),
        pixel_area_native=float(abs(transform.a * transform.e - transform.b * transform.d)),
        gsd_x=float(abs(transform.a)),
        gsd_y=float(abs(transform.e)),
        gsd_x_m=float(abs(transform.a)) if _is_metre_unit(units, unit_to_metre) else None,
        gsd_y_m=float(abs(transform.e)) if _is_metre_unit(units, unit_to_metre) else None,
        georeferenced=georeferenced,
        geographic_crs=geographic,
        projected_crs=projected,
        units=units,
        linear_unit_to_metre=unit_to_metre,
        total_pixel_count=int(source.width * source.height),
        physical_area_available=physical_available,
        physical_area_reason=physical_reason,
        validation_warnings=(
            ("Raster has no CRS; physical area is unavailable.",) if crs_type == "missing" else ()
        ),
    )
    validate_raster_metadata(metadata, max_pixels=max_pixels)
    return metadata


def read_raster_metadata(
    path: str | Path,
    *,
    max_pixels: int | None = None,
) -> RasterMetadata:
    """Read and validate metadata from a supported local raster path."""
    raster_path = Path(path)
    if raster_path.suffix.lower() not in SUPPORTED_RASTER_EXTENSIONS:
        raise RasterValidationError(f"Unsupported raster extension: {raster_path.suffix or '<none>'}.")
    if not raster_path.is_file():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")
    try:
        with rasterio.open(raster_path) as source:
            return metadata_from_dataset(source, raster_path, max_pixels=max_pixels)
    except rasterio.errors.RasterioError as error:
        raise RasterValidationError(f"Raster is unreadable: {raster_path}") from error


def read_raster_rgb(path: str | Path, *, max_pixels: int | None = None) -> tuple[np.ndarray, RasterMetadata]:
    """Read an unambiguous raster RGB array plus its metadata.

    One-band rasters are replicated into RGB. Four-band rasters use their first
    three bands and intentionally drop alpha. Five-or-more-band multispectral
    data is rejected rather than risking an NIR-as-RGB interpretation.
    """
    raster_path = Path(path)
    with rasterio.open(raster_path) as source:
        metadata = metadata_from_dataset(source, raster_path, max_pixels=max_pixels)
        data = source.read()
    if metadata.band_count == 1:
        rgb = np.repeat(data, 3, axis=0)
    else:
        rgb = data[:3]
    return np.moveaxis(rgb, 0, -1).astype(np.float32, copy=False), metadata
