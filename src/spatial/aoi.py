"""CRS-safe AOI/raster alignment and effective-AOI preparation."""

from __future__ import annotations

from dataclasses import dataclass

from pyproj import CRS, Transformer
from pyproj.exceptions import ProjError
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_shape

from src.geometry.crs import choose_metric_crs, coerce_crs
from src.io.kml_reader import AOIResult
from src.models import RasterMetadata


@dataclass(frozen=True)
class AOIAnalysis:
    """An AOI aligned to available raster coverage."""

    original_geometry: BaseGeometry | None
    aligned_original_geometry: BaseGeometry | None
    effective_geometry: BaseGeometry | None
    raster_footprint: BaseGeometry | None
    overlap_status: str
    overlap_fraction: float | None
    original_area_m2: float | None
    effective_area_m2: float | None
    area_available: bool
    aoi_spatial_match_available: bool
    source_crs: str
    raster_crs: str | None
    metric_crs: str | None
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


def raster_footprint(metadata: RasterMetadata) -> Polygon:
    """Build the raster footprint from Rasterio-derived bounds."""

    left, bottom, right, top = metadata.bounds

    return Polygon(
        (
            (left, bottom),
            (right, bottom),
            (right, top),
            (left, top),
        )
    )


def transform_geometry(
    geometry: BaseGeometry,
    source_crs: str | CRS,
    target_crs: str | CRS,
) -> BaseGeometry:
    """Transform a Shapely geometry using explicit CRS definitions."""

    transformer = Transformer.from_crs(
        source_crs,
        target_crs,
        always_xy=True,
    )

    return transform_shape(
        transformer.transform,
        geometry,
    )


def _metric_areas(
    aligned_original: BaseGeometry,
    effective: BaseGeometry,
    raster_crs: CRS,
    bounds: tuple[float, float, float, float],
) -> tuple[
    float | None,
    float | None,
    str | None,
    str | None,
]:
    """Calculate AOI areas in a defensible metric CRS."""

    metric_crs, reason = choose_metric_crs(
        raster_crs,
        bounds,
    )

    if metric_crs is None:
        return (
            None,
            None,
            None,
            reason,
        )

    try:
        original_metric = transform_geometry(
            aligned_original,
            raster_crs,
            metric_crs,
        )

        effective_metric = transform_geometry(
            effective,
            raster_crs,
            metric_crs,
        )

    except (ProjError, ValueError, TypeError) as error:
        return (
            None,
            None,
            metric_crs.to_string(),
            f"AOI could not be transformed to metric CRS: {error}",
        )

    return (
        float(original_metric.area),
        float(effective_metric.area),
        metric_crs.to_string(),
        None,
    )


def _unique_warnings(
    warnings: list[str],
) -> tuple[str, ...]:
    """Return warnings while preserving their original order."""

    return tuple(
        dict.fromkeys(
            warning
            for warning in warnings
            if warning
        )
    )


def _unique_errors(
    errors: list[str],
) -> tuple[str, ...]:
    """Return errors while preserving their original order."""

    return tuple(
        dict.fromkeys(
            error
            for error in errors
            if error
        )
    )


def analyze_aoi(
    aoi: AOIResult,
    metadata: RasterMetadata,
) -> AOIAnalysis:
    """
    Transform and intersect a KML AOI with the raster footprint.

    KML geometries are expected to be WGS84 according to the KML
    reader contract. Physical area is calculated only when a safe
    metric CRS can be established.
    """

    warnings = list(aoi.warnings)
    errors = list(aoi.errors)

    common = {
        "original_geometry": aoi.geometry,
        "source_crs": aoi.source_crs,
        "raster_crs": metadata.crs,
    }

    if not aoi.valid or aoi.geometry is None:
        return AOIAnalysis(
            **common,
            aligned_original_geometry=None,
            effective_geometry=None,
            raster_footprint=None,
            overlap_status="unavailable",
            overlap_fraction=None,
            original_area_m2=None,
            effective_area_m2=None,
            area_available=False,
            aoi_spatial_match_available=False,
            metric_crs=None,
            warnings=_unique_warnings(warnings),
            errors=_unique_errors(errors),
        )

    raster_crs = coerce_crs(
        metadata.crs
    )

    if not metadata.georeferenced or raster_crs is None:
        warnings.append(
            "AOI cannot be spatially matched because raster "
            "georeferencing is missing."
        )

        return AOIAnalysis(
            **common,
            aligned_original_geometry=None,
            effective_geometry=None,
            raster_footprint=None,
            overlap_status="unavailable",
            overlap_fraction=None,
            original_area_m2=None,
            effective_area_m2=None,
            area_available=False,
            aoi_spatial_match_available=False,
            metric_crs=None,
            warnings=_unique_warnings(warnings),
            errors=_unique_errors(errors),
        )

    try:
        aligned = transform_geometry(
            aoi.geometry,
            aoi.source_crs,
            raster_crs,
        )

    except (ProjError, ValueError, TypeError) as error:
        errors.append(
            "AOI could not be transformed into raster CRS: "
            f"{error}"
        )

        return AOIAnalysis(
            **common,
            aligned_original_geometry=None,
            effective_geometry=None,
            raster_footprint=None,
            overlap_status="unavailable",
            overlap_fraction=None,
            original_area_m2=None,
            effective_area_m2=None,
            area_available=False,
            aoi_spatial_match_available=False,
            metric_crs=None,
            warnings=_unique_warnings(warnings),
            errors=_unique_errors(errors),
        )

    footprint = raster_footprint(
        metadata
    )

    if aligned.is_empty or not aligned.is_valid:
        errors.append(
            "The transformed AOI geometry is empty or invalid."
        )

        return AOIAnalysis(
            **common,
            aligned_original_geometry=aligned,
            effective_geometry=None,
            raster_footprint=footprint,
            overlap_status="unavailable",
            overlap_fraction=None,
            original_area_m2=None,
            effective_area_m2=None,
            area_available=False,
            aoi_spatial_match_available=False,
            metric_crs=None,
            warnings=_unique_warnings(warnings),
            errors=_unique_errors(errors),
        )

    intersection = aligned.intersection(
        footprint
    )

    if intersection.is_empty or intersection.area <= 0:
        (
            original_area,
            _,
            metric_crs,
            area_error,
        ) = _metric_areas(
            aligned,
            intersection,
            raster_crs,
            metadata.bounds,
        )

        if area_error is not None:
            warnings.append(
                area_error
            )

        return AOIAnalysis(
            **common,
            aligned_original_geometry=aligned,
            effective_geometry=None,
            raster_footprint=footprint,
            overlap_status="outside",
            overlap_fraction=0.0,
            original_area_m2=original_area,
            effective_area_m2=(
                0.0
                if original_area is not None
                else None
            ),
            area_available=(
                original_area is not None
            ),
            aoi_spatial_match_available=True,
            metric_crs=metric_crs,
            warnings=_unique_warnings(warnings),
            errors=_unique_errors(errors),
        )

    status = (
        "inside"
        if aligned.within(footprint)
        else "partial_overlap"
    )

    fraction = (
        float(
            intersection.area
            / aligned.area
        )
        if aligned.area > 0
        else None
    )

    (
        original_area,
        effective_area,
        metric_crs,
        area_error,
    ) = _metric_areas(
        aligned,
        intersection,
        raster_crs,
        metadata.bounds,
    )

    if area_error is not None:
        warnings.append(
            area_error
        )

    return AOIAnalysis(
        **common,
        aligned_original_geometry=aligned,
        effective_geometry=intersection,
        raster_footprint=footprint,
        overlap_status=status,
        overlap_fraction=fraction,
        original_area_m2=original_area,
        effective_area_m2=effective_area,
        area_available=(
            original_area is not None
            and effective_area is not None
        ),
        aoi_spatial_match_available=True,
        metric_crs=metric_crs,
        warnings=_unique_warnings(warnings),
        errors=_unique_errors(errors),
    )


def analyze_raster_as_aoi(
    metadata: RasterMetadata,
) -> AOIAnalysis:
    """
    Treat the complete raster footprint as the analysis region.

    This is used when the user does not provide a KML boundary.

    Important behavior:

    - A georeferenced raster can provide a spatially meaningful
      analysis footprint.
    - A projected raster with metric units can provide physical
      area measurements directly.
    - A geographic raster can be reprojected to a metric CRS.
    - A raster without CRS/georeferencing can still be used for
      image-space tree detection and full-raster inclusion, but
      physical m²/hectare measurements remain unavailable.
    """

    warnings: list[str] = []
    errors: list[str] = []

    footprint = raster_footprint(
        metadata
    )

    raster_crs = coerce_crs(
        metadata.crs
    )

    # -------------------------------------------------------------
    # Missing georeferencing
    # -------------------------------------------------------------

    if not metadata.georeferenced or raster_crs is None:
        warnings.append(
            "No KML AOI supplied; the complete uploaded raster "
            "is treated as the analysis region."
        )

        warnings.append(
            "The raster does not have sufficient georeferencing "
            "for physical area or hectare-based measurements."
        )

        return AOIAnalysis(
            original_geometry=footprint,
            aligned_original_geometry=footprint,
            effective_geometry=footprint,
            raster_footprint=footprint,
            overlap_status="raster",
            overlap_fraction=1.0,
            original_area_m2=None,
            effective_area_m2=None,
            area_available=False,
            aoi_spatial_match_available=True,
            source_crs="PIXEL_SPACE",
            raster_crs=metadata.crs,
            metric_crs=None,
            warnings=_unique_warnings(warnings),
            errors=_unique_errors(errors),
        )

    # -------------------------------------------------------------
    # Georeferenced raster
    # -------------------------------------------------------------

    try:
        metric_crs, reason = choose_metric_crs(
            raster_crs,
            metadata.bounds,
        )

    except (
        ProjError,
        ValueError,
        TypeError,
    ) as error:
        metric_crs = None
        reason = (
            "Metric CRS could not be determined for the "
            f"raster footprint: {error}"
        )

    if metric_crs is None:
        if reason:
            warnings.append(
                reason
            )

        warnings.append(
            "No KML AOI supplied; the complete uploaded raster "
            "is treated as the analysis region."
        )

        warnings.append(
            "Physical area and hectare-based measurements are "
            "unavailable because no safe metric CRS is available."
        )

        return AOIAnalysis(
            original_geometry=footprint,
            aligned_original_geometry=footprint,
            effective_geometry=footprint,
            raster_footprint=footprint,
            overlap_status="raster",
            overlap_fraction=1.0,
            original_area_m2=None,
            effective_area_m2=None,
            area_available=False,
            aoi_spatial_match_available=True,
            source_crs=raster_crs.to_string(),
            raster_crs=metadata.crs,
            metric_crs=None,
            warnings=_unique_warnings(warnings),
            errors=_unique_errors(errors),
        )

    # -------------------------------------------------------------
    # Physical raster footprint area
    # -------------------------------------------------------------

    try:
        metric_footprint = transform_geometry(
            footprint,
            raster_crs,
            metric_crs,
        )

        raster_area_m2 = float(
            metric_footprint.area
        )

    except (
        ProjError,
        ValueError,
        TypeError,
    ) as error:
        warnings.append(
            "Raster footprint could not be transformed to "
            f"metric CRS: {error}"
        )

        warnings.append(
            "Physical area and hectare-based measurements are "
            "unavailable."
        )

        return AOIAnalysis(
            original_geometry=footprint,
            aligned_original_geometry=footprint,
            effective_geometry=footprint,
            raster_footprint=footprint,
            overlap_status="raster",
            overlap_fraction=1.0,
            original_area_m2=None,
            effective_area_m2=None,
            area_available=False,
            aoi_spatial_match_available=True,
            source_crs=raster_crs.to_string(),
            raster_crs=metadata.crs,
            metric_crs=metric_crs.to_string(),
            warnings=_unique_warnings(warnings),
            errors=_unique_errors(errors),
        )

    if raster_area_m2 <= 0:
        errors.append(
            "Raster footprint has zero or negative physical area."
        )

        return AOIAnalysis(
            original_geometry=footprint,
            aligned_original_geometry=footprint,
            effective_geometry=footprint,
            raster_footprint=footprint,
            overlap_status="raster",
            overlap_fraction=1.0,
            original_area_m2=None,
            effective_area_m2=None,
            area_available=False,
            aoi_spatial_match_available=True,
            source_crs=raster_crs.to_string(),
            raster_crs=metadata.crs,
            metric_crs=metric_crs.to_string(),
            warnings=_unique_warnings(warnings),
            errors=_unique_errors(errors),
        )

    warnings.append(
        "No KML AOI supplied; the complete uploaded raster "
        "is treated as the analysis region."
    )

    return AOIAnalysis(
        original_geometry=footprint,
        aligned_original_geometry=footprint,
        effective_geometry=footprint,
        raster_footprint=footprint,
        overlap_status="raster",
        overlap_fraction=1.0,
        original_area_m2=raster_area_m2,
        effective_area_m2=raster_area_m2,
        area_available=True,
        aoi_spatial_match_available=True,
        source_crs=raster_crs.to_string(),
        raster_crs=metadata.crs,
        metric_crs=metric_crs.to_string(),
        warnings=_unique_warnings(warnings),
        errors=_unique_errors(errors),
    )