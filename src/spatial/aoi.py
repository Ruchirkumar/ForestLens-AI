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
    """A KML AOI aligned to available raster coverage, when safely possible."""

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
    """Build a footprint from Rasterio-derived raster bounds in raster CRS."""
    left, bottom, right, top = metadata.bounds
    return Polygon(((left, bottom), (right, bottom), (right, top), (left, top)))


def transform_geometry(geometry: BaseGeometry, source_crs: str | CRS, target_crs: str | CRS) -> BaseGeometry:
    """Transform a Shapely geometry with explicit CRS inputs and XY ordering."""
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return transform_shape(transformer.transform, geometry)


def _metric_areas(
    aligned_original: BaseGeometry, effective: BaseGeometry, raster_crs: CRS,
    bounds: tuple[float, float, float, float],
) -> tuple[float | None, float | None, str | None, str | None]:
    metric_crs, reason = choose_metric_crs(raster_crs, bounds)
    if metric_crs is None:
        return None, None, None, reason
    try:
        original_metric = transform_geometry(aligned_original, raster_crs, metric_crs)
        effective_metric = transform_geometry(effective, raster_crs, metric_crs)
    except (ProjError, ValueError, TypeError) as error:
        return None, None, metric_crs.to_string(), f"AOI could not be transformed to metric CRS: {error}"
    return float(original_metric.area), float(effective_metric.area), metric_crs.to_string(), None


def analyze_aoi(aoi: AOIResult, metadata: RasterMetadata) -> AOIAnalysis:
    """Transform, intersect, and safely prepare a KML AOI for future analysis."""
    warnings = list(aoi.warnings)
    errors = list(aoi.errors)
    common = dict(original_geometry=aoi.geometry, source_crs=aoi.source_crs, raster_crs=metadata.crs)
    if not aoi.valid or aoi.geometry is None:
        return AOIAnalysis(**common, aligned_original_geometry=None, effective_geometry=None, raster_footprint=None, overlap_status="unavailable", overlap_fraction=None, original_area_m2=None, effective_area_m2=None, area_available=False, aoi_spatial_match_available=False, metric_crs=None, warnings=tuple(warnings), errors=tuple(errors))
    raster_crs = coerce_crs(metadata.crs)
    if not metadata.georeferenced or raster_crs is None:
        warnings.append("AOI cannot be spatially matched because raster georeferencing is missing.")
        return AOIAnalysis(**common, aligned_original_geometry=None, effective_geometry=None, raster_footprint=None, overlap_status="unavailable", overlap_fraction=None, original_area_m2=None, effective_area_m2=None, area_available=False, aoi_spatial_match_available=False, metric_crs=None, warnings=tuple(warnings), errors=tuple(errors))
    try:
        aligned = transform_geometry(aoi.geometry, aoi.source_crs, raster_crs)
    except (ProjError, ValueError, TypeError) as error:
        errors.append(f"AOI could not be transformed into raster CRS: {error}")
        return AOIAnalysis(**common, aligned_original_geometry=None, effective_geometry=None, raster_footprint=None, overlap_status="unavailable", overlap_fraction=None, original_area_m2=None, effective_area_m2=None, area_available=False, aoi_spatial_match_available=False, metric_crs=None, warnings=tuple(warnings), errors=tuple(errors))
    footprint = raster_footprint(metadata)
    intersection = aligned.intersection(footprint)
    if intersection.is_empty or intersection.area <= 0:
        original_area, _, metric_crs, area_error = _metric_areas(aligned, intersection, raster_crs, metadata.bounds)
        if area_error is not None:
            warnings.append(area_error)
        return AOIAnalysis(**common, aligned_original_geometry=aligned, effective_geometry=None, raster_footprint=footprint, overlap_status="outside", overlap_fraction=0.0, original_area_m2=original_area, effective_area_m2=0.0 if original_area is not None else None, area_available=original_area is not None, aoi_spatial_match_available=True, metric_crs=metric_crs, warnings=tuple(warnings), errors=tuple(errors))
    status = "inside" if aligned.within(footprint) else "partial_overlap"
    fraction = float(intersection.area / aligned.area) if aligned.area > 0 else None
    original_area, effective_area, metric_crs, area_error = _metric_areas(aligned, intersection, raster_crs, metadata.bounds)
    if area_error is not None:
        warnings.append(area_error)
    return AOIAnalysis(**common, aligned_original_geometry=aligned, effective_geometry=intersection, raster_footprint=footprint, overlap_status=status, overlap_fraction=fraction, original_area_m2=original_area, effective_area_m2=effective_area, area_available=original_area is not None and effective_area is not None, aoi_spatial_match_available=True, metric_crs=metric_crs, warnings=tuple(warnings), errors=tuple(errors))
