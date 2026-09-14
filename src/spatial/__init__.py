"""Spatial utilities, including CRS-safe AOI integration."""

from .aoi import AOIAnalysis, analyze_aoi, raster_footprint, transform_geometry
from .aoi_filter import AOIFilterResult, AOITreeResult, filter_trees_to_aoi, pixel_geometry_to_map

__all__ = ["AOIAnalysis", "AOIFilterResult", "AOITreeResult", "analyze_aoi", "filter_trees_to_aoi", "pixel_geometry_to_map", "raster_footprint", "transform_geometry"]
