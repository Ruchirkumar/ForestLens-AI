"""Filter pixel-space tree detections against a CRS-aligned effective AOI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from shapely import affinity
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from src.geometry.crown_refinement import CrownRefinementResult
from src.models import RasterMetadata
from src.spatial.aoi import AOIAnalysis, transform_geometry


@dataclass(frozen=True)
class AOITreeResult:
    """One detection's AOI relationship in raster CRS coordinates."""

    tree_id: int
    geometry_source: str
    map_geometry: BaseGeometry | None
    intersects_aoi: bool | None
    included: bool
    clipped_geometry: BaseGeometry | None
    intersection_area_m2: float | None


@dataclass(frozen=True)
class AOIFilterResult:
    """AOI filtering result; boundary-intersecting trees are retained whole."""

    tree_results: tuple[AOITreeResult, ...]
    spatial_match_available: bool
    warnings: tuple[str, ...]

    @property
    def included_tree_ids(self) -> tuple[int, ...]:
        return tuple(item.tree_id for item in self.tree_results if item.included)


def pixel_geometry_to_map(geometry: BaseGeometry, metadata: RasterMetadata) -> BaseGeometry:
    """Apply the raster affine transform to image-pixel geometry."""
    transform = metadata.transform
    return affinity.affine_transform(geometry, [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f])


def _tree_pixel_geometry(tree: CrownRefinementResult) -> tuple[BaseGeometry | None, str]:
    if tree.refinement_status == "refined" and tree.crown_polygon is not None:
        return tree.crown_polygon, "refined_crown"
    left, top, right, bottom = tree.bbox
    return (box(left, top, right, bottom), "bbox_proxy") if right > left and bottom > top else (None, "invalid_bbox")


def filter_trees_to_aoi(trees: Sequence[CrownRefinementResult], metadata: RasterMetadata, analysis: AOIAnalysis) -> AOIFilterResult:
    """Include each tree whose refined crown or bbox proxy intersects the AOI.

    ``clipped_geometry`` is retained for later canopy-area work. Step 6 does
    not fractionally count edge trees: an intersecting detection is included once.
    """
    if not analysis.aoi_spatial_match_available or analysis.effective_geometry is None:
        return AOIFilterResult(tuple(AOITreeResult(tree.tree_id, "unavailable", None, None, False, None, None) for tree in trees), False, ("Tree AOI filtering is unavailable because AOI/raster spatial matching is unavailable.",))
    results: list[AOITreeResult] = []
    for tree in trees:
        pixel_geometry, source = _tree_pixel_geometry(tree)
        if pixel_geometry is None:
            results.append(AOITreeResult(tree.tree_id, source, None, False, False, None, None))
            continue
        map_geometry = pixel_geometry_to_map(pixel_geometry, metadata)
        intersection = map_geometry.intersection(analysis.effective_geometry)
        intersects = not intersection.is_empty
        area_m2: float | None = None
        if intersects and analysis.metric_crs is not None and metadata.crs is not None:
            area_m2 = float(transform_geometry(intersection, metadata.crs, analysis.metric_crs).area)
        results.append(AOITreeResult(tree.tree_id, source, map_geometry, intersects, intersects, intersection if intersects else None, area_m2))
    return AOIFilterResult(tuple(results), True, ())
