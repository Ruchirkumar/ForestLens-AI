from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from rasterio.io import MemoryFile
from shapely.geometry import box

from src.geometry.crown_refinement import CrownRefinementResult
from src.io.kml_reader import parse_kml
from src.io.raster_reader import metadata_from_dataset
from src.spatial.aoi import analyze_aoi
from src.spatial.aoi_filter import filter_trees_to_aoi, pixel_geometry_to_map


def _kml(*placemarks: str) -> str:
    return '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>' + ''.join(placemarks) + '</Document></kml>'


def _polygon(coordinates: str) -> str:
    return f'<Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>{coordinates}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>'


def _metadata(crs: str | None, transform: Affine, width: int = 200, height: int = 200):
    with MemoryFile() as memory_file:
        with memory_file.open(driver="GTiff", width=width, height=height, count=3, dtype="uint8", crs=crs, transform=transform) as dataset:
            dataset.write(np.zeros((3, height, width), dtype=np.uint8))
            return metadata_from_dataset(dataset, "synthetic.tif")


def _tree(tree_id: int, bounds: tuple[float, float, float, float]) -> CrownRefinementResult:
    return CrownRefinementResult(tree_id, 0.8, bounds, "fallback_bbox", 0.0, np.ones((1, 1), dtype=bool), (0, 0), 1.0, box(*bounds), "bbox_proxy")


def test_polygon_kml_is_parsed_as_wgs84_polygon():
    result = parse_kml(_kml(_polygon("0,0 1,0 1,1 0,1 0,0")), source_name="aoi.kml")
    assert result.valid and result.geometry_type == "Polygon"
    assert result.source_crs == "EPSG:4326" and result.feature_count == 1
    assert result.source_name == "aoi.kml"


def test_all_polygon_placemarks_are_combined():
    result = parse_kml(_kml(_polygon("0,0 1,0 1,1 0,1 0,0"), _polygon("2,0 3,0 3,1 2,1 2,0")))
    assert result.valid and result.feature_count == 2
    assert result.geometry_type == "MultiPolygon" and result.geometry.area == pytest.approx(2.0)


def test_multipolygon_and_nested_folder_are_parsed():
    text = _kml('<Folder><Placemark><MultiGeometry><Polygon><outerBoundaryIs><LinearRing><coordinates>0,0 1,0 1,1 0,1 0,0</coordinates></LinearRing></outerBoundaryIs></Polygon><Polygon><outerBoundaryIs><LinearRing><coordinates>2,0 3,0 3,1 2,1 2,0</coordinates></LinearRing></outerBoundaryIs></Polygon></MultiGeometry></Placemark></Folder>')
    result = parse_kml(text)
    assert result.valid and result.feature_count == 1 and result.geometry_type == "MultiPolygon"


def test_point_only_kml_is_rejected_with_clear_error():
    result = parse_kml(_kml('<Placemark><Point><coordinates>0,0</coordinates></Point></Placemark>'))
    assert not result.valid and result.geometry is None
    assert "no supported Polygon" in result.errors[0]
    assert any("Point" in warning for warning in result.warnings)


def test_self_intersecting_polygon_is_repaired_or_rejected_explicitly():
    result = parse_kml(_kml(_polygon("0,0 1,1 1,0 0,1 0,0")))
    assert result.valid or result.errors
    if result.valid:
        assert result.geometry is not None and result.geometry.is_valid
        assert any("repaired" in warning for warning in result.warnings)


def test_projected_raster_transforms_kml_and_calculates_metric_area():
    metadata = _metadata("EPSG:3857", Affine(10, 0, 0, 0, -10, 2000))
    aoi = parse_kml(_kml(_polygon("0,0 0.01,0 0.01,0.01 0,0.01 0,0")))
    analysis = analyze_aoi(aoi, metadata)
    assert analysis.aoi_spatial_match_available and analysis.overlap_status == "inside"
    assert analysis.effective_geometry is not None and analysis.area_available
    assert analysis.effective_area_m2 is not None and analysis.effective_area_m2 > 1_000_000


def test_inside_partial_and_outside_aoi_overlap_states():
    metadata = _metadata("EPSG:3857", Affine(10, 0, 0, 0, -10, 2000))
    inside = analyze_aoi(parse_kml(_kml(_polygon("0,0 0.01,0 0.01,0.01 0,0.01 0,0"))), metadata)
    partial = analyze_aoi(parse_kml(_kml(_polygon("-0.01,-0.01 0.01,-0.01 0.01,0.01 -0.01,0.01 -0.01,-0.01"))), metadata)
    outside = analyze_aoi(parse_kml(_kml(_polygon("0.03,0.03 0.04,0.03 0.04,0.04 0.03,0.04 0.03,0.03"))), metadata)
    assert inside.overlap_status == "inside"
    assert partial.overlap_status == "partial_overlap" and partial.effective_geometry is not None
    assert outside.overlap_status == "outside" and outside.effective_geometry is None
    assert outside.area_available and outside.original_area_m2 is not None and outside.effective_area_m2 == 0.0


def test_missing_raster_crs_never_matches_kml_or_reports_area():
    metadata = _metadata(None, Affine(1, 0, 0, 0, -1, 200))
    analysis = analyze_aoi(parse_kml(_kml(_polygon("0,0 1,0 1,1 0,1 0,0"))), metadata)
    assert not analysis.aoi_spatial_match_available and not analysis.area_available
    assert analysis.overlap_status == "unavailable"
    assert any("georeferencing is missing" in warning for warning in analysis.warnings)


def test_geographic_raster_is_reprojected_for_m2_not_degree_area():
    metadata = _metadata("EPSG:4326", Affine(0.01, 0, 0, 0, -0.01, 0.02), width=2, height=2)
    analysis = analyze_aoi(parse_kml(_kml(_polygon("0,0 0.01,0 0.01,0.01 0,0.01 0,0"))), metadata)
    assert analysis.overlap_status == "inside" and analysis.area_available
    assert analysis.metric_crs is not None and analysis.metric_crs != "EPSG:4326"
    assert analysis.effective_area_m2 is not None and analysis.effective_area_m2 > 1_000_000


def test_tree_filter_includes_inside_and_boundary_trees_but_not_outside():
    metadata = _metadata("EPSG:3857", Affine(1, 0, 0, 0, -1, 20), width=20, height=20)
    aoi = parse_kml(_kml(_polygon("0,0 0.00009,0 0.00009,0.00009 0,0.00009 0,0")))
    analysis = analyze_aoi(aoi, metadata)
    result = filter_trees_to_aoi([_tree(1, (2, 12, 4, 14)), _tree(2, (9, 9, 12, 12)), _tree(3, (14, 12, 16, 14))], metadata, analysis)
    assert result.included_tree_ids == (1, 2)
    assert result.tree_results[1].clipped_geometry is not None
    assert result.tree_results[1].intersection_area_m2 is not None


def test_pixel_geometry_to_map_uses_the_full_affine_transform():
    metadata = _metadata("EPSG:3857", Affine(2, 0.5, 100, 0.25, -3, 200))
    mapped = pixel_geometry_to_map(box(1, 2, 4, 6), metadata)
    assert mapped.area == pytest.approx(73.5)
    assert mapped.bounds == pytest.approx((103.0, 182.25, 111.0, 195.0))
