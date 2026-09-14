from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from rasterio.io import MemoryFile
from shapely.geometry import box

from src.geometry.crown_refinement import CrownRefinementResult
from src.geometry.area import m2_to_hectares
from src.io.kml_reader import parse_kml
from src.io.raster_reader import metadata_from_dataset
from src.metrics.analysis import calculate_analysis_metrics
from src.spatial.aoi import AOIAnalysis, analyze_aoi
from src.spatial.aoi_filter import AOIFilterResult, AOITreeResult


def _metadata(crs: str | None = "EPSG:3857", width: int = 10, height: int = 10):
    with MemoryFile() as memory_file:
        with memory_file.open(driver="GTiff", width=width, height=height, count=3, dtype="uint8", crs=crs, transform=Affine(1, 0, 0, 0, -1, height)) as dataset:
            dataset.write(np.zeros((3, height, width), dtype=np.uint8))
            return metadata_from_dataset(dataset, "metrics.tif")


def _analysis(area: float = 100.0, *, available: bool = True) -> AOIAnalysis:
    effective = box(0, 0, 10, 10) if available else None
    return AOIAnalysis(
        original_geometry=effective, aligned_original_geometry=effective, effective_geometry=effective,
        raster_footprint=effective, overlap_status="inside", overlap_fraction=1.0,
        original_area_m2=area if available else None, effective_area_m2=area if available else None,
        area_available=available, aoi_spatial_match_available=available, source_crs="EPSG:4326",
        raster_crs="EPSG:3857" if available else None, metric_crs="EPSG:3857" if available else None,
        warnings=(), errors=(),
    )


def _tree(tree_id: int, status: str = "refined", score: float = 0.8, confidence: float = 0.6) -> CrownRefinementResult:
    return CrownRefinementResult(tree_id, score, (0, 0, 1, 1), status, confidence, np.ones((1, 1), dtype=bool), (0, 0), 1.0, box(0, 0, 1, 1), "refined_mask" if status == "refined" else "bbox_proxy")  # type: ignore[arg-type]


def _filter(*items: tuple[int, object]) -> AOIFilterResult:
    return AOIFilterResult(tuple(AOITreeResult(tree_id, "refined_crown", geometry, True, True, geometry, None) for tree_id, geometry in items), True, ())


def test_basic_count_area_hectares_density_and_cover():
    metrics = calculate_analysis_metrics([_tree(1)], _filter((1, box(0, 0, 5, 5))), _analysis(), _metadata())
    assert metrics.detected_tree_count == 1
    assert metrics.aoi_area_ha == pytest.approx(0.01)
    assert metrics.estimated_canopy_area_m2 == pytest.approx(25.0)
    assert metrics.estimated_canopy_cover_percent == pytest.approx(25.0)
    assert metrics.detected_tree_density_per_ha == pytest.approx(100.0)
    assert m2_to_hectares(10_000.0) == 1.0


def test_empty_detections_have_no_statistics_but_valid_zero_canopy():
    metrics = calculate_analysis_metrics([], _filter(), _analysis(), _metadata())
    assert metrics.detected_tree_count == 0 and metrics.refinement_rate is None
    assert metrics.mean_detection_score is None and metrics.estimated_canopy_area_m2 == 0.0


def test_overlapping_footprints_use_union_not_individual_sum():
    metrics = calculate_analysis_metrics([_tree(1, "refined"), _tree(2, "fallback_bbox")], _filter((1, box(0, 0, 4, 4)), (2, box(2, 0, 6, 4))), _analysis(), _metadata())
    assert metrics.refined_crown_area_m2 == pytest.approx(16.0)
    assert metrics.bbox_fallback_area_m2 == pytest.approx(16.0)
    assert metrics.sum_individual_crown_area_m2 == pytest.approx(32.0)
    assert metrics.estimated_canopy_area_m2 == pytest.approx(24.0)
    assert metrics.estimated_canopy_cover_percent == pytest.approx(24.0)


def test_boundary_clipped_geometry_only_contributes_inside_area():
    metrics = calculate_analysis_metrics([_tree(1)], _filter((1, box(8, 0, 10, 4))), _analysis(), _metadata())
    assert metrics.estimated_canopy_area_m2 == pytest.approx(8.0)


def test_refined_fallback_and_failed_tracking_excludes_failed_canopy_area():
    trees = [_tree(1, "refined"), _tree(2, "fallback_bbox"), _tree(3, "failed")]
    metrics = calculate_analysis_metrics(trees, _filter((1, box(0, 0, 2, 2)), (2, box(2, 0, 4, 2)), (3, box(4, 0, 9, 9))), _analysis(), _metadata())
    assert (metrics.refined_tree_count, metrics.bbox_fallback_count, metrics.failed_tree_count) == (1, 1, 1)
    assert metrics.total_detections_before_aoi == 3
    assert metrics.refinement_rate == pytest.approx(1 / 3)
    assert metrics.estimated_canopy_area_m2 == pytest.approx(8.0)


def test_detection_and_refinement_statistics_are_model_indicators_only():
    trees = [_tree(1, score=0.2, confidence=0.1), _tree(2, score=0.5, confidence=0.4), _tree(3, score=0.8, confidence=0.7)]
    metrics = calculate_analysis_metrics(trees, _filter((1, box(0, 0, 1, 1)), (2, box(1, 0, 2, 1)), (3, box(2, 0, 3, 1))), _analysis(), _metadata())
    assert metrics.mean_detection_score == pytest.approx(0.5)
    assert metrics.median_detection_score == pytest.approx(0.5)
    assert metrics.mean_refinement_confidence == pytest.approx(0.4)


def test_missing_crs_keeps_counts_but_withholds_all_physical_metrics():
    metrics = calculate_analysis_metrics([_tree(1)], _filter((1, box(0, 0, 1, 1))), _analysis(available=False), _metadata(None))
    assert metrics.detected_tree_count == 1 and not metrics.physical_metrics_available
    assert metrics.aoi_area_m2 is None and metrics.estimated_canopy_area_m2 is None
    assert metrics.measurement_basis == "pixel_only"


def test_geographic_raster_is_reprojected_before_metric_measurement():
    with MemoryFile() as memory_file:
        with memory_file.open(driver="GTiff", width=2, height=2, count=3, dtype="uint8", crs="EPSG:4326", transform=Affine(0.01, 0, 0, 0, -0.01, 0.02)) as dataset:
            dataset.write(np.zeros((3, 2, 2), dtype=np.uint8))
            metadata = metadata_from_dataset(dataset, "geographic.tif")
    kml = '<kml xmlns="http://www.opengis.net/kml/2.2"><Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>0,0 0.01,0 0.01,0.01 0,0.01 0,0</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></kml>'
    analysis = analyze_aoi(parse_kml(kml), metadata)
    metrics = calculate_analysis_metrics([_tree(1)], _filter((1, box(0, 0, 0.01, 0.01))), analysis, metadata)
    assert metrics.physical_metrics_available and metrics.measurement_basis == "geographic_raster_reprojected_to_metric_crs"
    assert metrics.estimated_canopy_area_m2 is not None and metrics.estimated_canopy_area_m2 > 1_000_000


def test_projected_metric_raster_reports_direct_metric_provenance():
    metrics = calculate_analysis_metrics([_tree(1)], _filter((1, box(0, 0, 1, 1))), _analysis(), _metadata())
    assert metrics.physical_metrics_available and metrics.measurement_basis == "georeferenced_metric_raster"


def test_zero_area_aoi_fails_clearly():
    with pytest.raises(ValueError, match="greater than zero"):
        calculate_analysis_metrics([_tree(1)], _filter((1, box(0, 0, 1, 1))), _analysis(0.0), _metadata())
