from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from affine import Affine
from rasterio.io import MemoryFile
from shapely.geometry import box

from src.geometry.crown_refinement import CrownRefinementResult
from src.io.raster_reader import metadata_from_dataset
from src.spatial.aoi import AOIAnalysis
from src.uncertainty.sensitivity import calculate_refinement_summary, run_threshold_sensitivity


def _metadata():
    with MemoryFile() as memory_file:
        with memory_file.open(driver="GTiff", width=20, height=20, count=3, dtype="uint8", crs="EPSG:3857", transform=Affine(1, 0, 0, 0, -1, 20)) as dataset:
            dataset.write(np.zeros((3, 20, 20), dtype=np.uint8))
            return metadata_from_dataset(dataset, "sensitivity.tif")


def _analysis() -> AOIAnalysis:
    polygon = box(0, 0, 20, 20)
    return AOIAnalysis(polygon, polygon, polygon, polygon, "inside", 1.0, 400.0, 400.0, True, True, "EPSG:4326", "EPSG:3857", "EPSG:3857", (), ())


def _predictions() -> pd.DataFrame:
    return pd.DataFrame({
        "xmin": [1, 5, 9], "ymin": [1, 5, 9], "xmax": [4, 8, 12], "ymax": [4, 8, 12],
        "label": ["Tree", "Tree", "Tree"], "score": [0.65, 0.45, 0.35],
        "geometry": [box(1, 1, 4, 4), box(5, 5, 8, 8), box(9, 9, 12, 12)],
    })


def test_refinement_summary_has_transparent_rates():
    results = [
        CrownRefinementResult(1, 0.8, (0, 0, 1, 1), "refined", 0.5, np.ones((1, 1), bool), (0, 0), 1, box(0, 0, 1, 1), "refined_mask"),
        CrownRefinementResult(2, 0.8, (0, 0, 1, 1), "fallback_bbox", 0.0, np.ones((1, 1), bool), (0, 0), 1, box(0, 0, 1, 1), "bbox_proxy"),
        CrownRefinementResult(3, 0.8, (0, 0, 0, 0), "failed", 0.0, np.zeros((0, 0), bool), (0, 0), 0, None, "none"),
    ]
    summary = calculate_refinement_summary(results)
    assert (summary.total_detections, summary.refined, summary.bbox_fallback, summary.failed) == (3, 1, 1, 1)
    assert summary.refinement_rate == pytest.approx(1 / 3) and summary.fallback_rate == pytest.approx(1 / 3)


def test_sensitivity_uses_same_raw_predictions_and_returns_ordered_rows():
    rows = run_threshold_sensitivity(_predictions(), np.full((20, 20, 3), (20, 180, 20), dtype=np.uint8), thresholds=(0.30, 0.40, 0.50, 0.60), metadata=_metadata(), aoi_analysis=_analysis())
    assert [row.threshold for row in rows] == [0.30, 0.40, 0.50, 0.60]
    assert [row.detected_tree_count for row in rows] == [3, 2, 1, 1]
    assert all(row.estimated_canopy_area_m2 is not None for row in rows)


def test_sensitivity_does_not_need_or_call_model_inference_and_keeps_missing_metrics_none():
    rows = run_threshold_sensitivity(_predictions(), np.full((20, 20, 3), 128, dtype=np.uint8), thresholds=(0.30, 0.60))
    assert [row.detected_tree_count for row in rows] == [3, 1]
    assert all(row.estimated_canopy_area_m2 is None and row.detected_tree_density_per_ha is None for row in rows)


def test_sensitivity_rejects_unsorted_or_duplicate_thresholds():
    with pytest.raises(ValueError, match="unique and sorted"):
        run_threshold_sensitivity(_predictions(), np.zeros((20, 20, 3), dtype=np.uint8), thresholds=(0.6, 0.3))
