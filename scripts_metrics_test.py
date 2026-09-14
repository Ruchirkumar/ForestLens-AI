"""Demonstrate Step 7 AOI metrics and measurement provenance without network use."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.io import MemoryFile
from shapely.geometry import box

from src.geometry.crown_refinement import CrownRefinementResult
from src.io.kml_reader import parse_kml
from src.io.raster_reader import metadata_from_dataset, read_raster_metadata
from src.metrics.analysis import calculate_analysis_metrics
from src.spatial.aoi import AOIAnalysis, analyze_aoi
from src.spatial.aoi_filter import AOIFilterResult, AOITreeResult


def _tree(tree_id: int, status: str, score: float, confidence: float) -> CrownRefinementResult:
    return CrownRefinementResult(tree_id, score, (0, 0, 1, 1), status, confidence, np.ones((1, 1), dtype=bool), (0, 0), 1.0, box(0, 0, 1, 1), "refined_mask" if status == "refined" else "bbox_proxy")  # type: ignore[arg-type]


def _metric_example():
    with MemoryFile() as memory_file:
        with memory_file.open(driver="GTiff", width=10, height=10, count=3, dtype="uint8", crs="EPSG:3857", transform=Affine(1, 0, 0, 0, -1, 10)) as dataset:
            dataset.write(np.zeros((3, 10, 10), dtype=np.uint8))
            metadata = metadata_from_dataset(dataset, "synthetic_metric.tif")
    effective_aoi = box(0, 0, 10, 10)
    analysis = AOIAnalysis(effective_aoi, effective_aoi, effective_aoi, effective_aoi, "inside", 1.0, 100.0, 100.0, True, True, "EPSG:4326", "EPSG:3857", "EPSG:3857", (), ())
    trees = [_tree(1, "refined", 0.9, 0.8), _tree(2, "refined", 0.7, 0.6), _tree(3, "fallback_bbox", 0.5, 0.0)]
    # Trees 1 and 2 overlap; tree 3 has already been clipped to the AOI edge.
    filtered = AOIFilterResult((
        AOITreeResult(1, "refined_crown", box(0, 0, 4, 4), True, True, box(0, 0, 4, 4), None),
        AOITreeResult(2, "refined_crown", box(2, 0, 6, 4), True, True, box(2, 0, 6, 4), None),
        AOITreeResult(3, "bbox_proxy", box(8, 0, 10, 4), True, True, box(8, 0, 10, 4), None),
    ), True, ())
    return calculate_analysis_metrics(trees, filtered, analysis, metadata)


def _kml() -> str:
    return '<kml xmlns="http://www.opengis.net/kml/2.2"><Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>0,0 0.01,0 0.01,0.01 0,0.01 0,0</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></kml>'


def _osbs_sample_path() -> Path:
    """Locate the installed DeepForest fixture without importing its ML stack."""
    spec = find_spec("deepforest")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("DeepForest package is unavailable.")
    return Path(next(iter(spec.submodule_search_locations))) / "data" / "OSBS_029.png"


def main() -> int:
    try:
        metrics = _metric_example()
        print("Synthetic georeferenced metric example")
        print(f"Detected trees: {metrics.detected_tree_count}")
        print(f"Refined: {metrics.refined_tree_count}")
        print(f"BBox fallbacks: {metrics.bbox_fallback_count}")
        print(f"AOI area: {metrics.aoi_area_m2:.2f} m2")
        print(f"Estimated canopy area: {metrics.estimated_canopy_area_m2:.2f} m2")
        print(f"Estimated canopy cover: {metrics.estimated_canopy_cover_percent:.2f}%")
        print(f"Detected-tree density: {metrics.detected_tree_density_per_ha:.2f} trees/ha")
        print(f"Mean detection score: {metrics.mean_detection_score:.3f}")
        print(f"Median detection score: {metrics.median_detection_score:.3f}")
        print(f"Refinement rate: {metrics.refinement_rate:.3f}")
        print(f"Measurement basis: {metrics.measurement_basis}")
        print("Overlapping footprints are unioned; AOI-edge geometry is clipped before measurement.")

        png_metadata = read_raster_metadata(_osbs_sample_path())
        unavailable = analyze_aoi(parse_kml(_kml()), png_metadata)
        empty_filter = AOIFilterResult((), False, ())
        png_metrics = calculate_analysis_metrics([], empty_filter, unavailable, png_metadata)
        print("\nMissing-CRS example (OSBS_029.png)")
        print("Physical metrics unavailable because raster lacks valid georeferencing." if not png_metrics.physical_metrics_available else "Physical metrics unexpectedly available.")
        print(f"Measurement basis: {png_metrics.measurement_basis}")
        return 0
    except Exception as error:
        print(f"Metrics test failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
