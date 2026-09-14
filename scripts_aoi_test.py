"""Demonstrate deterministic KML AOI parsing and CRS-safe raster matching."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.io import MemoryFile

from src.io.kml_reader import parse_kml
from src.io.raster_reader import metadata_from_dataset, read_raster_metadata
from src.spatial.aoi import analyze_aoi


def _fixture_kml() -> str:
    return ('<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>'
            '0,0 0.01,0 0.01,0.01 0,0.01 0,0'
            '</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></Document></kml>')


def _synthetic_metadata():
    with MemoryFile() as memory_file:
        with memory_file.open(driver="GTiff", width=200, height=200, count=3, dtype="uint8", crs="EPSG:3857", transform=Affine(10, 0, 0, 0, -10, 2000)) as dataset:
            dataset.write(np.zeros((3, 200, 200), dtype=np.uint8))
            return metadata_from_dataset(dataset, "synthetic_3857.tif")


def _osbs_sample_path() -> Path:
    """Locate the installed DeepForest fixture without importing its ML stack."""
    spec = find_spec("deepforest")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("DeepForest package is unavailable.")
    return Path(next(iter(spec.submodule_search_locations))) / "data" / "OSBS_029.png"


def main() -> int:
    try:
        aoi = parse_kml(_fixture_kml(), source_name="synthetic_aoi.kml")
        print("KML parsing")
        print(f"Geometry type: {aoi.geometry_type or 'unavailable'}")
        print(f"Valid: {aoi.valid}")
        print(f"Feature count: {aoi.feature_count}")
        print(f"Source CRS: {aoi.source_crs}")
        analysis = analyze_aoi(aoi, _synthetic_metadata())
        print("\nCRS transformation and AOI/raster overlap")
        print(f"Raster CRS: {analysis.raster_crs}")
        print(f"Spatial match available: {analysis.aoi_spatial_match_available}")
        print(f"Overlap: {analysis.overlap_status}")
        print(f"Physical area available: {analysis.area_available}")
        print(f"Effective AOI area: {analysis.effective_area_m2:.2f} m2" if analysis.effective_area_m2 is not None else "Effective AOI area: unavailable")
        missing_crs = analyze_aoi(aoi, read_raster_metadata(_osbs_sample_path()))
        print("\nMissing-CRS behavior (OSBS_029.png)")
        print(f"Raster CRS: {missing_crs.raster_crs or 'missing'}")
        print(f"Spatial match available: {missing_crs.aoi_spatial_match_available}")
        print(f"Overlap: {missing_crs.overlap_status}")
        print(f"Physical area available: {missing_crs.area_available}")
        for warning in missing_crs.warnings:
            print(f"Warning: {warning}")
        return 0
    except Exception as error:
        print(f"AOI test failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
