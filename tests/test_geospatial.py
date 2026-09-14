from dataclasses import replace

import numpy as np
import pytest
from affine import Affine
from rasterio.io import MemoryFile

from src.geometry.area import calculate_polygon_area_m2, calculate_raster_area_m2, get_pixel_area_m2
from src.geometry.crs import choose_metric_crs, map_to_pixel, pixel_bbox_to_map_polygon, pixel_to_map
from src.io.raster_reader import (
    RasterValidationError,
    metadata_from_dataset,
    validate_raster_metadata,
    validate_rgb_band_count,
)


def _metadata(
    *,
    crs: str | None,
    transform: Affine,
    band_count: int = 3,
    width: int = 10,
    height: int = 8,
):
    with MemoryFile() as memory_file:
        with memory_file.open(
            driver="GTiff",
            width=width,
            height=height,
            count=band_count,
            dtype="uint8",
            crs=crs,
            transform=transform,
        ) as dataset:
            dataset.write(np.zeros((band_count, height, width), dtype=np.uint8))
            return metadata_from_dataset(dataset, "memory.tif")


def test_projected_metre_metadata_has_metric_gsd_and_pixel_area():
    metadata = _metadata(
        crs="EPSG:32645",
        transform=Affine(0.5, 0, 500000, 0, -0.8, 2000000),
    )

    assert metadata.crs_type == "projected"
    assert metadata.georeferenced and metadata.projected_crs
    assert metadata.pixel_width == 0.5
    assert metadata.pixel_height == 0.8
    assert metadata.pixel_area_native == pytest.approx(0.4)
    assert metadata.gsd_x_m == 0.5 and metadata.gsd_y_m == 0.8
    assert get_pixel_area_m2(metadata) == pytest.approx(0.4)
    assert calculate_raster_area_m2(metadata) == pytest.approx(32.0)


def test_geographic_crs_never_returns_fake_metre_area():
    metadata = _metadata(
        crs="EPSG:4326",
        transform=Affine(0.0001, 0, 77, 0, -0.0001, 20),
    )

    assert metadata.crs_type == "geographic"
    assert metadata.georeferenced and metadata.geographic_crs
    assert metadata.gsd_x_m is None and metadata.gsd_y_m is None
    assert get_pixel_area_m2(metadata) is None
    assert calculate_raster_area_m2(metadata) is None
    metric_crs, reason = choose_metric_crs(metadata.crs, metadata.bounds)
    assert metric_crs is not None and metric_crs.is_projected
    assert "UTM zone" in reason


def test_missing_crs_remains_detection_compatible_but_has_no_physical_area():
    metadata = _metadata(crs=None, transform=Affine(1, 0, 0, 0, -1, 8))

    assert metadata.crs_type == "missing"
    assert not metadata.georeferenced
    assert not metadata.physical_area_available
    assert get_pixel_area_m2(metadata) is None
    assert "no valid projected metric CRS" in metadata.physical_area_reason


def test_non_square_and_rotated_pixels_use_affine_determinant():
    metadata = _metadata(
        crs="EPSG:32645",
        transform=Affine(2, 0.5, 500000, 0.25, -3, 2000000),
    )

    assert metadata.pixel_width == 2.0
    assert metadata.pixel_height == 3.0
    assert metadata.pixel_area_native == pytest.approx(6.125)
    assert get_pixel_area_m2(metadata) == pytest.approx(6.125)


@pytest.mark.parametrize("band_count", [1, 3, 4])
def test_safe_rgb_band_counts_are_accepted(band_count):
    validate_rgb_band_count(band_count)


def test_multispectral_band_count_is_rejected():
    with pytest.raises(RasterValidationError, match="multispectral"):
        validate_rgb_band_count(5)


def test_pixel_map_round_trip_and_bbox_polygon():
    metadata = _metadata(
        crs="EPSG:32645",
        transform=Affine(2, 0, 100, 0, -3, 200),
    )

    x_coordinate, y_coordinate = pixel_to_map(metadata, 4, 5)
    assert map_to_pixel(metadata, x_coordinate, y_coordinate) == (4, 5)
    map_polygon = pixel_bbox_to_map_polygon(metadata, (1, 2, 4, 6))
    assert map_polygon.is_valid
    assert map_polygon.area == pytest.approx(72.0)
    assert calculate_polygon_area_m2(map_polygon, metadata.crs) == pytest.approx(72.0)
    assert calculate_polygon_area_m2(map_polygon, "EPSG:4326") is None


def test_invalid_metadata_has_a_clear_validation_failure():
    metadata = _metadata(crs="EPSG:32645", transform=Affine(1, 0, 0, 0, -1, 8))

    with pytest.raises(RasterValidationError, match="dimensions"):
        validate_raster_metadata(replace(metadata, width=0, total_pixel_count=0))
