from __future__ import annotations

import cv2
import numpy as np
from affine import Affine
from rasterio.io import MemoryFile

from src.geometry.crown_refinement import CrownRefinementResult
from src.io.raster_reader import metadata_from_dataset
from src.quality.assessment import QualityConfig, evaluate_quality
from src.quality.warnings import generate_warnings
from src.uncertainty.sensitivity import calculate_refinement_summary


def _metadata(crs: str | None = "EPSG:3857", gsd: float = 0.5):
    with MemoryFile() as memory_file:
        with memory_file.open(driver="GTiff", width=32, height=32, count=3, dtype="uint8", crs=crs, transform=Affine(gsd, 0, 0, 0, -gsd, 16)) as dataset:
            dataset.write(np.zeros((3, 32, 32), dtype=np.uint8))
            return metadata_from_dataset(dataset, "quality.tif")


def _textured(value: int = 128, spread: int = 60) -> np.ndarray:
    rows, columns = np.indices((32, 32))
    image = np.full((32, 32, 3), value, dtype=np.uint8)
    image[(rows + columns) % 2 == 0] = np.clip(value - spread, 0, 255)
    image[(rows + columns) % 2 == 1] = np.clip(value + spread, 0, 255)
    return image


def test_bright_and_dark_textured_images_are_diagnostics_not_fake_scores():
    bright = evaluate_quality(_textured(235, 15), _metadata())
    dark = evaluate_quality(_textured(18, 15), _metadata())
    assert bright.status == "WARNING" and bright.can_analyze
    assert dark.status == "WARNING" and dark.can_analyze
    assert bright.brightness > 225 and dark.brightness < 30


def test_low_contrast_is_warning_and_uniform_image_is_blocked():
    low_contrast = np.full((32, 32, 3), 128, dtype=np.uint8)
    low_contrast[::2] += 1
    warning = evaluate_quality(low_contrast, _metadata())
    blocked = evaluate_quality(np.full((32, 32, 3), 128, dtype=np.uint8), _metadata())
    assert warning.status == "WARNING" and warning.can_analyze
    assert blocked.status == "BLOCKED" and not blocked.can_analyze


def test_sharp_image_has_higher_laplacian_variance_than_blurred_image():
    sharp_image = _textured()
    blurred_image = cv2.GaussianBlur(sharp_image, (9, 9), 0)
    sharp = evaluate_quality(sharp_image, _metadata())
    blurred = evaluate_quality(blurred_image, _metadata())
    assert sharp.sharpness > blurred.sharpness


def test_missing_gsd_warns_but_does_not_block_quality_gate():
    assessment = evaluate_quality(_textured(), _metadata(None))
    assert assessment.resolution_status == "resolution_unknown"
    assert assessment.status == "WARNING" and assessment.can_analyze


def test_good_warning_and_blocking_gsd_are_configured_gate_results():
    image = _textured()
    assert evaluate_quality(image, _metadata(gsd=0.5)).resolution_status == "good"
    assert evaluate_quality(image, _metadata(gsd=2.0)).resolution_status == "warning"
    blocked = evaluate_quality(image, _metadata(gsd=6.0))
    assert blocked.resolution_status == "blocked" and blocked.status == "BLOCKED"


def test_structured_warnings_cover_missing_georef_and_fallback_reliability():
    trees = [
        CrownRefinementResult(1, 0.8, (0, 0, 1, 1), "fallback_bbox", 0.0, np.ones((1, 1), bool), (0, 0), 1.0, None, "bbox_proxy"),
        CrownRefinementResult(2, 0.8, (0, 0, 1, 1), "fallback_bbox", 0.0, np.ones((1, 1), bool), (0, 0), 1.0, None, "bbox_proxy"),
    ]
    assessment = evaluate_quality(_textured(), _metadata(None))
    codes = {warning.code for warning in generate_warnings(assessment, calculate_refinement_summary(trees), missing_georeference=True)}
    assert {"MISSING_GEOREFERENCE", "UNKNOWN_RESOLUTION", "HIGH_FALLBACK_RATE", "LOW_REFINEMENT_RATE", "DOMAIN_WARNING"}.issubset(codes)


def test_quality_config_rejects_inverted_thresholds():
    try:
        QualityConfig(sharpness_warn_threshold=1, sharpness_block_threshold=2).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("Inverted sharpness thresholds must fail.")
