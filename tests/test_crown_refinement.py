import numpy as np
import pandas as pd

from src.geometry.crown_refinement import refine_crown, refine_crowns


def _image(height: int = 60, width: int = 60) -> np.ndarray:
    return np.full((height, width, 3), (80, 80, 80), dtype=np.uint8)


def _green_disk(image: np.ndarray, center: tuple[int, int], radius: int) -> None:
    rows, columns = np.ogrid[: image.shape[0], : image.shape[1]]
    selected = (rows - center[1]) ** 2 + (columns - center[0]) ** 2 <= radius**2
    image[selected] = (20, 180, 20)


def _predictions(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["tree_id", "score", "xmin", "ymin", "xmax", "ymax"],
    )


def test_green_object_at_detection_center_produces_valid_refined_mask():
    image = _image()
    _green_disk(image, (30, 30), 9)

    result = refine_crown(image, 1, 0.9, (15, 15, 45, 45))

    assert result.refinement_status == "refined"
    assert result.mask.any()
    assert result.crown_polygon is not None and result.crown_polygon.is_valid
    assert result.crown_area_pixels == float(result.mask.sum())
    assert 0.0 <= result.refinement_confidence <= 1.0


def test_non_vegetated_patch_uses_explicit_bbox_proxy_fallback():
    result = refine_crown(_image(), 1, 0.7, (10, 10, 30, 30))

    assert result.refinement_status == "fallback_bbox"
    assert result.area_source == "bbox_proxy"
    assert result.crown_area_pixels == 400.0
    assert result.crown_polygon is not None and result.crown_polygon.is_valid


def test_extremely_large_local_mask_uses_fallback():
    image = np.full((60, 60, 3), (20, 180, 20), dtype=np.uint8)

    result = refine_crown(image, 1, 0.7, (10, 10, 50, 50))

    assert result.refinement_status == "fallback_bbox"
    assert result.area_source == "bbox_proxy"


def test_extremely_small_local_mask_uses_fallback():
    image = _image()
    image[29:31, 29:31] = (20, 180, 20)

    result = refine_crown(image, 1, 0.7, (15, 15, 45, 45))

    assert result.refinement_status == "fallback_bbox"


def test_center_aware_selection_prefers_center_component_over_neighbour():
    image = _image()
    _green_disk(image, (30, 30), 7)
    _green_disk(image, (16, 16), 9)

    result = refine_crown(image, 1, 0.8, (10, 10, 50, 50))

    assert result.refinement_status == "refined"
    assert result.mask_origin[0] <= 30 < result.mask_origin[0] + result.mask.shape[1]
    assert result.mask_origin[1] <= 30 < result.mask_origin[1] + result.mask.shape[0]
    center_row = 30 - result.mask_origin[1]
    center_column = 30 - result.mask_origin[0]
    assert result.mask[center_row, center_column]
    assert not result.mask[16 - result.mask_origin[1], 16 - result.mask_origin[0]]


def test_out_of_bounds_bbox_is_clipped_safely():
    result = refine_crown(_image(), 1, 0.8, (-10, -8, 20, 24))

    assert result.refinement_status == "fallback_bbox"
    assert result.bbox == (0.0, 0.0, 20.0, 24.0)
    assert result.crown_area_pixels == 480.0


def test_refinement_preserves_original_image_and_predictions_and_handles_zero_multiple():
    image = _image()
    _green_disk(image, (20, 20), 6)
    _green_disk(image, (40, 40), 6)
    image_before = image.copy()
    predictions = _predictions(
        [
            {"tree_id": 1, "score": 0.8, "xmin": 8, "ymin": 8, "xmax": 32, "ymax": 32},
            {"tree_id": 2, "score": 0.7, "xmin": 28, "ymin": 28, "xmax": 52, "ymax": 52},
        ]
    )
    predictions_before = predictions.copy(deep=True)

    assert refine_crowns(image, _predictions([])) == []
    results = refine_crowns(image, predictions)

    assert len(results) == 2
    assert all(result.refinement_status == "refined" for result in results)
    assert np.array_equal(image, image_before)
    pd.testing.assert_frame_equal(predictions, predictions_before)
    assert all(0.0 <= result.refinement_confidence <= 1.0 for result in results)
