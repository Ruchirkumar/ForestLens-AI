import numpy as np
import pandas as pd

from src.visualization.overlay import render_detection_overlay


def _detections(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["tree_id", "xmin", "ymin", "xmax", "ymax", "score"])


def test_overlay_handles_zero_detections_without_mutating_image():
    image = np.full((12, 18, 3), 120, dtype=np.uint8)
    original = image.copy()

    rendered = render_detection_overlay(image, _detections([]))

    assert rendered.shape == image.shape
    assert np.array_equal(image, original)
    assert np.array_equal(rendered, original)


def test_overlay_handles_one_and_multiple_detections_without_mutating_dataframe():
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    detections = _detections(
        [
            {"tree_id": 1, "xmin": 2, "ymin": 3, "xmax": 14, "ymax": 16, "score": 0.8},
            {"tree_id": 2, "xmin": 16, "ymin": 18, "xmax": 28, "ymax": 29, "score": 0.7},
        ]
    )
    original = detections.copy(deep=True)

    rendered = render_detection_overlay(image, detections)

    assert rendered.shape == image.shape
    assert not np.array_equal(rendered, image)
    pd.testing.assert_frame_equal(detections, original)


def test_overlay_clips_boxes_to_image_boundaries():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    detections = _detections(
        [{"tree_id": 1, "xmin": -5, "ymin": -5, "xmax": 30, "ymax": 30, "score": 0.9}]
    )
    rendered = render_detection_overlay(image, detections)

    assert rendered.shape == (10, 10, 3)
    assert rendered[0, 9].tolist() == [255, 64, 64]
