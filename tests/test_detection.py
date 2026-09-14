import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.detection.model import (
    DeepForestDetectionError,
    ImageValidationError,
    filter_detections,
    image_to_rgb_float32,
    validate_prediction_schema,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "xmin": [20, 5, 10],
            "ymin": [0, 5, 10],
            "xmax": [30, 15, 20],
            "ymax": [10, 15, 20],
            "label": ["Tree", "Tree", "Tree"],
            "score": [0.40, 0.90, 0.70],
            "geometry": [None, None, None],
        }
    )


@pytest.mark.parametrize("threshold", [-0.01, 1.01, "0.3", True])
def test_filter_rejects_invalid_score_threshold(threshold):
    with pytest.raises(ValueError):
        filter_detections(_predictions(), threshold)


def test_filter_preserves_raw_predictions_and_assigns_deterministic_ids():
    raw = _predictions()
    original = raw.copy(deep=True)

    filtered = filter_detections(raw, 0.30)

    pd.testing.assert_frame_equal(raw, original)
    assert filtered["tree_id"].tolist() == [1, 2, 3]
    assert filtered["score"].tolist() == [0.90, 0.70, 0.40]
    pd.testing.assert_frame_equal(filtered, filter_detections(raw, 0.30))


def test_prediction_schema_requires_all_deepforest_columns():
    incomplete = _predictions().drop(columns="geometry")

    with pytest.raises(DeepForestDetectionError, match="geometry"):
        validate_prediction_schema(incomplete)


def test_pil_rgb_is_converted_to_independent_float32_rgb_array():
    image = Image.new("RGB", (4, 3), (10, 20, 30))

    result = image_to_rgb_float32(image)

    assert result.shape == (3, 4, 3)
    assert result.dtype == np.float32
    assert result[0, 0].tolist() == [10.0, 20.0, 30.0]


def test_pil_rgba_drops_alpha():
    image = Image.new("RGBA", (2, 2), (1, 2, 3, 4))

    result = image_to_rgb_float32(image)

    assert result.shape == (2, 2, 3)
    assert result[0, 0].tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("image", [np.zeros((4, 4)), np.zeros((4, 4, 2)), np.zeros((4, 4, 5))])
def test_invalid_channel_counts_are_rejected(image):
    with pytest.raises(ImageValidationError):
        image_to_rgb_float32(image)
