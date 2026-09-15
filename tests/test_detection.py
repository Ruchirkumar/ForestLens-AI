import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.detection.model import (
    DIRECT_MAX_DIMENSION,
    DeepForestDetectionError,
    ImageValidationError,
    TILE_IOU_THRESHOLD,
    TILE_OVERLAP,
    TILE_PATCH_SIZE,
    filter_detections,
    image_to_rgb_float32,
    predict_detections,
    prediction_mode,
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


class _DirectModel:
    def __init__(self):
        self.image = None

    def predict_image(self, *, image):
        self.image = image
        return _predictions()


class _TiledModel:
    def __init__(self):
        self.kwargs = None

    def predict_tile(self, **kwargs):
        self.kwargs = kwargs
        return _predictions()


def test_small_images_use_deepforest_direct_array_api():
    model = _DirectModel()
    image = np.zeros((DIRECT_MAX_DIMENSION, 8, 3), dtype=np.uint8)

    result = predict_detections(model, image)

    assert prediction_mode(image) == "direct"
    assert model.image is not None and model.image.dtype == np.float32
    pd.testing.assert_frame_equal(result, _predictions())


def test_large_images_use_verified_tiled_api_and_bgr_input():
    model = _TiledModel()
    image = np.zeros((DIRECT_MAX_DIMENSION + 1, 8, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]

    predict_detections(model, image)

    assert prediction_mode(image) == "tiled"
    assert model.kwargs is not None
    assert model.kwargs["patch_size"] == TILE_PATCH_SIZE
    assert model.kwargs["patch_overlap"] == TILE_OVERLAP
    assert model.kwargs["iou_threshold"] == TILE_IOU_THRESHOLD
    assert model.kwargs["dataloader_strategy"] == "single"
    assert model.kwargs["image"][0, 0].tolist() == [30.0, 20.0, 10.0]
