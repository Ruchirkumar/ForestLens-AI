"""DeepForest model loading, image preparation, and detection filtering.

DeepForest 2.1's ``predict_image`` API accepts an image array. This module
therefore deliberately does not accept filesystem paths for prediction; callers
must open a file before invoking the detector.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from PIL import Image

MODEL_NAME = "weecology/deepforest-tree"
MODEL_REVISION = "main"
# DeepForest 2.1.0's ``predict_tile`` mosaics predictions from overlapping
# windows. A 15% overlap creates a 120-pixel boundary margin for 800-pixel
# tiles. Its documented/default mosaic IoU of 0.15 suppresses duplicate
# overlap predictions without indiscriminately merging nearby crown candidates.
TILE_PATCH_SIZE = 800
TILE_OVERLAP = 0.15
TILE_IOU_THRESHOLD = 0.15
DIRECT_MAX_DIMENSION = 1400
REQUIRED_PREDICTION_COLUMNS = frozenset(
    {"xmin", "ymin", "xmax", "ymax", "label", "score", "geometry"}
)

ImageInput: TypeAlias = Image.Image | np.ndarray


class DeepForestDetectionError(RuntimeError):
    """Raised when DeepForest cannot be loaded or returns an invalid result."""


class ImageValidationError(ValueError):
    """Raised when an image cannot safely be interpreted as RGB."""


def load_deepforest_model(
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
) -> Any:
    """Load the pretrained DeepForest detector.

    The import is intentionally lazy so lightweight functions and unit tests do
    not require DeepForest to be imported. Streamlit can cache this normal
    Python function with ``st.cache_resource`` when an inference UI is added.
    """
    try:
        from deepforest import main
    except ImportError as error:
        raise DeepForestDetectionError(
            "DeepForest is not available in this Python environment. "
            "Activate the project's existing environment before running inference."
        ) from error

    try:
        model = main.deepforest()
        model.load_model(model_name=model_name, revision=revision)
    except Exception as error:
        raise DeepForestDetectionError(
            f"Unable to load DeepForest model '{model_name}' (revision '{revision}')."
        ) from error
    return model


def load_model() -> Any:
    """Backward-compatible alias for :func:`load_deepforest_model`."""
    return load_deepforest_model()


def image_to_rgb_float32(image: ImageInput) -> np.ndarray:
    """Return an independent ``H x W x 3`` RGB float32 image array.

    Only RGB and RGBA images are valid inputs. Alpha is dropped from RGBA;
    other channel counts are rejected rather than being guessed as RGB or a
    multispectral band order.
    """
    if isinstance(image, Image.Image):
        if image.mode not in {"RGB", "RGBA"}:
            raise ImageValidationError(
                f"Expected a PIL RGB or RGBA image, received mode '{image.mode}'."
            )
        array = np.asarray(image)
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise TypeError(
            "image must be a PIL.Image.Image or numpy.ndarray; file paths must be read first."
        )

    if array.ndim != 3:
        raise ImageValidationError(
            f"Expected an H x W x 3 or H x W x 4 image, received shape {array.shape}."
        )
    if array.shape[2] not in {3, 4}:
        raise ImageValidationError(
            "Expected 3 RGB channels or 4 RGBA channels, "
            f"received {array.shape[2]} channels."
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ImageValidationError("Image height and width must both be greater than zero.")
    if not np.issubdtype(array.dtype, np.number):
        raise ImageValidationError(f"Image dtype must be numeric, received {array.dtype}.")

    # astype(copy=True) protects the caller's image and removes an RGBA alpha band.
    return array[:, :, :3].astype(np.float32, copy=True)


def validate_prediction_schema(predictions: pd.DataFrame) -> pd.DataFrame:
    """Validate DeepForest's detection schema and return a defensive copy."""
    if not isinstance(predictions, pd.DataFrame):
        raise DeepForestDetectionError(
            "DeepForest prediction output must be a pandas DataFrame."
        )

    missing = REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns)
    if missing:
        raise DeepForestDetectionError(
            "DeepForest prediction output is missing required columns: "
            f"{sorted(missing)}."
        )
    return predictions.copy()


def _empty_predictions() -> pd.DataFrame:
    """Return an empty frame with the public DeepForest prediction schema."""
    return pd.DataFrame(columns=sorted(REQUIRED_PREDICTION_COLUMNS))


def prediction_mode(image: ImageInput) -> str:
    """Choose direct inference or tiled inference from validated dimensions."""
    rgb = image_to_rgb_float32(image)
    return "direct" if max(rgb.shape[:2]) <= DIRECT_MAX_DIMENSION else "tiled"


def predict_detections(model: Any, image: ImageInput) -> pd.DataFrame:
    """Run adaptive DeepForest prediction on a PIL or NumPy RGB/RGBA image.

    The raw result is returned without threshold filtering so downstream
    sensitivity analysis can compare thresholds against identical predictions.
    Images above ``DIRECT_MAX_DIMENSION`` use DeepForest 2.1.0's supported
    ``predict_tile`` API. The tiled array route expects BGR (OpenCV) ordering,
    while this application otherwise keeps imagery in canonical RGB order.
    """
    image_array = image_to_rgb_float32(image)
    try:
        if max(image_array.shape[:2]) <= DIRECT_MAX_DIMENSION:
            predictions = model.predict_image(image=image_array)
        else:
            predictions = model.predict_tile(
                image=image_array[:, :, ::-1].copy(),
                patch_size=TILE_PATCH_SIZE,
                patch_overlap=TILE_OVERLAP,
                iou_threshold=TILE_IOU_THRESHOLD,
                dataloader_strategy="single",
            )
    except Exception as error:
        raise DeepForestDetectionError("DeepForest prediction failed during adaptive inference.") from error

    if predictions is None:
        return _empty_predictions()

    return validate_prediction_schema(predictions)


def filter_detections(
    predictions: pd.DataFrame,
    score_threshold: float,
) -> pd.DataFrame:
    """Return deterministic, threshold-filtered detections with stable tree IDs.

    ``predictions`` is never modified. Higher confidence detections sort first,
    followed by bounding-box coordinates and label; ``tree_id`` starts at one
    after filtering.
    """
    if isinstance(score_threshold, bool) or not isinstance(
        score_threshold, (int, float, np.number)
    ):
        raise ValueError("score_threshold must be a number between 0 and 1.")
    threshold = float(score_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("score_threshold must be between 0 and 1 inclusive.")

    validated = validate_prediction_schema(predictions)
    filtered = validated.loc[validated["score"] >= threshold].copy()
    filtered["_source_order"] = np.arange(len(filtered))
    filtered = filtered.sort_values(
        by=["score", "xmin", "ymin", "xmax", "ymax", "label", "_source_order"],
        ascending=[False, True, True, True, True, True, True],
        kind="mergesort",
    ).drop(columns="_source_order")
    filtered = filtered.reset_index(drop=True)
    filtered.insert(0, "tree_id", np.arange(1, len(filtered) + 1, dtype=int))
    return filtered
