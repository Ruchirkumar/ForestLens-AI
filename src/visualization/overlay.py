"""Deterministic, model-evidence visualizations for DeepForest detections."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.detection.model import ImageInput, image_to_rgb_float32

_BOX_COLOR = (255, 64, 64)
_TEXT_COLOR = (255, 255, 255)
_TEXT_BACKGROUND = (24, 24, 24)


def _display_image(image: ImageInput) -> Image.Image:
    """Create a drawable RGB image without modifying the caller's image."""
    array = image_to_rgb_float32(image)
    if np.nanmax(array) <= 1.0 and np.nanmin(array) >= 0.0:
        array = array * 255.0
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def _label_for_detection(
    row: pd.Series,
    position: int,
    show_ids: bool,
    show_scores: bool,
) -> str:
    parts: list[str] = []
    if show_ids:
        tree_id = row.get("tree_id", position + 1)
        parts.append(f"T{int(tree_id):02d}")
    if show_scores:
        parts.append(f"{float(row['score']):.2f}")
    return " | ".join(parts)


def render_detection_overlay(
    image: ImageInput,
    predictions: pd.DataFrame,
    output_path: str | Path | None = None,
    *,
    show_ids: bool = True,
    show_scores: bool = True,
) -> np.ndarray:
    """Draw filtered DeepForest bounding-box detections on an RGB image.

    ``geometry`` is intentionally not used: DeepForest's current returned
    geometry corresponds to the detection box, not an exact crown segmentation.
    The input image and DataFrame remain unmodified.
    """
    required_columns = {"xmin", "ymin", "xmax", "ymax", "score"}
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame.")
    missing = required_columns.difference(predictions.columns)
    if missing:
        raise ValueError(
            f"Predictions are missing columns required for rendering: {sorted(missing)}."
        )

    rendered = _display_image(image)
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    width, height = rendered.size

    for position, (_, row) in enumerate(predictions.iterrows()):
        left = int(np.clip(np.floor(float(row["xmin"])), 0, width - 1))
        top = int(np.clip(np.floor(float(row["ymin"])), 0, height - 1))
        right = int(np.clip(np.ceil(float(row["xmax"])), 0, width - 1))
        bottom = int(np.clip(np.ceil(float(row["ymax"])), 0, height - 1))
        left, right = sorted((left, right))
        top, bottom = sorted((top, bottom))
        draw.rectangle((left, top, right, bottom), outline=_BOX_COLOR, width=2)

        label = _label_for_detection(row, position, show_ids, show_scores)
        if label:
            text_box = draw.textbbox((0, 0), label, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            label_x = min(left, max(0, width - text_width - 2))
            label_y = top - text_height - 4 if top >= text_height + 4 else top + 2
            draw.rectangle(
                (label_x, label_y, label_x + text_width + 2, label_y + text_height + 2),
                fill=_TEXT_BACKGROUND,
            )
            draw.text((label_x + 1, label_y + 1), label, fill=_TEXT_COLOR, font=font)

    result = np.asarray(rendered).copy()
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(destination)
    return result


def create_detection_overlay(*args, **kwargs) -> np.ndarray:
    """Backward-compatible alias for :func:`render_detection_overlay`."""
    return render_detection_overlay(*args, **kwargs)
