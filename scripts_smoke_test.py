"""Run a portable DeepForest detection smoke test.

Examples:
    python scripts_smoke_test.py
    python scripts_smoke_test.py "C:\\path\\to\\forest.tif"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from src.config import get_detection_score_threshold
from src.detection.model import filter_detections, load_deepforest_model, predict_detections


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DeepForest detector smoke test.")
    parser.add_argument(
        "image_path",
        nargs="?",
        type=Path,
        help="Optional RGB image path. Defaults to DeepForest's OSBS_029.png sample.",
    )
    return parser.parse_args()


def _resolve_image_path(image_path: Path | None) -> Path:
    if image_path is not None:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return image_path

    from deepforest import get_data

    return Path(get_data("OSBS_029.png"))


def _print_report(
    image_path: Path,
    image: Image.Image,
    raw_predictions: pd.DataFrame,
    filtered_predictions: pd.DataFrame,
    score_threshold: float,
) -> None:
    scores = filtered_predictions["score"]
    print(f"Image: {image_path}")
    print(f"Image dimensions: {image.width} x {image.height}")
    print(f"Raw detections: {len(raw_predictions)}")
    print(f"Score threshold: {score_threshold:.2f}")
    print(f"Filtered detections: {len(filtered_predictions)}")
    print(f"Mean confidence: {scores.mean():.3f}" if not scores.empty else "Mean confidence: n/a")
    print(f"Min confidence: {scores.min():.3f}" if not scores.empty else "Min confidence: n/a")
    print(f"Max confidence: {scores.max():.3f}" if not scores.empty else "Max confidence: n/a")
    print("First prediction rows:")
    columns = ["tree_id", "xmin", "ymin", "xmax", "ymax", "label", "score"]
    print(filtered_predictions.loc[:, columns].head(5).to_string(index=False))


def main() -> int:
    try:
        args = _arguments()
        image_path = _resolve_image_path(args.image_path)
        with Image.open(image_path) as source_image:
            image = source_image.copy()

        print("Loading DeepForest model...")
        model = load_deepforest_model()
        print("Running prediction...")
        raw_predictions = predict_detections(model, image)
        threshold = get_detection_score_threshold()
        filtered_predictions = filter_detections(raw_predictions, threshold)
        _print_report(image_path, image, raw_predictions, filtered_predictions, threshold)
        return 0
    except Exception as error:
        print(f"Smoke test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
