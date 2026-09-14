"""Generate a DeepForest evidence overlay using the built-in sample image.

Examples:
    python scripts_visualize_test.py
    python scripts_visualize_test.py "C:\\path\\to\\forest.tif"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from src.config import get_detection_score_threshold
from src.detection.model import filter_detections, load_deepforest_model, predict_detections
from src.visualization.overlay import render_detection_overlay

DEFAULT_OUTPUT = Path("artifacts") / "deepforest_test.png"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render DeepForest detection evidence.")
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


def main() -> int:
    try:
        args = _arguments()
        image_path = _resolve_image_path(args.image_path)
        with Image.open(image_path) as source_image:
            image = source_image.copy()

        model = load_deepforest_model()
        raw_predictions = predict_detections(model, image)
        threshold = get_detection_score_threshold()
        filtered_predictions = filter_detections(raw_predictions, threshold)
        render_detection_overlay(image, filtered_predictions, output_path=DEFAULT_OUTPUT)

        print(f"Input image: {image_path}")
        print(f"Raw detections: {len(raw_predictions)}")
        print(f"Filtered detections: {len(filtered_predictions)}")
        print(f"Output: {DEFAULT_OUTPUT}")
        return 0
    except Exception as error:
        print(f"Visualization test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
