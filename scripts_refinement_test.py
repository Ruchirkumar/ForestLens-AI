"""Run RGB-derived crown-footprint refinement on DeepForest's sample image.

Examples:
    python scripts_refinement_test.py
    python scripts_refinement_test.py "C:\\path\\to\\forest.tif"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from src.config import get_detection_score_threshold, load_config
from src.detection.model import filter_detections, load_deepforest_model, predict_detections
from src.geometry.crown_refinement import CrownRefinementConfig, refine_crowns
from src.visualization.crown_overlay import render_crown_refinement_overlay

DEFAULT_OUTPUT = Path("artifacts") / "crown_refinement_test.png"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local RGB crown-footprint refinement.")
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
        filtered_predictions = filter_detections(raw_predictions, get_detection_score_threshold())
        refinement_config = CrownRefinementConfig.from_mapping(load_config().get("refinement"))
        results = refine_crowns(image, filtered_predictions, refinement_config)
        render_crown_refinement_overlay(image, results, output_path=DEFAULT_OUTPUT)

        refined = [result for result in results if result.refinement_status == "refined"]
        fallbacks = [result for result in results if result.refinement_status == "fallback_bbox"]
        failed = [result for result in results if result.refinement_status == "failed"]
        mean_confidence = (
            sum(result.refinement_confidence for result in results) / len(results)
            if results
            else 0.0
        )
        total_crown_pixels = sum(result.crown_area_pixels for result in refined)
        total_bbox_proxy_pixels = sum(result.crown_area_pixels for result in fallbacks)

        print(f"Image: {image_path}")
        print(f"Raw detections: {len(raw_predictions)}")
        print(f"Filtered detections: {len(filtered_predictions)}")
        print(f"Refined crowns: {len(refined)}")
        print(f"Bbox fallbacks: {len(fallbacks)}")
        print(f"Failed refinements: {len(failed)}")
        print(f"Mean refinement confidence: {mean_confidence:.3f}")
        print(f"Total crown pixels: {total_crown_pixels:.1f}")
        print(f"Total bbox proxy pixels: {total_bbox_proxy_pixels:.1f}")
        print(f"Output: {DEFAULT_OUTPUT}")
        return 0
    except Exception as error:
        print(f"Refinement test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
