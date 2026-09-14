"""Run deterministic quality diagnostics and one-inference threshold sensitivity."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

from PIL import Image

from src.config import load_config
from src.detection.model import filter_detections, load_deepforest_model, predict_detections
from src.geometry.crown_refinement import CrownRefinementConfig, refine_crowns
from src.io.raster_reader import read_raster_metadata
from src.quality.assessment import QualityConfig, evaluate_quality
from src.quality.warnings import generate_warnings
from src.uncertainty.sensitivity import calculate_refinement_summary, run_threshold_sensitivity


def _osbs_sample_path() -> Path:
    """Locate the packaged DeepForest example without importing its ML stack."""
    spec = find_spec("deepforest")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("DeepForest package is unavailable.")
    return Path(next(iter(spec.submodule_search_locations))) / "data" / "OSBS_029.png"


def main() -> int:
    try:
        configuration = load_config()
        path = _osbs_sample_path()
        metadata = read_raster_metadata(path)
        with Image.open(path) as source:
            image = source.convert("RGB")
        quality = evaluate_quality(image, metadata, QualityConfig.from_mapping(configuration.get("quality")))
        print("QUALITY")
        print(f"Status: {quality.status}")
        print(f"Can analyze: {quality.can_analyze}")
        print(f"Sharpness: {quality.sharpness:.2f}")
        print(f"Brightness: {quality.brightness:.2f}")
        print(f"Contrast: {quality.contrast:.2f}")
        print(f"Resolution: {quality.resolution_status}")
        print("Warnings:")
        for warning in quality.warnings:
            print(f"- {warning}")

        # Deliberately one inference call. All threshold rows consume raw_predictions.
        model = load_deepforest_model()
        raw_predictions = predict_detections(model, image)
        thresholds = (0.30, 0.40, 0.50, 0.60)
        sensitivity = run_threshold_sensitivity(
            raw_predictions, image, thresholds=thresholds,
            refinement_config=CrownRefinementConfig.from_mapping(configuration.get("refinement")),
        )
        baseline = refine_crowns(image, filter_detections(raw_predictions, thresholds[0]), CrownRefinementConfig.from_mapping(configuration.get("refinement")))
        summary = calculate_refinement_summary(baseline)

        print("\nSENSITIVITY")
        print("Threshold | Detections | Refined | Fallback | Canopy Area")
        for row in sensitivity:
            area = "unavailable" if row.estimated_canopy_area_m2 is None else f"{row.estimated_canopy_area_m2:.2f} m2"
            print(f"{row.threshold:.2f} | {row.detected_tree_count} | {row.refined_count} | {row.fallback_count} | {area}")
        print("\nREFINEMENT")
        print(f"Total: {summary.total_detections}")
        print(f"Refined: {summary.refined}")
        print(f"Fallback: {summary.bbox_fallback}")
        print(f"Failed: {summary.failed}")
        print(f"Refinement rate: {summary.refinement_rate:.3f}" if summary.refinement_rate is not None else "Refinement rate: unavailable")
        print(f"Fallback rate: {summary.fallback_rate:.3f}" if summary.fallback_rate is not None else "Fallback rate: unavailable")
        print("\nRELIABILITY WARNINGS")
        for warning in generate_warnings(quality, summary, missing_georeference=not metadata.georeferenced):
            print(f"{warning.code}: {warning.message}")
        return 0
    except Exception as error:
        print(f"Quality test failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
