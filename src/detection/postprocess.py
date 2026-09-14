"""Compatibility exports for detection post-processing."""

from .model import REQUIRED_PREDICTION_COLUMNS, filter_detections, validate_prediction_schema


def validate_predictions(predictions):
    """Backward-compatible name for prediction schema validation."""
    return validate_prediction_schema(predictions)


__all__ = [
    "REQUIRED_PREDICTION_COLUMNS",
    "filter_detections",
    "validate_prediction_schema",
    "validate_predictions",
]
