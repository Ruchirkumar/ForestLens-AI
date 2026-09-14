"""Backward-compatible image-quality API."""

from .assessment import QualityAssessment, QualityConfig, evaluate_quality


def estimate_image_quality(*args, **kwargs) -> QualityAssessment:
    """Compatibility alias for :func:`evaluate_quality`."""
    return evaluate_quality(*args, **kwargs)
