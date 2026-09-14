"""Structured reliability warnings without claims about model accuracy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .assessment import QualityAssessment

if TYPE_CHECKING:
    from src.uncertainty.sensitivity import RefinementSummary


@dataclass(frozen=True)
class QualityWarning:
    code: str
    message: str


def generate_warnings(
    assessment: QualityAssessment | None = None,
    refinement: "RefinementSummary | None" = None,
    *,
    missing_georeference: bool = False,
) -> tuple[QualityWarning, ...]:
    """Generate transparent warning codes from known diagnostic conditions."""
    warnings: list[QualityWarning] = [QualityWarning("DOMAIN_WARNING", "Pretrained tree detector performance may vary with imagery resolution, forest type, acquisition conditions, and domain.")]
    if missing_georeference:
        warnings.append(QualityWarning("MISSING_GEOREFERENCE", "Physical area cannot be calculated because the raster lacks valid georeferencing."))
    if assessment is not None:
        if assessment.resolution_status == "resolution_unknown":
            warnings.append(QualityWarning("UNKNOWN_RESOLUTION", "Input resolution/GSD could not be verified."))
        diagnostic_text = " ".join((*assessment.warnings, *assessment.errors)).lower()
        if "sharpness" in diagnostic_text and assessment.status != "BLOCKED":
            warnings.append(QualityWarning("POOR_SHARPNESS", "Image sharpness is low and may reduce detection reliability."))
        if "contrast" in diagnostic_text:
            warnings.append(QualityWarning("LOW_CONTRAST", "Low image contrast may affect crown detection."))
        if "clipping" in diagnostic_text or "exposure" in diagnostic_text:
            warnings.append(QualityWarning("EXCESSIVE_EXPOSURE", "Strong clipping/exposure may affect crown detection."))
    if refinement is not None and refinement.total_detections:
        if refinement.fallback_rate >= 0.5:
            warnings.append(QualityWarning("HIGH_FALLBACK_RATE", "Most crown footprints use bbox fallback; canopy area is therefore a proxy estimate."))
        if refinement.refinement_rate < 0.25:
            warnings.append(QualityWarning("LOW_REFINEMENT_RATE", "Few detections produced image-derived crown footprints."))
    return tuple(warnings)


def build_warnings(*args, **kwargs) -> list[str]:
    """Legacy flat-message form of :func:`generate_warnings`."""
    return [warning.message for warning in generate_warnings(*args, **kwargs)]
