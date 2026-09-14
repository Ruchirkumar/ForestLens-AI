"""Deterministic image diagnostics used by the pre-analysis quality gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import cv2
import numpy as np

from src.detection.model import ImageInput, image_to_rgb_float32
from src.models import RasterMetadata

QualityStatus = Literal["GOOD", "WARNING", "BLOCKED"]


@dataclass(frozen=True)
class QualityConfig:
    """Engineering heuristics, deliberately not scientific accuracy limits."""

    warn_resolution_m: float = 1.0
    block_resolution_m: float = 5.0
    sharpness_warn_threshold: float = 20.0
    sharpness_block_threshold: float = 1.0
    contrast_warn_threshold: float = 15.0
    contrast_block_threshold: float = 2.0
    brightness_low_threshold: float = 30.0
    brightness_high_threshold: float = 225.0
    clipping_low_value: float = 5.0
    clipping_high_value: float = 250.0
    clipping_warn_fraction: float = 0.10
    clipping_block_fraction: float = 0.98

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> "QualityConfig":
        if values is None:
            return cls()
        unknown = set(values).difference(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown quality settings: {sorted(unknown)}.")
        try:
            config = cls(**values)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError("Invalid quality settings.") from error
        config.validate()
        return config

    def validate(self) -> None:
        if self.warn_resolution_m <= 0 or self.block_resolution_m < self.warn_resolution_m:
            raise ValueError("Resolution thresholds must be positive and block >= warn.")
        if self.sharpness_warn_threshold < 0 or self.sharpness_block_threshold < 0 or self.sharpness_block_threshold > self.sharpness_warn_threshold:
            raise ValueError("Sharpness thresholds must be non-negative and block <= warn.")
        if self.contrast_warn_threshold < 0 or self.contrast_block_threshold < 0 or self.contrast_block_threshold > self.contrast_warn_threshold:
            raise ValueError("Contrast thresholds must be non-negative and block <= warn.")
        if not 0 <= self.clipping_low_value < self.clipping_high_value <= 255:
            raise ValueError("Clipping values must be within 0..255 and ordered.")
        if not 0 <= self.clipping_warn_fraction <= self.clipping_block_fraction <= 1:
            raise ValueError("Clipping fractions must be ordered values between zero and one.")


@dataclass(frozen=True)
class QualityAssessment:
    """Image diagnostics; values are not a model-quality or accuracy score."""

    status: QualityStatus
    can_analyze: bool
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    blocking_reasons: tuple[str, ...]
    dimensions: tuple[int, int]
    pixel_count: int
    gsd_x_m: float | None
    gsd_y_m: float | None
    sharpness: float
    brightness: float
    contrast: float
    overexposure_fraction: float
    underexposure_fraction: float
    resolution_status: str


def _rgb_to_255(image: ImageInput) -> np.ndarray:
    rgb = image_to_rgb_float32(image)
    maximum = float(np.nanmax(rgb))
    if not np.isfinite(rgb).all():
        raise ValueError("Image contains non-finite pixel values.")
    if maximum <= 1.0:
        rgb = rgb * 255.0
    return np.clip(rgb, 0.0, 255.0)


def _resolution(metadata: RasterMetadata, config: QualityConfig) -> tuple[str, list[str], list[str]]:
    if metadata.gsd_x_m is None or metadata.gsd_y_m is None:
        return "resolution_unknown", ["Physical resolution could not be verified."], []
    gsd = max(metadata.gsd_x_m, metadata.gsd_y_m)
    if gsd >= config.block_resolution_m:
        return "blocked", [], [f"Resolution is {gsd:.2f} m/pixel, beyond the configured engineering gate."]
    if gsd >= config.warn_resolution_m:
        return "warning", [f"Resolution is {gsd:.2f} m/pixel and may limit crown separation."], []
    return "good", [], []


def evaluate_quality(
    image: ImageInput, metadata: RasterMetadata, config: QualityConfig | Mapping[str, object] | None = None
) -> QualityAssessment:
    """Assess basic usability with deterministic, configurable engineering rules."""
    active = config if isinstance(config, QualityConfig) else QualityConfig.from_mapping(config)
    active.validate()
    rgb = _rgb_to_255(image)
    height, width = rgb.shape[:2]
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    # CV_32F works across OpenCV builds that do not support float32-to-float64
    # filtering, while preserving the deterministic variance-of-Laplacian metric.
    sharpness = float(cv2.Laplacian(luminance.astype(np.float32), cv2.CV_32F).var())
    brightness = float(luminance.mean())
    contrast = float(luminance.std())
    overexposure = float(np.mean(luminance >= active.clipping_high_value))
    underexposure = float(np.mean(luminance <= active.clipping_low_value))
    resolution, warnings, errors = _resolution(metadata, active)

    if sharpness < active.sharpness_warn_threshold:
        warnings.append("Image sharpness is low and may reduce detection reliability.")
    if contrast < active.contrast_warn_threshold:
        warnings.append("Low image contrast may affect crown detection.")
    if brightness < active.brightness_low_threshold:
        warnings.append("Image brightness is very low and may affect crown detection.")
    elif brightness > active.brightness_high_threshold:
        warnings.append("Image brightness is very high and may affect crown detection.")
    if max(overexposure, underexposure) >= active.clipping_warn_fraction:
        warnings.append("Strong clipping or exposure may affect crown detection.")

    if sharpness <= active.sharpness_block_threshold and contrast <= active.contrast_block_threshold:
        errors.append("Image has extremely low sharpness and contrast and is unusable for automatic analysis.")
    if max(overexposure, underexposure) >= active.clipping_block_fraction:
        errors.append("Image is almost completely clipped and is unusable for automatic analysis.")
    if errors:
        status: QualityStatus = "BLOCKED"
    elif warnings:
        status = "WARNING"
    else:
        status = "GOOD"
    return QualityAssessment(
        status=status, can_analyze=status != "BLOCKED", warnings=tuple(dict.fromkeys(warnings)),
        errors=tuple(dict.fromkeys(errors)), blocking_reasons=tuple(dict.fromkeys(errors)),
        dimensions=(width, height), pixel_count=width * height, gsd_x_m=metadata.gsd_x_m,
        gsd_y_m=metadata.gsd_y_m, sharpness=sharpness, brightness=brightness, contrast=contrast,
        overexposure_fraction=overexposure, underexposure_fraction=underexposure,
        resolution_status=resolution,
    )
