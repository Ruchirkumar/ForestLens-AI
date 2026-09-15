"""Shared data models for raster metadata and legacy pipeline results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from affine import Affine
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RasterMetadata:
    """Validated raster metadata without assuming image-space units are metres."""

    path: str | None
    name: str | None
    width: int
    height: int
    band_count: int
    dtype: str
    crs: str | None
    crs_type: Literal["projected", "geographic", "missing", "unknown"]
    transform: Affine
    bounds: tuple[float, float, float, float]
    pixel_width: float
    pixel_height: float
    pixel_area_native: float
    gsd_x: float
    gsd_y: float
    gsd_x_m: float | None
    gsd_y_m: float | None
    georeferenced: bool
    geographic_crs: bool
    projected_crs: bool
    units: str | None
    linear_unit_to_metre: float | None
    total_pixel_count: int
    physical_area_available: bool
    physical_area_reason: str
    validation_warnings: tuple[str, ...] = ()

    @property
    def bands(self) -> int:
        """Backward-compatible alias for ``band_count``."""
        return self.band_count

    @property
    def is_georeferenced(self) -> bool:
        """Backward-compatible alias for ``georeferenced``."""
        return self.georeferenced


@dataclass
class AnalysisResult:
    """Legacy pipeline result container retained for compatibility."""

    tree_count: int
    canopy_area_m2: float | None = None
    canopy_area_ha: float | None = None
    aoi_area_m2: float | None = None
    canopy_cover_percent: float | None = None
    mean_confidence: float | None = None
    detections: pd.DataFrame = field(default_factory=pd.DataFrame)
    overlay_image: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0
