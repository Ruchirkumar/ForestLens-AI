from dataclasses import dataclass, field
from typing import Literal, Optional

from affine import Affine

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RasterMetadata:
    """Validated raster metadata without assuming that image-space units are metres."""

    path: Optional[str]
    name: Optional[str]
    width: int
    height: int
    band_count: int
    dtype: str
    crs: Optional[str]
    crs_type: Literal["projected", "geographic", "missing", "unknown"]
    transform: Affine
    bounds: tuple[float, float, float, float]
    pixel_width: float
    pixel_height: float
    pixel_area_native: float
    gsd_x: float
    gsd_y: float
    gsd_x_m: Optional[float]
    gsd_y_m: Optional[float]
    georeferenced: bool
    geographic_crs: bool
    projected_crs: bool
    units: Optional[str]
    linear_unit_to_metre: Optional[float]
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
    tree_count: int
    canopy_area_m2: Optional[float] = None
    canopy_area_ha: Optional[float] = None
    aoi_area_m2: Optional[float] = None
    canopy_cover_percent: Optional[float] = None
    mean_confidence: Optional[float] = None
    detections: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    overlay_image: Optional[np.ndarray] = None
    warnings: list[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0
