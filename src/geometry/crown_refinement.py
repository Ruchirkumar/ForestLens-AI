"""Conservative RGB-derived crown-footprint refinement.

DeepForest detections are object-detection boxes, not crown segmentations. This
module uses local RGB cues to derive a provisional footprint inside each box's
neighbourhood. A failed or implausible refinement remains explicitly marked as
a bounding-box proxy rather than being presented as a segmented crown.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, pi, sqrt
from typing import Literal, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from src.detection.model import ImageInput, image_to_rgb_float32

RefinementStatus = Literal["refined", "fallback_bbox", "failed"]
AreaSource = Literal["refined_mask", "bbox_proxy", "none"]


@dataclass(frozen=True)
class CrownRefinementConfig:
    """Tunable, image-space controls for conservative local refinement."""

    bbox_padding_fraction: float = 0.10
    min_component_pixels: int = 12
    min_mask_fraction: float = 0.01
    max_patch_fraction: float = 0.85
    min_refinement_confidence: float = 0.25
    min_vegetation_score: float = 0.35
    center_search_radius_fraction: float = 0.45
    morphology_kernel_size: int = 3

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> "CrownRefinementConfig":
        """Build a config from the optional ``refinement`` YAML mapping."""
        if values is None:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values).difference(allowed)
        if unknown:
            raise ValueError(f"Unknown crown refinement settings: {sorted(unknown)}.")
        try:
            config = cls(**values)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError("Invalid crown refinement configuration.") from error
        config.validate()
        return config

    def validate(self) -> None:
        """Validate settings before any image processing begins."""
        fractions = {
            "bbox_padding_fraction": self.bbox_padding_fraction,
            "min_mask_fraction": self.min_mask_fraction,
            "max_patch_fraction": self.max_patch_fraction,
            "min_refinement_confidence": self.min_refinement_confidence,
            "min_vegetation_score": self.min_vegetation_score,
            "center_search_radius_fraction": self.center_search_radius_fraction,
        }
        if any(not 0.0 <= float(value) <= 1.0 for value in fractions.values()):
            raise ValueError("Crown refinement fractions and scores must be between 0 and 1.")
        if self.min_component_pixels < 1:
            raise ValueError("min_component_pixels must be at least one.")
        if self.morphology_kernel_size < 1 or self.morphology_kernel_size % 2 == 0:
            raise ValueError("morphology_kernel_size must be a positive odd integer.")


@dataclass
class CrownRefinementResult:
    """One image-derived footprint or an explicitly marked bounding-box proxy.

    ``mask`` is a local boolean mask whose upper-left image coordinate is
    ``mask_origin``. ``crown_area_pixels`` is its selected-pixel count only for
    ``refined_mask`` results. Bbox proxies retain their continuous bounding-box
    area in image pixel units. ``refinement_confidence`` is an algorithmic
    quality indicator, not a calibrated probability or accuracy estimate.
    """

    tree_id: int
    detection_score: float
    bbox: tuple[float, float, float, float]
    refinement_status: RefinementStatus
    refinement_confidence: float
    mask: np.ndarray
    mask_origin: tuple[int, int]
    crown_area_pixels: float
    crown_polygon: BaseGeometry | None
    area_source: AreaSource


def _clip_bbox(
    bbox: tuple[float, float, float, float], image_width: int, image_height: int
) -> tuple[float, float, float, float] | None:
    xmin, ymin, xmax, ymax = bbox
    if not np.isfinite([xmin, ymin, xmax, ymax]).all():
        return None
    left = float(np.clip(xmin, 0, image_width))
    top = float(np.clip(ymin, 0, image_height))
    right = float(np.clip(xmax, 0, image_width))
    bottom = float(np.clip(ymax, 0, image_height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _expanded_patch_bounds(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    padding_fraction: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    padding_x = (right - left) * padding_fraction
    padding_y = (bottom - top) * padding_fraction
    return (
        max(0, int(floor(left - padding_x))),
        max(0, int(floor(top - padding_y))),
        min(image_width, int(ceil(right + padding_x))),
        min(image_height, int(ceil(bottom + padding_y))),
    )


def _rgb_to_uint8(image: np.ndarray) -> np.ndarray:
    clipped = np.clip(image, 0.0, None)
    if float(np.nanmax(clipped)) <= 1.0:
        clipped = clipped * 255.0
    return np.clip(clipped, 0, 255).astype(np.uint8)


def _vegetation_mask(patch: np.ndarray, config: CrownRefinementConfig) -> np.ndarray:
    """Build a local RGB vegetation candidate mask without fixed green cutoffs."""
    rgb = _rgb_to_uint8(patch).astype(np.float32)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    channel_sum = red + green + blue + 1e-6
    normalized_exg = (2.0 * green - red - blue) / channel_sum
    normalized_green = green / channel_sum
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0

    # The RGB indices retain meaningful relative colour information, while Otsu
    # chooses a threshold locally for each detector neighbourhood.
    exg_score = np.clip((normalized_exg + 0.5) / 2.0, 0.0, 1.0)
    green_score = np.clip((normalized_green - 0.20) / 0.50, 0.0, 1.0)
    score = 0.65 * exg_score + 0.25 * green_score + 0.10 * saturation
    score_uint8 = np.round(score * 255.0).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(
        score_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = max(config.min_vegetation_score, float(otsu_threshold) / 255.0)
    candidates = (score >= threshold).astype(np.uint8)

    kernel = np.ones(
        (config.morphology_kernel_size, config.morphology_kernel_size), dtype=np.uint8
    )
    cleaned = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned.astype(bool)


def _select_center_component(
    candidates: np.ndarray,
    center_x: float,
    center_y: float,
    bbox_width: float,
    bbox_height: float,
    config: CrownRefinementConfig,
) -> tuple[np.ndarray | None, bool]:
    """Select the connected vegetation component associated with the box centre."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidates.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return None, False

    height, width = candidates.shape
    center_col = int(np.clip(round(center_x), 0, width - 1))
    center_row = int(np.clip(round(center_y), 0, height - 1))
    direct_label = int(labels[center_row, center_col])
    if direct_label > 0:
        return labels == direct_label, True

    radius = max(2.0, sqrt(bbox_width**2 + bbox_height**2) * config.center_search_radius_fraction)
    choices: list[tuple[float, int, int]] = []
    for label in range(1, count):
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        component_x, component_y = centroids[label]
        distance = float(np.hypot(component_x - center_x, component_y - center_y))
        if distance <= radius:
            choices.append((distance, -component_area, label))
    if not choices:
        return None, False
    selected_label = min(choices)[2]
    return labels == selected_label, False


def _polygon_from_mask(mask: np.ndarray, origin: tuple[int, int]) -> BaseGeometry | None:
    """Create a valid original-image-coordinate polygon from a local mask."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    points = [(float(point[0][0] + origin[0]), float(point[0][1] + origin[1])) for point in contour]
    polygon: BaseGeometry = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or not polygon.is_valid or polygon.geom_type not in {"Polygon", "MultiPolygon"}:
        return None
    return polygon


def _refinement_confidence(mask: np.ndarray, center_inside: bool) -> float:
    """Return an algorithmic mask-quality indicator in [0, 1]."""
    area = float(mask.sum())
    if area == 0:
        return 0.0
    occupancy = area / float(mask.size)
    occupancy_score = max(0.0, 1.0 - abs(occupancy - 0.35) / 0.65)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    compactness = 0.0 if perimeter <= 0 else min(1.0, 4.0 * pi * area / (perimeter**2))
    edge_count = sum((mask[0, :].any(), mask[-1, :].any(), mask[:, 0].any(), mask[:, -1].any()))
    boundary_score = 1.0 - edge_count / 4.0
    center_score = 1.0 if center_inside else 0.6
    return float(np.clip(
        0.40 * center_score + 0.25 * occupancy_score + 0.25 * compactness + 0.10 * boundary_score,
        0.0,
        1.0,
    ))


def _fallback_result(
    tree_id: int,
    detection_score: float,
    bbox: tuple[float, float, float, float],
) -> CrownRefinementResult:
    left, top, right, bottom = bbox
    width = max(1, int(ceil(right) - floor(left)))
    height = max(1, int(ceil(bottom) - floor(top)))
    return CrownRefinementResult(
        tree_id=tree_id,
        detection_score=detection_score,
        bbox=bbox,
        refinement_status="fallback_bbox",
        refinement_confidence=0.0,
        mask=np.ones((height, width), dtype=bool),
        mask_origin=(int(floor(left)), int(floor(top))),
        crown_area_pixels=float((right - left) * (bottom - top)),
        crown_polygon=box(left, top, right, bottom),
        area_source="bbox_proxy",
    )


def _failed_result(tree_id: int, detection_score: float) -> CrownRefinementResult:
    return CrownRefinementResult(
        tree_id=tree_id,
        detection_score=detection_score,
        bbox=(0.0, 0.0, 0.0, 0.0),
        refinement_status="failed",
        refinement_confidence=0.0,
        mask=np.zeros((0, 0), dtype=bool),
        mask_origin=(0, 0),
        crown_area_pixels=0.0,
        crown_polygon=None,
        area_source="none",
    )


def refine_crown(
    image: ImageInput,
    tree_id: int,
    detection_score: float,
    bbox: tuple[float, float, float, float],
    config: CrownRefinementConfig | None = None,
) -> CrownRefinementResult:
    """Refine one detection locally, or return an explicit bbox-proxy fallback."""
    active_config = config or CrownRefinementConfig()
    active_config.validate()
    rgb = image_to_rgb_float32(image)
    image_height, image_width = rgb.shape[:2]
    clipped_bbox = _clip_bbox(bbox, image_width, image_height)
    if clipped_bbox is None:
        return _failed_result(tree_id, float(detection_score))

    patch_left, patch_top, patch_right, patch_bottom = _expanded_patch_bounds(
        clipped_bbox, image_width, image_height, active_config.bbox_padding_fraction
    )
    patch = rgb[patch_top:patch_bottom, patch_left:patch_right]
    candidates = _vegetation_mask(patch, active_config)
    left, top, right, bottom = clipped_bbox
    center_x = (left + right) / 2.0 - patch_left
    center_y = (top + bottom) / 2.0 - patch_top
    selected_mask, center_inside = _select_center_component(
        candidates,
        center_x,
        center_y,
        right - left,
        bottom - top,
        active_config,
    )
    if selected_mask is None:
        return _fallback_result(tree_id, float(detection_score), clipped_bbox)

    selected_pixels = int(selected_mask.sum())
    min_pixels = max(
        active_config.min_component_pixels,
        int(ceil((right - left) * (bottom - top) * active_config.min_mask_fraction)),
    )
    if selected_pixels < min_pixels or selected_pixels / selected_mask.size >= active_config.max_patch_fraction:
        return _fallback_result(tree_id, float(detection_score), clipped_bbox)

    confidence = _refinement_confidence(selected_mask, center_inside)
    polygon = _polygon_from_mask(selected_mask, (patch_left, patch_top))
    if polygon is None or confidence < active_config.min_refinement_confidence:
        return _fallback_result(tree_id, float(detection_score), clipped_bbox)

    return CrownRefinementResult(
        tree_id=tree_id,
        detection_score=float(detection_score),
        bbox=clipped_bbox,
        refinement_status="refined",
        refinement_confidence=confidence,
        mask=selected_mask.copy(),
        mask_origin=(patch_left, patch_top),
        crown_area_pixels=float(selected_pixels),
        crown_polygon=polygon,
        area_source="refined_mask",
    )


def refine_crowns(
    image: ImageInput,
    predictions: pd.DataFrame,
    config: CrownRefinementConfig | None = None,
) -> list[CrownRefinementResult]:
    """Refine filtered detections without mutating their DataFrame or image.

    The function expects the ``tree_id`` assigned by ``filter_detections`` and
    returns one result per row, including explicit fallbacks and failures.
    """
    required = {"tree_id", "score", "xmin", "ymin", "xmax", "ymax"}
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame.")
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing refinement columns: {sorted(missing)}.")
    active_config = config or CrownRefinementConfig()
    active_config.validate()
    rgb = image_to_rgb_float32(image)
    results: list[CrownRefinementResult] = []
    for _, row in predictions.iterrows():
        results.append(
            refine_crown(
                rgb,
                tree_id=int(row["tree_id"]),
                detection_score=float(row["score"]),
                bbox=(
                    float(row["xmin"]),
                    float(row["ymin"]),
                    float(row["xmax"]),
                    float(row["ymax"]),
                ),
                config=active_config,
            )
        )
    return results


def polygon_iou(first: BaseGeometry | None, second: BaseGeometry | None) -> float:
    """Return pairwise image-coordinate polygon IoU without removing overlaps."""
    if first is None or second is None or first.is_empty or second.is_empty:
        return 0.0
    union_area = first.union(second).area
    return 0.0 if union_area <= 0 else float(first.intersection(second).area / union_area)
