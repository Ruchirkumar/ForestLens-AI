"""Evidence overlays comparing DeepForest boxes and refined crown footprints."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.detection.model import ImageInput, image_to_rgb_float32
from src.geometry.crown_refinement import CrownRefinementResult

_DETECTION_COLOR = (255, 220, 0)
_REFINED_COLOR = (0, 220, 110)
_FALLBACK_COLOR = (255, 126, 0)
_FAILED_COLOR = (220, 60, 60)
_TEXT_COLOR = (255, 255, 255)
_TEXT_BACKGROUND = (20, 20, 20)


def _display_image(image: ImageInput) -> Image.Image:
    array = image_to_rgb_float32(image)
    if np.nanmax(array) <= 1.0 and np.nanmin(array) >= 0.0:
        array = array * 255.0
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def _clip_point(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    return (
        int(np.clip(round(point[0]), 0, width - 1)),
        int(np.clip(round(point[1]), 0, height - 1)),
    )


def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    result: CrownRefinementResult,
    width: int,
    height: int,
) -> None:
    polygon = result.crown_polygon
    if polygon is None:
        return
    color = _REFINED_COLOR if result.refinement_status == "refined" else _FALLBACK_COLOR
    geometries = getattr(polygon, "geoms", [polygon])
    for geometry in geometries:
        exterior = getattr(geometry, "exterior", None)
        if exterior is None:
            continue
        points = [_clip_point(point, width, height) for point in exterior.coords]
        if len(points) >= 2:
            draw.line(points, fill=color, width=2, joint="curve")


def _draw_legend(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, offset_x: int) -> int:
    rows = [
        ("DeepForest bbox", _DETECTION_COLOR),
        ("Refined RGB footprint", _REFINED_COLOR),
        ("Bbox proxy fallback", _FALLBACK_COLOR),
    ]
    y = 4
    for text, color in rows:
        text_box = draw.textbbox((0, 0), text, font=font)
        text_width = text_box[2] - text_box[0]
        draw.rectangle((offset_x + 4, y, offset_x + text_width + 28, y + 12), fill=_TEXT_BACKGROUND)
        draw.line((offset_x + 7, y + 6, offset_x + 20, y + 6), fill=color, width=2)
        draw.text((offset_x + 23, y + 1), text, fill=_TEXT_COLOR, font=font)
        y += 13
    return y + 4


def render_crown_refinement_overlay(
    image: ImageInput,
    results: Sequence[CrownRefinementResult],
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Render DeepForest boxes and derived crown boundaries for review.

    Yellow marks every detector box. Green marks a local RGB-derived footprint;
    orange marks a fallback bounding-box proxy. This is a visual diagnostic, not
    a ground-truth segmentation display.
    """
    imagery = _display_image(image)
    image_width, image_height = imagery.size
    font = ImageFont.load_default()
    panel_width = 210
    row_height = 12
    canvas_height = max(image_height, 48 + len(results) * row_height)
    rendered = Image.new("RGB", (image_width + panel_width, canvas_height), _TEXT_BACKGROUND)
    rendered.paste(imagery, (0, 0))
    draw = ImageDraw.Draw(rendered)
    width, height = image_width, image_height

    for result in results:
        left, top, right, bottom = result.bbox
        bbox_points = [
            _clip_point((left, top), width, height),
            _clip_point((right, bottom), width, height),
        ]
        draw.rectangle(bbox_points, outline=_DETECTION_COLOR, width=1)
        _draw_polygon(draw, result, width, height)

        # A compact ID keeps imagery review legible; full score/status details
        # appear in the side panel below instead of covering dense crowns.
        draw.text(
            (bbox_points[0][0] + 1, bbox_points[0][1] + 1),
            f"T{result.tree_id:02d}",
            fill=_TEXT_COLOR,
            font=font,
            stroke_width=1,
            stroke_fill=_TEXT_BACKGROUND,
        )

    panel_x = image_width
    row_y = _draw_legend(draw, font, panel_x)
    for result in results:
        status = "refined" if result.refinement_status == "refined" else "bbox proxy"
        color = _REFINED_COLOR if result.refinement_status == "refined" else _FALLBACK_COLOR
        if result.refinement_status == "failed":
            status, color = "failed", _FAILED_COLOR
        draw.line((panel_x + 7, row_y + 6, panel_x + 18, row_y + 6), fill=color, width=2)
        draw.text(
            (panel_x + 22, row_y + 1),
            f"T{result.tree_id:02d} | {result.detection_score:.2f} | {status}",
            fill=_TEXT_COLOR,
            font=font,
        )
        row_y += row_height
    result_array = np.asarray(rendered).copy()
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(destination)
    return result_array
