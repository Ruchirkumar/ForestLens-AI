import numpy as np

from src.geometry.crown_refinement import refine_crown
from src.visualization.crown_overlay import render_crown_refinement_overlay


def test_crown_overlay_renders_refined_and_fallback_results_without_mutating_image():
    image = np.full((50, 50, 3), (80, 80, 80), dtype=np.uint8)
    image[18:32, 18:32] = (20, 180, 20)
    original = image.copy()
    refined = refine_crown(image, 1, 0.8, (12, 12, 38, 38))
    fallback = refine_crown(image, 2, 0.7, (0, 0, 8, 8))

    rendered = render_crown_refinement_overlay(image, [refined, fallback])

    assert rendered.shape[0] >= image.shape[0]
    assert rendered.shape[1] > image.shape[1]
    assert not np.array_equal(rendered, image)
    assert np.array_equal(image, original)
