from src.geometry.area import m2_to_hectares, pixel_area_m2


def test_pixel_area():
    assert pixel_area_m2(100, 0.5, 0.5) == 25.0


def test_hectares():
    assert m2_to_hectares(10_000) == 1.0
