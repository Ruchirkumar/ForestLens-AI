def should_tile(width: int, height: int, threshold_pixels: int = 4_000_000) -> bool:
    return width * height > threshold_pixels
