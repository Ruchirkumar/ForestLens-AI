"""Inspect geospatial metadata without inventing a physical scale.

The built-in DeepForest PNG is always inspected. Any local TIFFs already under
``data/`` are inspected as optional fixtures; this script never downloads data.
"""

from __future__ import annotations

from pathlib import Path

from deepforest import get_data

from src.geometry.area import calculate_raster_area_m2, get_pixel_area_m2
from src.io.raster_reader import read_raster_metadata


def _format_value(value: float | None, suffix: str = "") -> str:
    return "unavailable" if value is None else f"{value:.6f}{suffix}"


def _inspect(path: Path) -> None:
    metadata = read_raster_metadata(path)
    pixel_area = get_pixel_area_m2(metadata)
    raster_area = calculate_raster_area_m2(metadata)
    native_units = metadata.units or "native/unknown units"
    print(f"Raster: {path}")
    print(f"Dimensions: {metadata.width} x {metadata.height}")
    print(f"Bands: {metadata.band_count}")
    print(f"CRS: {metadata.crs or 'missing'}")
    print(f"CRS type: {metadata.crs_type}")
    print(f"Georeferenced: {metadata.georeferenced}")
    print(f"Pixel size: {metadata.pixel_width:.6f} x {metadata.pixel_height:.6f} {native_units}")
    print(f"Pixel area: {_format_value(metadata.pixel_area_native)} {native_units}² (native)")
    if metadata.gsd_x_m is not None and metadata.gsd_y_m is not None:
        print(f"GSD: {metadata.gsd_x_m:.6f} x {metadata.gsd_y_m:.6f} m")
    else:
        print("GSD: unavailable in metres")
    print(f"Physical area available: {metadata.physical_area_available}")
    print(f"Raster footprint area: {_format_value(raster_area, ' m²')}")
    if pixel_area is None:
        print("Physical area: unavailable (valid metric georeferencing required)")
    print()


def _inputs() -> list[Path]:
    paths = [Path(get_data("OSBS_029.png"))]
    data_directory = Path("data")
    if data_directory.is_dir():
        paths.extend(
            sorted(
                path
                for path in data_directory.rglob("*")
                if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
            )
        )
    return list(dict.fromkeys(paths))


def main() -> int:
    try:
        for path in _inputs():
            _inspect(path)
        return 0
    except Exception as error:
        print(f"Geospatial test failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
