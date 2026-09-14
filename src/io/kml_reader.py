"""KML Area-of-Interest parsing with explicit WGS84 semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from fastkml import kml
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

KML_SOURCE_CRS = "EPSG:4326"
_SUPPORTED_TYPES = frozenset({"Polygon", "MultiPolygon"})


@dataclass(frozen=True)
class AOIResult:
    """The polygonal content of a KML document.

    KML coordinates are defined as WGS84 longitude and latitude (EPSG:4326).
    This reader never infers a different source CRS.
    """

    geometry: BaseGeometry | None
    geometry_type: str | None
    source_crs: str
    valid: bool
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    feature_count: int
    source_name: str | None
    original_geometry: BaseGeometry | None = None


def _children(feature: Any) -> Iterable[Any]:
    """Return FastKML child features across supported FastKML versions."""
    children = getattr(feature, "features", ())
    return children() if callable(children) else children


def _to_shapely(kml_geometry: Any) -> BaseGeometry | None:
    """Convert a FastKML/pygeoif geometry without private FastKML APIs."""
    if kml_geometry is None:
        return None
    candidate = getattr(kml_geometry, "geometry", kml_geometry)
    interface = getattr(candidate, "__geo_interface__", None)
    return shape(interface) if interface is not None else None


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    """Extract usable polygon components, including polygonal collections."""
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        return [part for member in geometry.geoms for part in _polygon_parts(member)]
    return []


def _validate_geometry(geometry: BaseGeometry) -> tuple[BaseGeometry | None, list[str], list[str]]:
    """Validate a polygonal AOI and make only the standard conservative repair."""
    warnings: list[str] = []
    if geometry.is_empty:
        return None, warnings, ["KML polygon geometry is empty."]
    candidate = geometry
    if not candidate.is_valid:
        try:
            repaired = candidate.buffer(0)
        except (ValueError, TypeError):
            return None, warnings, ["KML polygon geometry is invalid and could not be safely repaired."]
        if repaired.is_empty or not repaired.is_valid:
            return None, warnings, ["KML polygon geometry is invalid and could not be safely repaired."]
        parts = _polygon_parts(repaired)
        if not parts:
            return None, warnings, ["KML polygon repair produced no usable Polygon or MultiPolygon geometry."]
        candidate = unary_union(parts)
        warnings.append("Invalid KML polygon geometry was repaired using buffer(0).")
    if candidate.geom_type not in _SUPPORTED_TYPES:
        parts = _polygon_parts(candidate)
        if not parts:
            return None, warnings, ["KML contains no supported Polygon or MultiPolygon geometry."]
        candidate = unary_union(parts)
    if candidate.is_empty or not candidate.is_valid:
        return None, warnings, ["KML polygon geometry is invalid after validation."]
    # This native-coordinate check only rejects degeneracy; it is not a m2 value.
    if candidate.area <= 0:
        return None, warnings, ["KML polygon geometry has zero area."]
    return candidate, warnings, []


def parse_kml(kml_text: str | bytes, *, source_name: str | None = None) -> AOIResult:
    """Parse all usable Polygon/MultiPolygon content from a KML string."""
    text = kml_text.decode("utf-8") if isinstance(kml_text, bytes) else kml_text
    try:
        document = kml.KML.from_string(text)
    except (TypeError, ValueError, SyntaxError) as error:
        return AOIResult(None, None, KML_SOURCE_CRS, False, (), (f"KML could not be parsed: {error}",), 0, source_name)

    polygons: list[Polygon] = []
    unsupported_types: set[str] = set()
    feature_count = 0

    def visit(feature: Any) -> None:
        nonlocal feature_count
        geometry = _to_shapely(getattr(feature, "geometry", None))
        if geometry is not None:
            parts = _polygon_parts(geometry)
            if parts:
                polygons.extend(parts)
                feature_count += 1
            else:
                unsupported_types.add(geometry.geom_type)
        for child in _children(feature):
            visit(child)

    visit(document)
    warnings = tuple(f"Unsupported KML geometry ignored: {item}." for item in sorted(unsupported_types))
    if not polygons:
        return AOIResult(None, None, KML_SOURCE_CRS, False, warnings, ("KML contains no supported Polygon or MultiPolygon geometry.",), 0, source_name)

    original = unary_union(polygons)
    geometry, validation_warnings, errors = _validate_geometry(original)
    return AOIResult(
        geometry, geometry.geom_type if geometry is not None else None, KML_SOURCE_CRS,
        geometry is not None and not errors, warnings + tuple(validation_warnings), tuple(errors),
        feature_count, source_name, original,
    )


def read_kml_geometry(path: str | Path) -> AOIResult:
    """Read a local KML file and return its validated WGS84 AOI geometry."""
    kml_path = Path(path)
    if not kml_path.is_file():
        raise FileNotFoundError(f"KML file not found: {kml_path}")
    return parse_kml(kml_path.read_bytes(), source_name=kml_path.name)
