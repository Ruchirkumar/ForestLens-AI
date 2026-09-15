# ForestLens AI Architecture

Browser
→ Streamlit
→ input validation
→ CRS/GSD quality gate
→ optional KML AOI
→ DeepForest inference
→ post-processing
→ crown geometry/area
→ uncertainty/sensitivity
→ spatial analytics
→ visualization/export

## Non-goals for the MVP

- biomass estimation
- carbon-stock calculation
- carbon-credit issuance
- database
- live satellite marketplace/API dependency
- LLM chatbot
- custom model training

## Geospatial foundation

Raster metadata is read independently of the UI. Detection accepts
non-georeferenced PNG/JPG/TIFF imagery, while physical area remains unavailable
unless a projected CRS supplies reliable linear units. Geographic CRSs are
classified but require a later, explicit projection step before planar area is
measured.

## KML AOI integration

KML AOIs are optional and are read as WGS84 (`EPSG:4326`) Polygon or
MultiPolygon geometry. Nested Documents, Folders, and Placemarks are searched
recursively, and all usable polygons are unioned. The AOI is transformed to the
raster CRS only when raster georeferencing is available, then intersected with
the full affine-transformed raster footprint. The resulting effective AOI is the
intersection for
partial overlaps. Image-pixel tree geometry is transformed through the raster
affine transform before its AOI comparison. Boundary-intersecting detections are
included once, while clipped footprints are retained for later metrics.

## Analytical metrics and provenance

The metrics layer is pure: it consumes AOI-filtered detections, their clipped
map geometry, AOI analysis, and raster metadata. It reports detected-model
counts, refinement/fallback/failed composition, model-score summaries, and
measurement provenance. Estimated canopy area is the unary union of valid,
clipped crown/proxy footprints in a metric CRS; individual areas are retained
only as diagnostics. This prevents overlapping detections from being double
counted. Trees touching an AOI boundary count once, but only their geometry
inside the effective AOI contributes to canopy area.

## Quality gate and robustness

Before analysis, deterministic image diagnostics inspect dimensions, sharpness
(variance of Laplacian), luminance, contrast, clipping, and available GSD.
Configurable thresholds are engineering heuristics rather than validated model
accuracy limits. A blocked image is not automatically analysed; unknown GSD is
a warning because non-georeferenced images remain useful for detection review.
Threshold sensitivity consumes one raw DeepForest inference result and reruns
only deterministic filtering/refinement for each threshold. It reports how
outputs change with the detection threshold, not a confidence interval.
