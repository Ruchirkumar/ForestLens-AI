# ForestLens AI

Resolution-aware tree crown detection and canopy-area estimation from high-resolution forest imagery.

## First milestone

Validate the pretrained DeepForest detector on the actual forest imagery selected for the hackathon.
Do not build the full UI pipeline until this smoke test produces sensible detections.

## Windows setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run UI

```powershell
streamlit run app.py
```

## Smoke test

```powershell
python scripts_smoke_test.py
```

The smoke test uses DeepForest's built-in `OSBS_029.png` image by default. To
test a local RGB image instead, pass its path as the optional argument:

```powershell
python scripts_smoke_test.py "C:\path\to\forest.tif"
```

## Detection visualization

```powershell
python scripts_visualize_test.py
```

This writes an evidence overlay to `artifacts/deepforest_test.png`. The
pretrained DeepForest model performs tree-crown **object detection**: its boxes
represent model detections, and the returned `geometry` currently corresponds
to each detection bounding box. It is not an exact crown segmentation or an
exact canopy-area measurement.

## Crown-footprint refinement test

```powershell
python scripts_refinement_test.py
```

This runs a conservative, local RGB-derived footprint refinement on the model
detections and writes `artifacts/crown_refinement_test.png`. Refined footprints
are image-derived estimates only; any unsuccessful refinement remains visibly
and structurally marked as a bounding-box proxy. Pixel counts are not physical
areas without valid raster scale information.

## Geospatial metadata test

```powershell
python scripts_geospatial_test.py
```

GeoTIFF is preferred when a later workflow needs physical measurements. PNG/JPG
remain useful for detection and visual validation but normally have no spatial
reference. ForestLens only reports physical area when it can establish a
defensible metric basis from raster georeferencing or an explicitly supplied
GSD; it never turns latitude/longitude degrees into metres with a fixed
conversion. GSD is spatial scale, not model accuracy.

## Optional KML AOI test

```powershell
python scripts_aoi_test.py
```

An optional KML Area of Interest is interpreted as WGS84 (`EPSG:4326`) and may
contain Polygon or MultiPolygon features, including nested KML folders. All
polygon features are combined. ForestLens transforms the AOI into a valid raster
CRS before matching it; a PNG/JPG or other raster with no CRS cannot be matched
to KML coordinates. Geographic raster CRSs are projected before m2 area is
reported, never measured directly in degrees. Partially covered AOIs are clipped
to raster coverage. Trees intersecting an AOI edge are retained as detections;
their clipped geometry is preserved for a later canopy-area step.

## AOI analytical metrics test

```powershell
python scripts_metrics_test.py
```

Step 7 reports **Detected trees** as retained DeepForest model predictions, not
a true tree population. **Estimated canopy area** is the union of AOI-clipped
refined crown estimates and explicit bbox proxies, so overlapping footprints are
not summed twice. Bbox fallbacks remain visible in the metric provenance. A
valid spatial reference is required for m2, hectares, canopy cover, and
detected-tree density; geographic CRS geometry is reprojected to a metric CRS.
Detection scores are model scores, not accuracy, and refinement confidence is
an algorithmic quality indicator rather than a calibrated probability.

## Quality gate and threshold sensitivity

```powershell
python scripts_quality_test.py
```

Quality checks report deterministic engineering diagnostics—sharpness,
brightness, contrast, clipping, and available physical resolution—not a
scientific quality or accuracy score. Unknown GSD warns but does not block image
analysis. Detection-threshold sensitivity reuses one raw DeepForest prediction
set to show how reported detections change at 0.30, 0.40, 0.50, and 0.60; it is
not a confidence interval or error margin. The refinement/fallback composition
is reported transparently, and missing georeferencing still prevents physical
measurements.

## Engineering principles

- Never invent ground scale.
- Never invent accuracy.
- Confidence is not accuracy.
- Crown-area results are explicitly estimates.
- Detected tree crowns are image-derived model detections, not an exact tree count.
- Carbon-stock/carbon-credit calculations are outside the MVP.
- Demo data should be reproducible and properly attributed.

## Planned high-value features

- resolution/image quality gate
- tree crown count
- estimated detected canopy area
- KML AOI
- per-tree evidence
- confidence bands
- threshold sensitivity analysis
- spatial density grid
- run manifest
- CSV / GeoJSON / JSON export
