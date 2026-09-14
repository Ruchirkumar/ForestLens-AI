# Known Limitations

- Dense overlapping crowns can be missed or merged.
- Shadows and unusual crown shapes can reduce detection quality.
- Model performance can vary with imagery resolution and domain shift.
- A DeepForest detection box is not a measured crown boundary.
- Canopy area is an estimate unless a validated crown segmentation method is used.
- Non-georeferenced images need an explicit ground sampling distance for physical area.
- Model confidence must not be presented as accuracy.
- The prototype does not estimate biomass or carbon stock.

## Crown Footprint Estimation

DeepForest provides tree-crown object detections as bounding boxes. ForestLens
adds a local RGB image-derived refinement step to estimate a footprint within
each detection neighbourhood. A refined footprint is not ground-truth crown
segmentation: dense overlapping crowns, shadows, mixed vegetation, image
quality, and resolution can all cause errors.

When the local mask is absent or algorithmically implausible, ForestLens retains
an explicitly marked bounding-box fallback rather than fabricating a refined
footprint. The reported pixel count is not physical area until a valid raster
scale/GSD and affine transform are available.

## Geospatial Measurements

GeoTIFF imagery is preferred for later physical measurement because it can
carry a CRS and affine transform. PNG/JPG imagery remains detection-compatible
but normally lacks georeferencing. Geographic latitude/longitude CRSs are not
directly suitable for planar area calculations, and ForestLens never applies a
fixed degrees-to-metres conversion.

Valid georeferencing provides spatial scale; it does not make DeepForest boxes
or RGB-refined footprints exact. GSD is not model accuracy, and physical crown
or canopy area is not calculated at this stage.

## KML AOIs

KML AOI support accepts only Polygon and MultiPolygon content and assumes the
KML-standard WGS84 coordinates. Point and line-only KML files are not AOIs.
Missing raster georeferencing prevents reliable KML-to-raster matching; pixel
coordinates are never compared directly with longitude/latitude. A geographic
raster CRS is projected before planar m2 area is calculated. AOI clipping and
edge-tree inclusion prepare later measurements only: refined crown footprints
remain estimates and bbox fallbacks are not true segmentation.

## Analytical Metrics

“Detected trees” are DeepForest model predictions, not actual forest-population
counts. Estimated canopy area and estimated canopy cover depend on RGB-derived
footprints and bbox fallback proxies; neither is true semantic segmentation.
Overlapping footprints are unioned and AOI boundaries clip their measured area.
Physical metrics require valid spatial reference and a safe metric CRS; a
geographic raster is reprojected before planar measurement. Detection scores
are not accuracy, and refinement confidence is an algorithmic quality indicator,
not calibrated probability.

## Quality and Threshold Sensitivity

Sharpness, brightness, contrast, clipping, and resolution checks are engineering
diagnostics, not scientifically validated measures of detection accuracy. The
quality gate can flag clearly unusable imagery, but ordinary imagery variation
and unknown GSD produce warnings rather than claims about model performance.
Threshold sensitivity only shows how results change under selected detection
score thresholds; it is not an accuracy range, confidence interval, or error
margin. Pretrained DeepForest performance can vary by imagery resolution, forest
type, acquisition conditions, and domain. High bbox-fallback rates are reported
because fallback footprints remain area proxies rather than segmentation.
