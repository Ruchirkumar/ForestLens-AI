
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import numpy as np

from src.detection.model import (
    DeepForestDetectionError,
    filter_detections,
    load_model,
    predict_detections,
)
from src.geometry.crown_refinement import refine_crowns
from src.io.kml_reader import read_kml_geometry
from src.io.raster_reader import read_raster_rgb
from src.metrics.analysis import calculate_analysis_metrics
from src.spatial.aoi import (
    analyze_aoi,
    analyze_raster_as_aoi,
)
from src.spatial.aoi_filter import filter_trees_to_aoi
from src.uncertainty.sensitivity import threshold_sensitivity
from src.visualization.overlay import render_detection_overlay


st.set_page_config(
    page_title="ForestLens AI",
    page_icon="🌲",
    layout="wide",
)


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .hero {
        padding: 1.6rem 1.8rem;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin-bottom: 0.25rem;
    }

    .hero p {
        margin-top: 0.25rem;
        color: #666;
        font-size: 1.05rem;
    }

    .section-note {
        color: #666;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_model():
    """Load and cache the DeepForest model."""
    return load_model()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def format_optional_number(
    value,
    suffix: str = "",
) -> str:
    if value is None:
        return "Unavailable"

    return f"{value:,.2f}{suffix}"


def metadata_warning_text(metadata) -> list[str]:
    warnings: list[str] = []

    if not metadata.georeferenced:
        warnings.append(
            "The raster has no usable georeferencing. "
            "Physical area and density measurements cannot be "
            "treated as spatially calibrated."
        )

    if metadata.crs is None:
        warnings.append(
            "No CRS is available for this raster."
        )

    if metadata.gsd_x_m is None or metadata.gsd_y_m is None:
        warnings.append(
            "Ground sampling distance could not be verified "
            "from the available raster metadata."
        )

    return warnings


def save_uploaded_file_temporarily(
    uploaded_file,
) -> str:
    """
    Save an uploaded file to a temporary filesystem path.

    The raster/KML readers in this project accept filesystem paths,
    so Streamlit's in-memory UploadedFile object is materialized
    temporarily for processing.
    """

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    temporary = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    try:
        temporary.write(
            uploaded_file.getvalue()
        )
        temporary.flush()
    finally:
        temporary.close()

    return temporary.name


def cleanup_temp_file(
    path: str | None,
) -> None:
    if not path:
        return

    try:
        Path(path).unlink(
            missing_ok=True
        )
    except OSError:
        pass


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🌲 ForestLens AI</h1>
        <p>
            Resolution-aware tree crown detection and canopy analysis
            from forest imagery.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Analysis Settings")

    threshold = st.slider(
        "Detection confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05,
        help=(
            "Only detections with a confidence score at or above "
            "this value are displayed."
        ),
    )

    st.caption(
        "Higher thresholds keep fewer, higher-confidence detections. "
        "Lower thresholds may detect more trees but can increase "
        "false positives."
    )

    st.divider()

    st.subheader("About ForestLens")

    st.write(
        "ForestLens AI uses a pretrained DeepForest detector to "
        "identify individual tree candidates and combines those "
        "detections with transparent geospatial and crown-refinement "
        "analysis."
    )

    st.info(
        "GeoTIFF imagery is recommended when physical area measurements "
        "are required because geospatial metadata can provide pixel size "
        "and coordinate reference information."
    )


# ---------------------------------------------------------------------
# Upload image
# ---------------------------------------------------------------------

st.subheader("1. Upload Forest Imagery")

uploaded_file = st.file_uploader(
    "Choose a forest image",
    type=[
        "tif",
        "tiff",
        "png",
        "jpg",
        "jpeg",
    ],
    help=(
        "High-resolution RGB imagery is recommended. "
        "GeoTIFF is preferred for geospatial analysis."
    ),
)


# ---------------------------------------------------------------------
# Upload KML
# ---------------------------------------------------------------------

st.subheader("2. Optional Area of Interest")

kml_file = st.file_uploader(
    "Upload KML boundary",
    type=["kml"],
    help="Upload a Polygon or MultiPolygon boundary.",
)


if uploaded_file is None:
    st.info(
        "Upload a forest image above to start the analysis."
    )

    st.markdown(
        """
        ### Recommended input

        - High-resolution aerial or satellite forest imagery
        - RGB imagery
        - GeoTIFF when spatial measurements are required
        - Images where individual tree crowns are reasonably visible

        ### Analysis flow

        **Image → DeepForest detection → Crown refinement → "
        "Optional KML AOI → Canopy metrics → Sensitivity analysis → Results**
        """
    )

    st.stop()


# ---------------------------------------------------------------------
# Read raster
# ---------------------------------------------------------------------

raster_temp_path: str | None = None

try:
    raster_temp_path = (
        save_uploaded_file_temporarily(
            uploaded_file
        )
    )

    raster_image, metadata = read_raster_rgb(
        raster_temp_path
    )

except Exception as error:
    cleanup_temp_file(
        raster_temp_path
    )

    st.error(
        f"Could not read the uploaded image: {error}"
    )

    st.info(
        "Please upload a valid RGB/RGBA PNG, JPEG, or GeoTIFF "
        "image within the supported project limits."
    )

    st.stop()


# ---------------------------------------------------------------------
# Physical measurement scale
# ---------------------------------------------------------------------

# None means that no manual GSD is being supplied.
# When verified raster GSD exists, the metrics engine will use the
# metadata-derived measurement basis automatically.
manual_gsd_m: float | None = None


# ---------------------------------------------------------------------
# Preview + metadata
# ---------------------------------------------------------------------

st.subheader("3. Image & Spatial Metadata")

preview_col, info_col = st.columns(
    [2.2, 1]
)

with preview_col:
    preview_image = raster_image

    if preview_image.dtype != np.uint8:
        preview_image = np.clip(
            preview_image,
            0.0,
            255.0,
        ).astype(
            np.uint8
        )

    st.image(
        preview_image,
        caption=uploaded_file.name,
        use_container_width=True,
    )

with info_col:
    st.metric(
        "Width",
        f"{raster_image.shape[1]:,} px",
    )

    st.metric(
        "Height",
        f"{raster_image.shape[0]:,} px",
    )

    st.metric(
        "Channels",
        f"{raster_image.shape[2]}",
    )

    if uploaded_file.size:
        st.metric(
            "File size",
            f"{uploaded_file.size / (1024 * 1024):.2f} MB",
        )

    st.divider()

    st.write(
        "**CRS**",
        (
            metadata.crs
            if metadata.crs is not None
            else "Unavailable"
        ),
    )

    st.write(
        "**Georeferenced**",
        (
            "Yes"
            if metadata.georeferenced
            else "No"
        ),
    )

    st.write(
        "**GSD**",
        (
            f"{max(metadata.gsd_x_m, metadata.gsd_y_m):.3f} m"
            if (
                metadata.gsd_x_m is not None
                and metadata.gsd_y_m is not None
            )
            else "Unavailable"
        ),
    )

    st.write(
        "**Pixel area**",
        (
            f"{metadata.pixel_area_native:.4f} m²"
            if metadata.physical_area_available
            else "Unavailable"
        ),
    )


raster_warnings = metadata_warning_text(
    metadata
)

for warning in raster_warnings:
    st.warning(
        warning
    )


# ---------------------------------------------------------------------
# Physical Measurement Scale
# ---------------------------------------------------------------------

st.subheader("4. Physical Measurement Scale")

if (
    metadata.gsd_x_m is not None
    and metadata.gsd_y_m is not None
):
    st.success(
        "Ground sampling distance is available from raster metadata. "
        "Metadata-derived scale will be used automatically."
    )

else:
    st.info(
        "This image does not contain a verified ground sampling distance. "
        "If you know the image GSD, enter it below to enable physical "
        "canopy-area and hectare-based measurements."
    )

    manual_gsd_input = st.number_input(
        "Known GSD (metres per pixel)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.01,
        format="%.4f",
        help=(
            "Example: 0.10 means one pixel represents 10 cm "
            "on the ground. Leave 0 to keep physical metrics unavailable."
        ),
    )

    if manual_gsd_input > 0:
        manual_gsd_m = float(
            manual_gsd_input
        )

        st.caption(
            f"Using user-provided scale: "
            f"{manual_gsd_m:.4f} m/pixel"
        )

        st.warning(
            "This GSD was supplied by the user and was not independently "
            "verified from the image."
        )


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------

st.divider()

run_analysis = st.button(
    "Run Forest Analysis",
    type="primary",
    use_container_width=True,
)


if not run_analysis:
    cleanup_temp_file(
        raster_temp_path
    )

    st.caption(
        "Adjust the detection threshold if required, then click "
        "**Run Forest Analysis**."
    )

    st.stop()


progress = st.progress(0)
status = st.empty()


try:
    # ---------------------------------------------------------------
    # Step 1: Model
    # ---------------------------------------------------------------

    status.info(
        "Loading DeepForest model..."
    )

    progress.progress(
        10
    )

    model = get_model()

    # ---------------------------------------------------------------
    # Step 2: Detection
    # ---------------------------------------------------------------

    status.info(
        "Detecting individual tree candidates..."
    )

    progress.progress(
        25
    )

    raw_predictions = predict_detections(
        model,
        raster_image,
    )

    predictions = filter_detections(
        raw_predictions,
        score_threshold=threshold,
    )

    progress.progress(
        45
    )

    # ---------------------------------------------------------------
    # Step 3: Crown refinement
    # ---------------------------------------------------------------

    status.info(
        "Refining detected crown footprints..."
    )

    progress.progress(
        50
    )

    crown_results = refine_crowns(
        raster_image,
        predictions,
    )

    progress.progress(
        65
    )

    # ---------------------------------------------------------------
    # Step 4: AOI
    # ---------------------------------------------------------------

    status.info(
        "Preparing analysis region..."
    )

    if kml_file is not None:
        # -----------------------------------------------------------
        # KML AOI
        # -----------------------------------------------------------

        kml_temp_path: str | None = None

        try:
            kml_temp_path = (
                save_uploaded_file_temporarily(
                    kml_file
                )
            )

            aoi = read_kml_geometry(
                kml_temp_path
            )

        finally:
            cleanup_temp_file(
                kml_temp_path
            )

        if not aoi.valid or aoi.geometry is None:
            st.warning(
                "The KML could not provide a valid polygonal AOI. "
                "The complete raster will be used as the analysis region."
            )

            for error in aoi.errors:
                st.error(
                    error
                )

            aoi_analysis = (
                analyze_raster_as_aoi(
                    metadata
                )
            )

        else:
            aoi_analysis = analyze_aoi(
                aoi,
                metadata,
            )

            if not aoi_analysis.aoi_spatial_match_available:
                st.warning(
                    "The KML could not be spatially matched to the "
                    "uploaded raster. The complete raster will be "
                    "used as the analysis region instead."
                )

                for warning in aoi_analysis.warnings:
                    st.warning(
                        warning
                    )

                aoi_analysis = (
                    analyze_raster_as_aoi(
                        metadata
                    )
                )

            elif aoi_analysis.overlap_status == "outside":
                st.error(
                    "The uploaded KML AOI does not overlap the raster. "
                    "No trees are included from that AOI."
                )

    else:
        # -----------------------------------------------------------
        # Whole raster is the AOI
        # -----------------------------------------------------------

        aoi_analysis = (
            analyze_raster_as_aoi(
                metadata
            )
        )

    # ---------------------------------------------------------------
    # Step 5: AOI filtering
    # ---------------------------------------------------------------

    aoi_filter = filter_trees_to_aoi(
        crown_results,
        metadata,
        aoi_analysis,
    )

    progress.progress(
        80
    )

    # ---------------------------------------------------------------
    # Step 6: Metrics
    # ---------------------------------------------------------------

    status.info(
        "Calculating canopy and density metrics..."
    )

    metrics = calculate_analysis_metrics(
        crown_results,
        aoi_filter,
        aoi_analysis,
        metadata,
        manual_gsd_m=manual_gsd_m,
    )

    progress.progress(
        90
    )

    # ---------------------------------------------------------------
    # Step 7: Visualization
    # ---------------------------------------------------------------

    status.info(
        "Generating detection visualization..."
    )

    overlay = render_detection_overlay(
        raster_image,
        predictions,
        show_ids=True,
        show_scores=True,
    )

    progress.progress(
        100
    )

    status.success(
        "Forest analysis completed."
    )

finally:
    cleanup_temp_file(
        raster_temp_path
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

st.divider()

st.subheader(
    "Analysis Summary"
)

summary = {
    "detected_trees": len(predictions),
    "mean_confidence": (
        float(
            predictions["score"].mean()
        )
        if len(predictions) > 0
        else 0.0
    ),
    "max_confidence": (
        float(
            predictions["score"].max()
        )
        if len(predictions) > 0
        else 0.0
    ),
}


metric_1, metric_2, metric_3, metric_4 = st.columns(
    4
)

with metric_1:
    st.metric(
        "Detected Trees",
        f"{summary['detected_trees']:,}",
    )

with metric_2:
    st.metric(
        "Mean Confidence",
        f"{summary['mean_confidence']:.1%}",
    )

with metric_3:
    st.metric(
        "Refined Crowns",
        f"{metrics.refined_tree_count:,}",
    )

with metric_4:
    st.metric(
        "BBox Fallbacks",
        f"{metrics.bbox_fallback_count:,}",
    )


# ---------------------------------------------------------------------
# Canopy metrics
# ---------------------------------------------------------------------

st.subheader(
    "Canopy & Spatial Metrics"
)

canopy_1, canopy_2, canopy_3, canopy_4 = st.columns(
    4
)

with canopy_1:
    st.metric(
        "Trees in Analysis Region",
        f"{metrics.detected_tree_count:,}",
    )

with canopy_2:
    st.metric(
        "Estimated Canopy Area",
        format_optional_number(
            metrics.estimated_canopy_area_m2,
            " m²",
        ),
    )

with canopy_3:
    st.metric(
        "Estimated Canopy Cover",
        (
            f"{metrics.estimated_canopy_cover_percent:.2f}%"
            if metrics.estimated_canopy_cover_percent is not None
            else "Unavailable"
        ),
    )

with canopy_4:
    st.metric(
        "Tree Density",
        (
            f"{metrics.detected_tree_density_per_ha:.2f} trees/ha"
            if metrics.detected_tree_density_per_ha is not None
            else "Unavailable"
        ),
    )


# ---------------------------------------------------------------------
# Measurement provenance
# ---------------------------------------------------------------------

st.subheader(
    "Measurement Basis"
)

st.write(
    "**Measurement basis:** "
    f"`{metrics.measurement_basis}`"
)

if metrics.measurement_status.spatial_reference:
    st.write(
        "**Spatial reference:** "
        f"`{metrics.measurement_status.spatial_reference}`"
    )

if metrics.measurement_status.metric_crs:
    st.write(
        "**Metric CRS:** "
        f"`{metrics.measurement_status.metric_crs}`"
    )

if metrics.measurement_status.warnings:
    for warning in metrics.measurement_status.warnings:
        st.warning(
            warning
        )


# ---------------------------------------------------------------------
# AOI information
# ---------------------------------------------------------------------

st.subheader(
    "Analysis Region"
)

if kml_file is None:
    st.info(
        "No KML boundary was supplied. "
        "The complete uploaded raster is being used as the "
        "analysis region."
    )

else:
    if aoi_analysis.overlap_status == "inside":
        st.success(
            "The KML AOI is fully inside the raster footprint."
        )

    elif aoi_analysis.overlap_status == "partial_overlap":
        st.warning(
            "The KML AOI partially overlaps the raster. "
            "Only the overlapping region is analyzed."
        )

    elif aoi_analysis.overlap_status == "outside":
        st.error(
            "The KML AOI is outside the raster footprint. "
            "No trees from the raster are included for this AOI."
        )

    else:
        st.info(
            "AOI spatial matching is unavailable."
        )

if aoi_analysis.overlap_fraction is not None:
    st.write(
        "**AOI/raster overlap fraction:** "
        f"{aoi_analysis.overlap_fraction:.1%}"
    )

if aoi_analysis.warnings:
    for warning in aoi_analysis.warnings:
        st.warning(
            warning
        )

if aoi_analysis.errors:
    for error in aoi_analysis.errors:
        st.error(
            error
        )


# ---------------------------------------------------------------------
# Detection visualization
# ---------------------------------------------------------------------

st.subheader(
    "Detected Tree Candidates"
)

st.image(
    overlay,
    use_container_width=True,
)

st.caption(
    "Red bounding boxes represent DeepForest tree detections. "
    "They are detection boxes, not exact crown segmentation. "
    "Refined crown footprints are used separately for canopy "
    "metrics when the refinement result is valid."
)


# ---------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------

st.subheader(
    "Detection Reliability"
)

tree_count = len(
    predictions
)

mean_confidence = summary[
    "mean_confidence"
]

if tree_count == 0:
    st.warning(
        "No trees passed the selected confidence threshold. "
        "Try a lower threshold or a different image."
    )

elif mean_confidence < 0.50:
    st.warning(
        "The average model confidence is relatively low. "
        "Treat the detected tree count as an estimate and "
        "review the visual detections."
    )

elif mean_confidence < 0.70:
    st.info(
        "The model produced moderate-confidence detections. "
        "Visual review is recommended before operational use."
    )

else:
    st.success(
        "The model produced relatively strong confidence scores "
        "for the selected threshold."
    )

st.caption(
    "Model confidence is not the same as validated detection accuracy."
)


# ---------------------------------------------------------------------
# Crown refinement summary
# ---------------------------------------------------------------------

st.subheader(
    "Crown Refinement"
)

refinement_total = (
    metrics.refined_tree_count
    + metrics.bbox_fallback_count
    + metrics.failed_tree_count
)

if refinement_total > 0:
    refinement_rate = (
        metrics.refined_tree_count
        / refinement_total
    )
else:
    refinement_rate = 0.0


ref_1, ref_2, ref_3 = st.columns(
    3
)

with ref_1:
    st.metric(
        "Refined",
        f"{metrics.refined_tree_count:,}",
    )

with ref_2:
    st.metric(
        "BBox Fallback",
        f"{metrics.bbox_fallback_count:,}",
    )

with ref_3:
    st.metric(
        "Failed",
        f"{metrics.failed_tree_count:,}",
    )

st.caption(
    f"Refinement success rate: {refinement_rate:.1%}. "
    "BBox fallbacks are retained as detection proxies but should "
    "not be interpreted as exact crown segmentation."
)


# ---------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------

st.subheader(
    "Detection Sensitivity"
)

st.write(
    "The same raw DeepForest predictions are evaluated at multiple "
    "confidence thresholds. This shows how sensitive the estimated "
    "tree count is to the selected confidence cutoff."
)

try:
    sensitivity_results = threshold_sensitivity(
        raw_predictions
    )

    if sensitivity_results:
        sensitivity_df = pd.DataFrame(
            sensitivity_results
        )

        rename_map = {
            "threshold": "Threshold",
            "detections": "Detected Trees",
            "refined": "Refined Crowns",
            "fallback": "BBox Fallbacks",
            "failed": "Failed Refinements",
        }

        sensitivity_display = sensitivity_df.rename(
            columns=rename_map
        )

        if "Threshold" in sensitivity_display.columns:
            sensitivity_display[
                "Threshold"
            ] = (
                sensitivity_display[
                    "Threshold"
                ]
                * 100
            ).round(
                0
            ).astype(
                int
            ).astype(
                str
            ) + "%"

        st.dataframe(
            sensitivity_display,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Sensitivity analysis reuses the same raw model predictions. "
            "The detector is not re-run for every threshold."
        )

    else:
        st.info(
            "No sensitivity results were generated."
        )

except Exception as error:
    st.warning(
        "Sensitivity analysis could not be generated: "
        f"{error}"
    )


# ---------------------------------------------------------------------
# Detection table
# ---------------------------------------------------------------------

st.subheader(
    "Detection Data"
)

if tree_count > 0:
    display_columns = [
        "tree_id",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "label",
        "score",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in predictions.columns
    ]

    table = predictions[
        available_columns
    ].copy()

    if "score" in table.columns:
        table[
            "score"
        ] = table[
            "score"
        ].round(
            4
        )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = predictions.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        label="Download Detection CSV",
        data=csv_data,
        file_name="forestlens_detections.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:
    st.info(
        "No detection records are available for the selected threshold."
    )


# ---------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------

st.subheader(
    "Important Interpretation"
)

st.warning(
    "DeepForest detections are bounding-box based. They should not "
    "be interpreted as exact tree crown segmentation. Crown area "
    "values are estimates derived from the available refinement "
    "method and should be reviewed before operational use."
)

st.write(
    "Physical canopy measurements are reported only when the raster "
    "metadata provides a defensible spatial measurement basis or "
    "when a user-provided GSD is explicitly supplied for a complete "
    "raster analysis. Images without sufficient geospatial information "
    "remain useful for tree detection but are not assigned fabricated "
    "physical areas."
)


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.divider()

st.caption(
    "ForestLens AI • Tree detection from forest imagery • "
    "Results should be visually reviewed before operational decisions."
)

