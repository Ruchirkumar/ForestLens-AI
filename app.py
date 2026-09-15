from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.detection.model import (
    DeepForestDetectionError,
    filter_detections,
    load_model,
    predict_detections,
)
from src.config import load_config
from src.geometry.crown_refinement import refine_crowns
from src.io.kml_reader import read_kml_geometry
from src.io.raster_reader import read_raster_rgb
from src.metrics.analysis import calculate_analysis_metrics
from src.quality.assessment import QualityConfig, evaluate_quality
from src.quality.warnings import generate_warnings
from src.spatial.aoi import analyze_aoi, analyze_raster_as_aoi
from src.spatial.aoi_filter import filter_trees_to_aoi
from src.uncertainty.sensitivity import calculate_refinement_summary, threshold_sensitivity
from src.visualization.crown_overlay import render_crown_refinement_overlay


st.set_page_config(
    page_title="ForestLens AI — Forest Intelligence",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------
st.markdown(
    """
<style>
:root {
    --fl-ink: #10231a;
    --fl-muted: #617067;
    --fl-green: #176b45;
    --fl-green-dark: #0d4b31;
    --fl-green-soft: #eaf5ee;
    --fl-border: #dce7df;
    --fl-surface: #ffffff;
    --fl-bg: #f5f8f6;
    --fl-warn: #8a5a00;
    --fl-danger: #a33a3a;
}

.stApp {
    background:
        radial-gradient(circle at 85% 4%, rgba(37, 122, 78, 0.07), transparent 28rem),
        linear-gradient(180deg, #f8fbf9 0%, #f4f7f5 100%);
    color: var(--fl-ink);
}

.block-container {
    max-width: 1420px;
    padding-top: 1.1rem;
    padding-bottom: 4rem;
}

header[data-testid="stHeader"] {
    background: rgba(248, 251, 249, 0.86);
}

[data-testid="stSidebar"] {
    background: #10231a;
}

[data-testid="stSidebar"] * {
    color: #eef7f1 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #c7d9ce !important;
}

[data-testid="stSidebar"] .stSlider label {
    color: #eef7f1 !important;
}

.fl-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: .25rem 0 1.1rem 0;
    border-bottom: 1px solid var(--fl-border);
    margin-bottom: 1.5rem;
}

.fl-brand {
    display: flex;
    align-items: center;
    gap: .7rem;
    font-weight: 750;
    letter-spacing: -.02em;
    font-size: 1.08rem;
}

.fl-mark {
    width: 36px;
    height: 36px;
    display: grid;
    place-items: center;
    border-radius: 11px;
    background: var(--fl-green);
    color: white;
    font-size: 1.15rem;
}

.fl-nav-right {
    color: var(--fl-muted);
    font-size: .86rem;
}

.fl-hero {
    position: relative;
    overflow: hidden;
    border: 1px solid #cfe1d5;
    border-radius: 24px;
    padding: 3.2rem 3.1rem;
    background:
        linear-gradient(120deg, rgba(255,255,255,.98), rgba(238,248,241,.92)),
        radial-gradient(circle at 92% 30%, rgba(23,107,69,.15), transparent 24rem);
    box-shadow: 0 18px 55px rgba(25, 58, 39, .08);
    margin-bottom: 1.5rem;
}

.fl-kicker {
    display: inline-block;
    color: var(--fl-green-dark);
    background: var(--fl-green-soft);
    border: 1px solid #cce2d3;
    border-radius: 999px;
    padding: .35rem .7rem;
    font-size: .75rem;
    font-weight: 750;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.fl-hero h1 {
    margin: .85rem 0 .55rem 0;
    font-size: clamp(2.35rem, 5vw, 4.6rem);
    line-height: .98;
    letter-spacing: -.055em;
    color: var(--fl-ink);
}

.fl-hero p {
    max-width: 760px;
    margin: 0;
    color: #4f6258;
    font-size: 1.08rem;
    line-height: 1.65;
}

.fl-pills {
    display: flex;
    flex-wrap: wrap;
    gap: .55rem;
    margin-top: 1.4rem;
}

.fl-pill {
    border: 1px solid var(--fl-border);
    background: rgba(255,255,255,.72);
    border-radius: 999px;
    padding: .42rem .72rem;
    color: #41534a;
    font-size: .78rem;
    font-weight: 650;
}

.fl-section {
    margin: 2rem 0 .75rem 0;
}

.fl-section h2 {
    font-size: 1.38rem;
    letter-spacing: -.025em;
    margin-bottom: .25rem;
}

.fl-section p {
    color: var(--fl-muted);
    margin-top: 0;
}

.fl-card {
    background: var(--fl-surface);
    border: 1px solid var(--fl-border);
    border-radius: 18px;
    padding: 1.15rem 1.2rem;
    box-shadow: 0 7px 28px rgba(23, 49, 35, .045);
}

.fl-status {
    border-radius: 14px;
    padding: .8rem 1rem;
    background: var(--fl-green-soft);
    border: 1px solid #cbe2d2;
    color: #1b5b3c;
    font-size: .88rem;
}

.fl-status-warn {
    background: #fff8e9;
    border-color: #f0dfad;
    color: var(--fl-warn);
}

.fl-status-danger {
    background: #fff0f0;
    border-color: #efcccc;
    color: var(--fl-danger);
}

.fl-metric {
    min-height: 122px;
    padding: 1rem 1.05rem;
    border: 1px solid var(--fl-border);
    border-radius: 16px;
    background: white;
    box-shadow: 0 7px 22px rgba(23,49,35,.04);
}

.fl-metric-label {
    color: var(--fl-muted);
    font-size: .77rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .055em;
}

.fl-metric-value {
    margin-top: .35rem;
    color: var(--fl-ink);
    font-size: 1.85rem;
    line-height: 1.05;
    font-weight: 780;
    letter-spacing: -.04em;
}

.fl-metric-help {
    margin-top: .45rem;
    color: #738279;
    font-size: .76rem;
}

.fl-provenance {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: .8rem;
}

.fl-provenance-item {
    padding: .9rem;
    border: 1px solid var(--fl-border);
    border-radius: 13px;
    background: #fbfdfb;
}

.fl-provenance-item span {
    display: block;
    color: #75837b;
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .05em;
    font-weight: 700;
}

.fl-provenance-item strong {
    display: block;
    margin-top: .3rem;
    color: var(--fl-ink);
    font-size: .92rem;
}

.fl-footer {
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--fl-border);
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    color: #718078;
    font-size: .78rem;
}

div[data-testid="stFileUploader"] {
    border-radius: 16px;
}

button[kind="primary"] {
    border-radius: 11px;
}

@media (max-width: 800px) {
    .fl-hero {
        padding: 2rem 1.35rem;
    }
    .fl-provenance {
        grid-template-columns: 1fr;
    }
    .fl-footer {
        flex-direction: column;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


def save_uploaded_file_temporarily(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temporary.write(uploaded_file.getvalue())
        temporary.flush()
    finally:
        temporary.close()
    return temporary.name


def cleanup_temp_file(path: str | None) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="fl-metric">
            <div class="fl-metric-label">{label}</div>
            <div class="fl-metric-value">{value}</div>
            <div class="fl-metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def optional_area(value) -> str:
    return "Unavailable" if value is None else f"{value:,.2f} m²"


def confidence_state(score: float) -> tuple[str, str]:
    if score < 0.50:
        return "Review required", "fl-status fl-status-warn"
    if score < 0.70:
        return "Moderate confidence", "fl-status fl-status-warn"
    return "Strong model confidence", "fl-status"


def measurement_label(metrics) -> tuple[str, str]:
    basis = metrics.measurement_basis
    if basis == "georeferenced_metric_raster":
        return "Verified spatial scale", "Metadata / metric CRS"
    if basis == "geographic_raster_reprojected_to_metric_crs":
        return "Verified spatial scale", "Geographic raster reprojected"
    if basis == "user_provided_gsd":
        return "User-provided scale", "Unverified GSD"
    if basis == "pixel_only":
        return "Pixel-only analysis", "No defensible physical scale"
    return "Physical scale unavailable", "No safe measurement basis"


def render_sensitivity_chart(raw_predictions: pd.DataFrame) -> None:
    results = threshold_sensitivity(raw_predictions)
    if not results:
        st.info("No sensitivity results are available.")
        return

    df = pd.DataFrame(results)
    if "threshold" not in df or "tree_count" not in df:
        st.info("Sensitivity output did not contain the expected fields.")
        return

    st.line_chart(
        df,
        x="threshold",
        y="tree_count",
        x_label="Confidence threshold",
        y_label="Detected tree candidates",
    )

    display = df.rename(
        columns={
            "threshold": "Threshold",
            "tree_count": "Tree candidates",
            "refined": "Refined",
            "fallback": "BBox proxy",
            "failed": "Failed",
        }
    )
    st.dataframe(display, hide_index=True)


try:
    APP_CONFIG = load_config()
    MAX_UPLOAD_MB = int(APP_CONFIG["app"]["max_upload_mb"])
    MAX_PIXELS = int(APP_CONFIG["input"]["max_pixels"])
    QUALITY_CONFIG = QualityConfig.from_mapping(APP_CONFIG.get("quality"))
except (KeyError, TypeError, ValueError) as error:
    st.error(f"ForestLens configuration is invalid: {error}")
    st.stop()


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="fl-nav">
        <div class="fl-brand">
            <div class="fl-mark">🌲</div>
            <div>ForestLens AI</div>
        </div>
        <div class="fl-nav-right">Forest intelligence · no account required</div>
    </div>

    <div class="fl-hero">
        <span class="fl-kicker">AI-assisted forest analysis</span>
        <h1>See the forest.<br>Quantify the canopy.</h1>
        <p>
            Detect individual tree candidates from high-resolution forest imagery,
            evaluate crown evidence, and calculate spatial measurements only when
            a defensible image scale is available.
        </p>
        <div class="fl-pills">
            <span class="fl-pill">DeepForest detection</span>
            <span class="fl-pill">AOI-aware analysis</span>
            <span class="fl-pill">GSD-aware measurements</span>
            <span class="fl-pill">Transparent confidence</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Analysis controls")
    threshold = st.slider(
        "Detection confidence threshold",
        0.10,
        0.90,
        0.30,
        0.05,
        help="Detections below this confidence are excluded from the reported count.",
    )
    st.caption(
        "This changes the reported detection set; it does not retrain or recalibrate "
        "the model."
    )

    st.divider()
    st.markdown("### Input guidance")
    st.markdown(
        "- **GeoTIFF** is preferred for spatial measurements.\n"
        "- RGB/RGBA imagery is supported.\n"
        "- Clear, high-resolution crowns produce more useful detections.\n"
        "- Physical area requires verified raster scale or an explicitly supplied GSD."
    )

# ---------------------------------------------------------------------
# Input workspace
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="fl-section">
        <h2>01 · Input workspace</h2>
        <p>Upload imagery and, optionally, define a geographic analysis boundary.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_col, aoi_col = st.columns([1.6, 1], gap="large")

with upload_col:
    uploaded_file = st.file_uploader(
        "Forest imagery",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        help="GeoTIFF is recommended when you need physical canopy area or density.",
    )

with aoi_col:
    kml_file = st.file_uploader(
        "Optional KML boundary",
        type=["kml"],
        help="Use a Polygon or MultiPolygon to restrict analysis spatially.",
    )

if uploaded_file is not None and uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
    st.error(
        f"This file is larger than the configured {MAX_UPLOAD_MB} MB upload limit. "
        "Use a smaller image or raise the documented project limit."
    )
    st.stop()

if uploaded_file is None:
    st.markdown(
        """
        <div class="fl-card">
            <strong>Start with a real forest image</strong><br>
            <span style="color:#68786f">
            ForestLens can still count tree candidates from a non-georeferenced
            image, but it will not invent real-world area measurements without a
            defensible scale.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="fl-footer">
            <span>ForestLens AI · AI-assisted tree and canopy analysis</span>
            <span>Detection confidence is not validated accuracy.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------
# Read image
# ---------------------------------------------------------------------
raster_temp_path = None
try:
    raster_temp_path = save_uploaded_file_temporarily(uploaded_file)
    raster_image, metadata = read_raster_rgb(raster_temp_path, max_pixels=MAX_PIXELS)
except Exception as error:
    cleanup_temp_file(raster_temp_path)
    st.error(f"Unable to read this image: {error}")
    st.info(
        "Use a supported RGB/RGBA PNG, JPEG, or GeoTIFF image. "
        "For spatial analysis, prefer GeoTIFF."
    )
    st.stop()

manual_gsd_m: float | None = None

try:
    quality_assessment = evaluate_quality(raster_image, metadata, QUALITY_CONFIG)
except Exception as error:
    cleanup_temp_file(raster_temp_path)
    st.error(f"Image quality checks could not be completed: {error}")
    st.stop()

if not quality_assessment.can_analyze:
    cleanup_temp_file(raster_temp_path)
    st.error("This image failed the basic usability gate, so analysis was not run.")
    for reason in quality_assessment.blocking_reasons:
        st.error(reason)
    st.stop()

# ---------------------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="fl-section">
        <h2>02 · Dataset overview</h2>
        <p>Review the image and measurement provenance before running inference.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

preview_col, metadata_col = st.columns([1.75, 1], gap="large")

with preview_col:
    preview = raster_image
    if preview.dtype != np.uint8:
        preview = np.clip(preview, 0, 255).astype(np.uint8)
    st.image(preview, caption=uploaded_file.name, width="stretch")

with metadata_col:
    st.markdown('<div class="fl-card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Width", f"{raster_image.shape[1]:,} px")
        st.metric("Height", f"{raster_image.shape[0]:,} px")
    with c2:
        st.metric("Channels", str(raster_image.shape[2]))
        st.metric(
            "File size",
            f"{uploaded_file.size / (1024 * 1024):.2f} MB"
            if uploaded_file.size
            else "—",
        )
    st.divider()
    st.write(f"**CRS:** `{metadata.crs or 'Unavailable'}`")
    st.write(
        f"**Georeferenced:** "
        f"{'Yes' if metadata.georeferenced else 'No'}"
    )
    if metadata.gsd_x_m is not None and metadata.gsd_y_m is not None:
        st.write(
            f"**GSD:** `{max(metadata.gsd_x_m, metadata.gsd_y_m):.4f} m/pixel`"
        )
    else:
        st.write("**GSD:** `Unavailable`")
    st.markdown("</div>", unsafe_allow_html=True)

if quality_assessment.warnings:
    st.warning("Image diagnostics require review before interpreting detections.")
    for warning in quality_assessment.warnings:
        st.caption(warning)

# ---------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="fl-section">
        <h2>03 · Measurement scale</h2>
        <p>Physical metrics are enabled only when image scale is defensible.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if metadata.gsd_x_m is not None and metadata.gsd_y_m is not None:
    st.markdown(
        """
        <div class="fl-status">
            <strong>Verified spatial scale</strong><br>
            Ground sampling distance was read from the raster metadata.
            Physical measurements can use the metadata-derived scale.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    scale_left, scale_right = st.columns([1, 1.4], gap="large")
    with scale_left:
        manual_gsd_input = st.number_input(
            "Known GSD (metres / pixel)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.01,
            format="%.4f",
            help="Example: 0.10 means one pixel represents 10 cm on the ground.",
        )
        if manual_gsd_input > 0:
            manual_gsd_m = float(manual_gsd_input)
    with scale_right:
        if manual_gsd_m is not None:
            st.markdown(
                f"""
                <div class="fl-status fl-status-warn">
                    <strong>User-provided scale</strong><br>
                    Using <b>{manual_gsd_m:.4f} m/pixel</b>. This value is not
                    independently verified from the image and does not create
                    georeferencing or enable KML alignment.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="fl-status fl-status-warn">
                    <strong>Physical scale unavailable</strong><br>
                    Tree detection can still run. Canopy area and hectare-based
                    density remain unavailable until a verified or user-supplied
                    GSD is available.
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="fl-section">
        <h2>04 · Analysis</h2>
        <p>Run the same transparent pipeline used for the reported results.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

run_analysis = st.button(
    "Run forest analysis",
    type="primary",
    width="stretch",
)

if not run_analysis:
    cleanup_temp_file(raster_temp_path)
    st.stop()

progress = st.progress(0)
status = st.empty()
started = time.perf_counter()

try:
    status.info("Loading DeepForest model…")
    progress.progress(10)
    model = get_model()

    status.info("Detecting individual tree candidates…")
    progress.progress(25)
    raw_predictions = predict_detections(model, raster_image)
    predictions = filter_detections(
        raw_predictions,
        score_threshold=threshold,
    )

    status.info("Refining crown evidence…")
    progress.progress(48)
    crown_results = refine_crowns(raster_image, predictions)

    status.info("Preparing analysis region…")
    progress.progress(63)

    # KML failures never silently become a different AOI.
    if kml_file is not None:
        kml_temp_path = None
        try:
            kml_temp_path = save_uploaded_file_temporarily(kml_file)
            aoi = read_kml_geometry(kml_temp_path)
        finally:
            cleanup_temp_file(kml_temp_path)

        if not aoi.valid or aoi.geometry is None:
            st.error(
                "The uploaded KML does not contain a valid polygonal AOI. "
                "Analysis was stopped to prevent an unintended full-raster fallback."
            )
            for error in aoi.errors:
                st.error(error)
            cleanup_temp_file(raster_temp_path)
            st.stop()

        aoi_analysis = analyze_aoi(aoi, metadata)

        if not aoi_analysis.aoi_spatial_match_available:
            st.error(
                "The KML could not be spatially aligned with the raster. "
                "Analysis was stopped rather than silently switching to the full raster."
            )
            for warning in aoi_analysis.warnings:
                st.warning(warning)
            cleanup_temp_file(raster_temp_path)
            st.stop()

        if aoi_analysis.overlap_status == "outside":
            st.error(
                "The KML AOI is outside the raster footprint. Analysis was stopped "
                "rather than reporting an empty or full-raster result."
            )
            cleanup_temp_file(raster_temp_path)
            st.stop()
    else:
        aoi_analysis = analyze_raster_as_aoi(metadata)

    aoi_filter = filter_trees_to_aoi(
        crown_results,
        metadata,
        aoi_analysis,
    )

    status.info("Calculating spatial and canopy metrics…")
    progress.progress(80)

    metrics = calculate_analysis_metrics(
        crown_results,
        aoi_filter,
        aoi_analysis,
        metadata,
        manual_gsd_m=manual_gsd_m,
    )

    status.info("Rendering detection evidence…")
    progress.progress(92)

    overlay = render_crown_refinement_overlay(
        raster_image,
        crown_results,
    )

    elapsed = time.perf_counter() - started
    progress.progress(100)
    status.success(f"Analysis complete · {elapsed:.1f}s")

except DeepForestDetectionError as error:
    cleanup_temp_file(raster_temp_path)
    st.error(f"Model inference failed: {error}")
    st.info(
        "The application could not safely produce a result. "
        "No fabricated metrics were generated."
    )
    st.stop()
except Exception as error:
    cleanup_temp_file(raster_temp_path)
    st.error(f"Analysis failed: {error}")
    st.info("Review the input image, KML boundary, and measurement scale.")
    st.stop()
finally:
    cleanup_temp_file(raster_temp_path)

# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------
tree_count = len(predictions)
mean_confidence = (
    float(predictions["score"].mean())
    if tree_count
    else 0.0
)
median_confidence = (
    float(predictions["score"].median())
    if tree_count
    else 0.0
)
scale_label, scale_source = measurement_label(metrics)

st.markdown(
    """
    <div class="fl-section">
        <h2>05 · Analysis results</h2>
        <p>The results below separate model detections, crown evidence, and
        physically calibrated measurements.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    metric_card(
        "Tree candidates",
        f"{tree_count:,}",
        f"Threshold ≥ {threshold:.2f}",
    )
with m2:
    metric_card(
        "Mean confidence",
        f"{mean_confidence:.1%}",
        "Model score, not validated accuracy",
    )
with m3:
    metric_card(
        "Refined crowns",
        f"{metrics.refined_tree_count:,}",
        "Image-derived crown evidence",
    )
with m4:
    metric_card(
        "BBox proxies",
        f"{metrics.bbox_fallback_count:,}",
        "Detection geometry retained as fallback",
    )

st.markdown("<br>", unsafe_allow_html=True)

result_tabs = st.tabs(
    ["Overview", "Detection map", "Spatial metrics", "Reliability", "Data"]
)

# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------
with result_tabs[0]:
    left, right = st.columns([1.65, 1], gap="large")

    with left:
        st.image(
            overlay,
            caption="Detected tree candidates",
            width="stretch",
        )
        st.caption(
            "Bounding boxes indicate DeepForest detections. They are not exact "
            "crown segmentation."
        )

    with right:
        state_text, state_class = confidence_state(mean_confidence)
        st.markdown(
            f'<div class="{state_class}"><strong>{state_text}</strong><br>'
            f'Average model confidence: {mean_confidence:.1%}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="fl-card">
                <strong>Measurement provenance</strong>
                <div class="fl-provenance" style="margin-top:.8rem">
                    <div class="fl-provenance-item">
                        <span>Status</span>
                        <strong>{scale_label}</strong>
                    </div>
                    <div class="fl-provenance-item">
                        <span>Source</span>
                        <strong>{scale_source}</strong>
                    </div>
                    <div class="fl-provenance-item">
                        <span>Spatial reference</span>
                        <strong>{metrics.measurement_status.spatial_reference or 'None'}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        if metrics.physical_metrics_available:
            st.success("Physical metrics are available on a defensible scale.")
        else:
            st.warning(
                "Physical metrics are not available for this run. "
                "Tree detection remains valid as a pixel-space result."
            )

# ---------------------------------------------------------------------
# Detection map
# ---------------------------------------------------------------------
with result_tabs[1]:
    st.image(overlay, width="stretch")
    st.caption(
        "Yellow boxes are DeepForest detections; green outlines are locally refined "
        "RGB footprints; orange outlines are bounding-box proxies. None are exact "
        "crown segmentation, and the visual evidence should be reviewed before decisions."
    )

# ---------------------------------------------------------------------
# Spatial metrics
# ---------------------------------------------------------------------
with result_tabs[2]:
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        metric_card(
            "Trees in AOI",
            f"{metrics.detected_tree_count:,}",
            "Detections included after AOI filtering",
        )
    with s2:
        metric_card(
            "Canopy footprint",
            optional_area(metrics.estimated_canopy_area_m2),
            "Estimate from available crown geometry",
        )
    with s3:
        metric_card(
            "Canopy cover",
            (
                f"{metrics.estimated_canopy_cover_percent:.2f}%"
                if metrics.estimated_canopy_cover_percent is not None
                else "Unavailable"
            ),
            "AOI-relative estimate",
        )
    with s4:
        metric_card(
            "Tree density",
            (
                f"{metrics.detected_tree_density_per_ha:.2f} / ha"
                if metrics.detected_tree_density_per_ha is not None
                else "Unavailable"
            ),
            "Requires defensible physical scale",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="fl-card">
            <strong>Measurement provenance</strong>
            <div class="fl-provenance" style="margin-top:.8rem">
                <div class="fl-provenance-item">
                    <span>Basis</span>
                    <strong>{metrics.measurement_basis}</strong>
                </div>
                <div class="fl-provenance-item">
                    <span>Spatial reference</span>
                    <strong>{metrics.measurement_status.spatial_reference or 'Unavailable'}</strong>
                </div>
                <div class="fl-provenance-item">
                    <span>Metric CRS</span>
                    <strong>{metrics.measurement_status.metric_crs or 'Not used'}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if metrics.measurement_warnings:
        for warning in metrics.measurement_warnings:
            st.warning(warning)

    if kml_file is None:
        st.info(
            "No KML boundary was supplied. The complete raster is the analysis region."
        )
    elif aoi_analysis.overlap_status == "inside":
        st.success("KML AOI is fully inside the raster footprint.")
    elif aoi_analysis.overlap_status == "partial_overlap":
        st.warning(
            "KML AOI partially overlaps the raster. Only the overlapping region is analyzed."
        )
    elif aoi_analysis.overlap_status == "outside":
        st.error("KML AOI is outside the raster footprint.")
    else:
        st.info("AOI spatial matching status is unavailable.")

    if aoi_analysis.overlap_fraction is not None:
        st.write(f"**AOI/raster overlap:** {aoi_analysis.overlap_fraction:.1%}")

# ---------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------
with result_tabs[3]:
    r1, r2 = st.columns([1, 1], gap="large")

    with r1:
        metric_card(
            "Mean confidence",
            f"{mean_confidence:.1%}",
            "Average score of reported detections",
        )
        st.markdown("<br>", unsafe_allow_html=True)
        metric_card(
            "Median confidence",
            f"{median_confidence:.1%}",
            "Less sensitive to extreme scores",
        )

    with r2:
        refinement_total = (
            metrics.refined_tree_count
            + metrics.bbox_fallback_count
            + metrics.failed_tree_count
        )
        refinement_rate = (
            metrics.refined_tree_count / refinement_total
            if refinement_total
            else 0.0
        )
        metric_card(
            "Refinement success",
            f"{refinement_rate:.1%}",
            "Share producing image-derived crown geometry",
        )
        st.markdown("<br>", unsafe_allow_html=True)
        metric_card(
            "Failed refinements",
            f"{metrics.failed_tree_count:,}",
            "Detections excluded from usable crown geometry",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if tree_count == 0:
        st.error(
            "No detections passed the selected threshold. Lowering the threshold "
            "may increase candidates, but can also increase false positives."
        )
    elif mean_confidence < 0.50:
        st.warning(
            "Average model confidence is low. Review the detection overlay carefully."
        )
    elif mean_confidence < 0.70:
        st.warning(
            "Average model confidence is moderate. Visual review is recommended."
        )
    else:
        st.success(
            "The selected detections have relatively strong model confidence scores."
        )

    st.caption(
        "Confidence is a model score, not a field-validated accuracy percentage. "
        "Validated accuracy requires labelled ground truth."
    )

    st.markdown("### Threshold sensitivity")
    st.write(
        "The same raw predictions are reused at different confidence thresholds. "
        "This shows how stable the reported tree count is to the chosen cutoff."
    )
    try:
        render_sensitivity_chart(raw_predictions)
    except Exception as error:
        st.warning(f"Sensitivity analysis could not be displayed: {error}")

    st.markdown("### Crown evidence")
    st.write(
        f"**{metrics.refined_tree_count:,}** detections produced image-derived "
        f"crown geometry, while **{metrics.bbox_fallback_count:,}** remained as "
        "bounding-box proxies. Proxy geometry must not be interpreted as exact crown segmentation."
    )

    for warning in generate_warnings(
        quality_assessment,
        calculate_refinement_summary(crown_results),
    ):
        st.warning(warning.message)

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
with result_tabs[4]:
    if tree_count:
        display_columns = [
            "tree_id",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "label",
            "score",
        ]
        available = [c for c in display_columns if c in predictions.columns]
        table = predictions[available].copy()
        if "score" in table:
            table["score"] = table["score"].round(4)

        st.dataframe(
            table,
            hide_index=True,
        )

        csv_data = predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download detection CSV",
            data=csv_data,
            file_name="forestlens_detections.csv",
            mime="text/csv",
        )
    else:
        st.info("No detection records are available for this threshold.")

# ---------------------------------------------------------------------
# Interpretation / methodology
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="fl-section">
        <h2>06 · Interpretation & methodology</h2>
        <p>ForestLens deliberately separates what the model detects from what
        can be physically measured.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

i1, i2 = st.columns(2, gap="large")

with i1:
    st.markdown(
        """
        <div class="fl-card">
            <strong>What the detection count means</strong>
            <p style="color:#617067">
            The reported count is the number of DeepForest tree candidates that
            pass the selected confidence threshold. It is not presented as
            field-verified ground truth.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with i2:
    st.markdown(
        """
        <div class="fl-card">
            <strong>What physical measurements mean</strong>
            <p style="color:#617067">
            Canopy area, cover and hectare-based density are reported only when
            a defensible raster scale or explicitly supplied GSD is available.
            The application does not fabricate square metres from an unscaled image.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if metrics.measurement_warnings:
    with st.expander("Measurement warnings", expanded=False):
        for warning in metrics.measurement_warnings:
            st.write(f"• {warning}")

st.markdown(
    """
    <div class="fl-footer">
        <span><strong>ForestLens AI</strong> · AI-assisted forest intelligence</span>
        <span>DeepForest detections · transparent measurement provenance · visual review recommended</span>
    </div>
    """,
    unsafe_allow_html=True,
)
