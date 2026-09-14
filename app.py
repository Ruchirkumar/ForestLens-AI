from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image

from src.detection.model import (
    DeepForestDetectionError,
    filter_detections,
    load_model,
    predict_detections,
)
from src.visualization.overlay import render_detection_overlay


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ForestLens AI",
    page_icon="🌲",
    layout="wide",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

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

    .info-card {
        padding: 1rem 1.2rem;
        border: 1px solid rgba(128,128,128,0.20);
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL CACHE
# ============================================================

@st.cache_resource(show_spinner=False)
def get_model():
    """Load and cache the DeepForest model."""
    return load_model()


# ============================================================
# HEADER
# ============================================================

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


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Analysis Settings")

    threshold = st.slider(
        "Detection confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05,
        help=(
            "Only detections with a model confidence score at or above "
            "this value will be displayed."
        ),
    )

    st.caption(
        "Higher thresholds reduce lower-confidence detections. "
        "Lower thresholds may detect more trees but can increase false positives."
    )

    st.divider()

    st.subheader("About ForestLens")

    st.write(
        "ForestLens AI analyzes forest imagery using a pretrained "
        "DeepForest tree detector."
    )

    st.info(
        "GeoTIFF imagery is recommended when physical area measurements "
        "are required because geospatial metadata can provide pixel size "
        "and coordinate reference information."
    )


# ============================================================
# IMAGE INPUT
# ============================================================

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


# ============================================================
# OPTIONAL KML
# ============================================================

st.subheader("2. Optional Area of Interest")

kml_file = st.file_uploader(
    "Upload KML boundary",
    type=["kml"],
    help="Upload a Polygon or MultiPolygon boundary.",
)

if kml_file is not None:
    st.info(
        "KML boundary uploaded. The current interface acknowledges the AOI; "
        "full AOI filtering will be connected to the geospatial analysis "
        "pipeline separately."
    )


# ============================================================
# NO IMAGE STATE
# ============================================================

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

        **Image → DeepForest detection → Confidence filtering → "
        "Detection visualization → Results table → CSV export**
        """
    )

    st.stop()


# ============================================================
# IMAGE READING
# ============================================================

try:

    image = Image.open(uploaded_file)

    if image.mode not in {"RGB", "RGBA"}:
        st.error(
            f"Unsupported image mode: `{image.mode}`. "
            "Please upload an RGB or RGBA image."
        )
        st.stop()

    # Convert RGBA to RGB for the detector.
    preview_image = image.convert("RGB")

except Exception as error:

    st.error(
        f"Could not read the uploaded image: {error}"
    )
    st.stop()


# ============================================================
# IMAGE INFORMATION
# ============================================================

st.subheader("3. Image Preview")

preview_col, info_col = st.columns([2.2, 1])

with preview_col:

    st.image(
        preview_image,
        caption=uploaded_file.name,
        use_container_width=True,
    )

with info_col:

    st.metric(
        "Width",
        f"{preview_image.width:,} px",
    )

    st.metric(
        "Height",
        f"{preview_image.height:,} px",
    )

    st.metric(
        "Channels",
        "3 RGB",
    )

    if uploaded_file.size:
        st.metric(
            "File size",
            f"{uploaded_file.size / (1024 * 1024):.2f} MB",
        )


# ============================================================
# RUN ANALYSIS
# ============================================================

st.divider()

run_analysis = st.button(
    "Run Forest Analysis",
    type="primary",
    use_container_width=True,
)


if not run_analysis:
    st.caption(
        "Adjust the detection threshold if required, then click "
        "**Run Forest Analysis**."
    )
    st.stop()


# ============================================================
# ANALYSIS
# ============================================================

progress = st.progress(0)
status = st.empty()

try:

    # --------------------------------------------------------
    # Step 1 — Load model
    # --------------------------------------------------------

    status.info("Loading DeepForest model...")
    progress.progress(15)

    model = get_model()

    # --------------------------------------------------------
    # Step 2 — Run detection
    # --------------------------------------------------------

    status.info("Analyzing forest imagery...")
    progress.progress(30)

    raw_predictions = predict_detections(
        model,
        preview_image,
    )

    progress.progress(60)

    # --------------------------------------------------------
    # Step 3 — Filter detections
    # --------------------------------------------------------

    status.info("Filtering tree detections...")

    predictions = filter_detections(
        raw_predictions,
        score_threshold=threshold,
    )

    progress.progress(75)

    # --------------------------------------------------------
    # Step 4 — Generate evidence overlay
    # --------------------------------------------------------

    status.info("Generating detection visualization...")

    overlay = render_detection_overlay(
        preview_image,
        predictions,
        show_ids=True,
        show_scores=True,
    )

    progress.progress(100)

    status.success("Forest analysis completed.")

except DeepForestDetectionError as error:

    progress.empty()
    status.empty()

    st.error(
        f"Forest analysis failed: {error}"
    )

    st.info(
        "The image could not be processed by the DeepForest detection pipeline."
    )

    st.stop()

except Exception as error:

    progress.empty()
    status.empty()

    st.error(
        "An unexpected error occurred during analysis."
    )

    st.exception(error)

    st.stop()


# ============================================================
# RESULTS SUMMARY
# ============================================================

st.divider()

st.subheader("Analysis Summary")


tree_count = len(predictions)


if tree_count > 0:

    mean_confidence = float(
        predictions["score"].mean()
    )

    max_confidence = float(
        predictions["score"].max()
    )

    min_confidence = float(
        predictions["score"].min()
    )

else:

    mean_confidence = 0.0
    max_confidence = 0.0
    min_confidence = 0.0


summary_1, summary_2, summary_3, summary_4 = st.columns(4)


with summary_1:

    st.metric(
        "Detected Trees",
        f"{tree_count:,}",
    )


with summary_2:

    st.metric(
        "Mean Confidence",
        f"{mean_confidence:.1%}",
    )


with summary_3:

    st.metric(
        "Highest Confidence",
        f"{max_confidence:.1%}",
    )


with summary_4:

    st.metric(
        "Detection Threshold",
        f"{threshold:.0%}",
    )


# ============================================================
# DETECTION VISUALIZATION
# ============================================================

st.subheader("Detected Tree Crowns")

st.image(
    overlay,
    use_container_width=True,
)

st.caption(
    "Red bounding boxes represent DeepForest tree detections. "
    "They are detection boxes, not exact crown segmentation boundaries."
)


# ============================================================
# RELIABILITY
# ============================================================

st.subheader("Detection Reliability")


if tree_count == 0:

    st.warning(
        "No trees passed the selected confidence threshold. "
        "Try a lower threshold or a different image."
    )

elif mean_confidence < 0.50:

    st.warning(
        "The average model confidence is relatively low. "
        "Treat the detected tree count as an estimate and review "
        "the visual detections."
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
    "Important: model confidence is not the same as validated detection accuracy."
)


# ============================================================
# DETECTION DATA
# ============================================================

st.subheader("Detection Data")


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

    table = predictions[available_columns].copy()

    if "score" in table.columns:

        table["score"] = table["score"].round(4)

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # CSV EXPORT
    # --------------------------------------------------------

    csv_data = predictions.to_csv(
        index=False
    ).encode("utf-8")

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


# ============================================================
# INTERPRETATION / LIMITATIONS
# ============================================================

st.subheader("Important Interpretation")


st.warning(
    "The current DeepForest result represents detected tree bounding boxes. "
    "It should not be interpreted as exact crown segmentation or exact canopy area."
)


st.write(
    "ForestLens currently focuses on reliable presentation of the detector's "
    "evidence. Physical canopy-area estimation should only be reported when "
    "valid geospatial metadata and a defensible crown-area method are available."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ForestLens AI • Tree detection from forest imagery • "
    "Results should be visually reviewed before operational decisions."
)