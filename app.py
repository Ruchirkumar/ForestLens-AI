import streamlit as st

st.set_page_config(
    page_title="ForestLens AI",
    page_icon="🌲",
    layout="wide",
)

st.title("ForestLens AI")
st.caption("Resolution-aware tree crown detection and canopy-area estimation")

st.info(
    "Starter build. First milestone: validate the pretrained detector on the "
    "actual high-resolution forest imagery before wiring the full analysis pipeline."
)

left, right = st.columns(2)

with left:
    st.subheader("1. Upload forest imagery")
    st.file_uploader(
        "GeoTIFF / PNG / JPG",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        help="GeoTIFF is recommended because it preserves spatial metadata.",
    )

with right:
    st.subheader("2. Optional AOI")
    st.file_uploader(
        "KML boundary",
        type=["kml"],
        help="Polygon or MultiPolygon KML.",
    )

st.subheader("3. Analysis settings")
st.slider(
    "Detection threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.30,
    step=0.05,
)

st.button("Run analysis", type="primary", disabled=True)

with st.expander("Why is analysis disabled?"):
    st.write(
        "We are deliberately testing the real model/data path first. "
        "This prevents the UI from hiding model or imagery problems."
    )

st.markdown("---")
st.subheader("Planned results")
c1, c2, c3 = st.columns(3)
c1.metric("Tree crowns", "—")
c2.metric("Estimated canopy area", "—")
c3.metric("Canopy cover", "—")
