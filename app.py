"""GeoVision AI - Streamlit app.   Run:  streamlit run app.py"""
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from geovision.config import CLASS_NAMES, PRESET_WINDOWS
from geovision.pipeline import export_geotiff, fetch_pair, run_analysis, synthetic_pair
from geovision.viz import make_report, overlay, rgb

st.set_page_config(page_title="GeoVision AI", page_icon="🛰️", layout="wide")
st.title("🛰️ GeoVision AI")
st.caption("Explainable satellite change detection - Sentinel-2 streamed from open STAC catalogs, no local copy.")

CKPT = Path("checkpoints/geovision.pt")
DEMO_CKPT = Path("checkpoints/demo.pt")


@st.cache_resource
def get_model(path: str):
    from geovision.models import load_model
    return load_model(path, "cpu")


@st.cache_data(show_spinner="Streaming Sentinel-2 scenes (only the pixels needed)...")
def get_pair(bbox, before, after, res, cloud, provider, name):
    return fetch_pair(bbox, before, after, res, cloud, provider, name)


with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Stream Sentinel-2 (STAC)", "Offline synthetic demo"])
    if source.startswith("Stream"):
        from geovision.data.stac_loader import PRESETS
        choice = st.selectbox("Area", list(PRESETS) + ["custom bbox"],
                              format_func=lambda k: f"{k} - {PRESETS[k].description}" if k in PRESETS else k)
        if choice in PRESETS:
            bbox = PRESETS[choice].bbox
            d_before, d_after, d_cloud = PRESET_WINDOWS[choice]
        else:
            c = st.columns(2)
            bbox = (c[0].number_input("West", value=77.62, format="%.4f"), c[1].number_input("South", value=12.86, format="%.4f"),
                    c[0].number_input("East", value=77.74, format="%.4f"), c[1].number_input("North", value=12.97, format="%.4f"))
            d_before, d_after, d_cloud = "2019-01-01/2019-03-31", "2026-01-01/2026-03-31", 30
        before = st.text_input("Before window", d_before)
        after = st.text_input("After window", d_after)
        cloud = st.slider("Max cloud cover %", 5, 90, int(d_cloud))
        res = st.select_slider("Resolution (m/pixel)", [10, 20, 40, 60], value=20)
        provider = st.selectbox("Catalog", ["earth-search", "planetary"])
    else:
        seed = st.number_input("Scene seed", value=123, step=1)

    st.header("Model")
    ckpts = [p for p in (CKPT, DEMO_CKPT) if p.exists()]
    method = st.radio("Method", ["Siamese U-Net", "Rule-based (indices)"], index=0 if ckpts else 1)
    ckpt = st.selectbox("Checkpoint", [str(p) for p in ckpts]) if ckpts else None
    explain = st.checkbox("Grad-CAM explanation", value=True)
    if method == "Siamese U-Net" and not ckpts:
        st.warning("No checkpoint found. Run `python -m geovision demo` or `python -m geovision train`. Using rules.")
    run = st.button("Run analysis", type="primary", width="stretch")

if run:
    try:
        pair = (get_pair(tuple(bbox), before, after, res, cloud, provider, choice)
                if source.startswith("Stream") else synthetic_pair(256, int(seed)))
    except Exception as e:
        st.error(f"Could not load imagery: {type(e).__name__}: {e}")
        st.stop()
    use_model = method == "Siamese U-Net" and ckpt is not None
    with st.spinner("Detecting change..."):
        res_ = run_analysis(pair, "model" if use_model else "rules", get_model(ckpt) if use_model else None,
                            explain=explain and use_model)
    st.session_state["out"] = (pair, res_)

if "out" in st.session_state:
    pair, res_ = st.session_state["out"]
    ins = res_.insights
    st.subheader(ins["headline"])
    cols = st.columns(4)
    for col, e in zip(cols, ins["classes"]):
        col.metric(e["name"], f"{e['hectares']:.1f} ha", f"{e['pct_of_aoi']:.1f}% of area", delta_color="off")

    t1, t2, t3, t4 = st.tabs(["Change map", "Explainability", "Insights", "Export"])
    with t1:
        c = st.columns(3)
        c[0].image(rgb(pair.before), caption="Before", width="stretch")
        c[1].image(rgb(pair.after), caption="After", width="stretch")
        c[2].image(overlay(rgb(pair.after) * 0.8, res_.change_map), caption="Detected change", width="stretch")
        st.caption("  ".join(f"**{n}**" for n in CLASS_NAMES[1:]) + " - red / orange / blue / purple")
        if pair.label is not None:
            from geovision.train import mean_iou_changed
            st.info(f"Synthetic ground truth available - mean IoU over change classes: "
                    f"{mean_iou_changed(res_.change_map, pair.label):.3f}")
    with t2:
        if res_.cam is not None:
            import matplotlib.cm as cm
            heat = cm.jet(res_.cam)[..., :3]
            st.image(np.clip(0.45 * rgb(pair.after) + 0.55 * heat, 0, 1), caption="Grad-CAM: regions driving the prediction",
                     width="stretch")
        else:
            st.info("Enable Grad-CAM and use the Siamese U-Net to see where the model looked.")
        st.markdown("**Spectral evidence per class (mean change, after - before)**")
        st.dataframe([{"Class": e["name"], **{k.upper(): v for k, v in e["mean_delta"].items()},
                       "Confidence": e["confidence"]} for e in ins["classes"]], width="stretch")
    with t3:
        for line in ins["narrative"] or ["No significant change detected."]:
            st.markdown(f"- {line}")
        st.dataframe([{k: v for k, v in e.items() if k != "mean_delta"} for e in ins["classes"]], width="stretch")
    with t4:
        fig = make_report(pair, res_)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130)
        st.download_button("Download report (PNG)", buf.getvalue(), "geovision_report.png", "image/png")
        st.download_button("Download insights (JSON)", json.dumps(ins, indent=2), "insights.json", "application/json")
        try:
            with tempfile.TemporaryDirectory() as d:
                export_geotiff(f"{d}/change.tif", res_.change_map, pair)
                st.download_button("Download change map (GeoTIFF)", Path(f"{d}/change.tif").read_bytes(),
                                   "change.tif", "image/tiff")
        except Exception:
            st.caption("GeoTIFF export is available for streamed (georeferenced) scenes.")
else:
    st.info("Pick a source in the sidebar and press **Run analysis**. "
            "Choose *Offline synthetic demo* to try the app without internet.")
