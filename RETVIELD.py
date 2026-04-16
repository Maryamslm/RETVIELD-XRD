"""
app.py
Streamlit application entry point for XRD Pattern Visualization & Phase Matching.
"""
import streamlit as st
import pandas as pd
import numpy as np
from data_loader import load_observed_data, load_phase_data, generate_dummy_observed, generate_dummy_phases
from peak_detection import detect_peaks
from phase_matching import match_phases, get_phase_tracks
from plotting import build_xrd_figure

st.set_page_config(page_title="XRD Analyzer", page_icon="🔬", layout="wide")

# Session state initialization
for key in ["obs_data", "calc_data", "phase_refs", "peak_matches", "phase_tracks", "overlap_data"]:
    if key not in st.session_state:
        st.session_state[key] = None

with st.sidebar:
    st.title("⚙️ XRD Controls")
    
    st.subheader("📁 Data Input")
    use_dummy = st.checkbox("Load Dummy Sample Data", value=False)
    
    if use_dummy:
        st.session_state.obs_data = generate_dummy_observed()
        st.session_state.phase_refs = generate_dummy_phases()
        st.success("✅ Dummy data loaded")
    else:
        obs_file = st.file_uploader("Upload Observed Data (CSV)", type=["csv"])
        calc_file = st.file_uploader("Upload Calculated Pattern (CSV, optional)", type=["csv"])
        phase_file = st.file_uploader("Upload Phase References (CSV)", type=["csv"])
        
        if obs_file is not None:
            try:
                st.session_state.obs_data = load_observed_data(obs_file)
                st.success("✅ Observed data loaded")
            except Exception as e: st.error(f"❌ {e}")
                
        if calc_file is not None:
            try:
                st.session_state.calc_data = load_observed_data(calc_file)
                st.success("✅ Calculated data loaded")
            except Exception as e: st.error(f"❌ {e}")
                
        if phase_file is not None:
            try:
                st.session_state.phase_refs = load_phase_data(phase_file)
                st.success("✅ Phase references loaded")
            except Exception as e: st.error(f"❌ {e}")

    st.subheader("🔍 Peak Detection")
    prominence = st.slider("Peak Prominence", 10.0, 500.0, 50.0, 10.0)
    distance = st.slider("Min Peak Distance (pts)", 1, 50, 10)
    
    st.subheader("🧩 Phase Matching")
    tolerance = st.slider("Overlap Tolerance (°)", 0.05, 0.5, 0.15, 0.01)
    show_overlaps = st.checkbox("Highlight Overlapped Peaks", value=True)
    
    st.subheader("🎨 Appearance")
    theme = st.selectbox("Theme", ["light", "dark"], index=0)
    font_size = st.slider("Label Font Size", 10, 18, 14)
    show_labels = st.checkbox("Show Phase Labels", value=True)
    
    run_btn = st.button("🔄 Generate Plot", type="primary", use_container_width=True)

st.title("🔬 XRD Pattern Visualization & Phase Matching")

if run_btn or st.session_state.obs_data is not None:
    obs = st.session_state.obs_data
    calc = st.session_state.calc_data
    refs = st.session_state.phase_refs
    
    if obs is not None:
        # 1. Detect Peaks
        peaks = detect_peaks(obs["2theta"].values, obs["intensity"].values, prominence, distance)
        
        # 2. Match Phases & Find Overlaps
        if refs is not None:
            matches = match_phases(peaks, refs, tolerance)
            tracks = get_phase_tracks(refs, obs["2theta"].min(), obs["2theta"].max())
            overlaps = matches[matches["is_overlapped"]] if show_overlaps else None
            
            st.session_state.peak_matches = matches
            st.session_state.phase_tracks = tracks
            st.session_state.overlap_data = overlaps
        else:
            matches = pd.DataFrame()
            tracks = {}
            overlaps = None
            
        # 3. Calculate Residuals
        residuals = None
        if calc is not None and not calc.empty:
            calc_interp = np.interp(obs["2theta"], calc["2theta"], calc["intensity"])
            residuals = obs["intensity"] - calc_interp
            
        # 4. Build Plot
        fig = build_xrd_figure(
            obs_data=obs, calc_data=calc, residuals=residuals,
            phase_tracks=st.session_state.phase_tracks,
            overlap_data=st.session_state.overlap_data,
            theme=theme, show_labels=show_labels,
            show_overlaps=show_overlaps, font_size=font_size
        )
        
        st.plotly_chart(fig, use_container_width=True, key="main_plot")
        
        # 5. Export Controls
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("⬇️ Export Observed CSV", obs.to_csv(index=False), "observed.csv", "text/csv")
        with c2:
            if not matches.empty:
                st.download_button("⬇️ Export Matches CSV", matches.to_csv(index=False), "matches.csv", "text/csv")
        with c3:
            try:
                img_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
                st.download_button("⬇️ Export Plot PNG", img_bytes, "xrd_plot.png", "image/png")
            except Exception:
                st.caption("⚠️ Install `kaleido` for PNG export: `pip install kaleido`")
            
        if not matches.empty:
            with st.expander("📊 Detected Peaks & Phase Matches"):
                st.dataframe(matches, use_container_width=True)
    else:
        st.info("👈 Upload data or enable dummy data in the sidebar to begin.")
else:
    st.info("👈 Configure parameters and click **Generate Plot**.")
