"""
xrd_analyzer_app.py

A complete Streamlit application for XRD pattern visualization, peak detection, 
and phase matching.

Usage:
    streamlit run xrd_analyzer_app.py

Dependencies:
    pip install streamlit pandas numpy scipy plotly kaleido
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import os
from io import BytesIO

# ═══════════════════════════════════════════════════════════════════
# MODULE 1: DATA LOADER
# Handles CSV loading and dummy data generation
# ═══════════════════════════════════════════════════════════════════

def load_observed_data(filepath: str) -> pd.DataFrame:
    """Load observed XRD data from CSV. Expects columns: 2theta, intensity"""
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower()
        
        # Rename standard variations
        rename_map = {"2θ": "2theta", "counts": "intensity", "i": "intensity"}
        df = df.rename(columns=rename_map)
        
        if "2theta" not in df.columns:
            raise ValueError("CSV must contain a '2theta' or '2θ' column.")
        if "intensity" not in df.columns:
            raise ValueError("CSV must contain an 'intensity' or 'counts' column.")
            
        df = df[["2theta", "intensity"]].sort_values("2theta").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading observed data: {e}")
        return pd.DataFrame()

def load_phase_data(filepath: str) -> pd.DataFrame:
    """Load phase reference data from CSV. Expects: phase, 2theta, hkl"""
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower()
        
        required_cols = {"phase", "2theta", "hkl"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Phase CSV must contain columns: {required_cols}")
            
        df = df.sort_values("2theta").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading phase data: {e}")
        return pd.DataFrame()

def generate_dummy_observed() -> pd.DataFrame:
    """Generate dummy observed XRD pattern for testing."""
    np.random.seed(42)
    tt = np.linspace(10, 80, 1400)
    # Define peaks: position, intensity, width (sigma)
    peaks_pos = [25.2, 35.1, 37.8, 40.5, 52.3, 65.0, 70.2]
    intensities = [1000, 800, 1200, 600, 450, 300, 500]
    widths = [0.15, 0.18, 0.16, 0.14, 0.2, 0.19, 0.17]
    
    signal = np.zeros_like(tt)
    for pos, amp, w in zip(peaks_pos, intensities, widths):
        signal += amp * np.exp(-((tt - pos)**2) / (2 * w**2))
        
    # Add background and noise
    bg = 50 + 0.5 * tt
    I = signal + bg + np.random.normal(0, 10, len(tt))
    return pd.DataFrame({"2theta": tt, "intensity": np.maximum(I, 0)})

def generate_dummy_phases() -> pd.DataFrame:
    """Generate dummy phase reference data with intentional overlaps."""
    return pd.DataFrame({
        "phase": ["α-Co"]*3 + ["Cr23C6"]*3 + ["σ-CoCr"]*2,
        "2theta": [25.2, 35.1, 65.0, 37.8, 40.5, 70.2, 35.15, 52.3],
        "hkl": ["(100)", "(002)", "(110)", "(111)", "(200)", "(311)", "(100)", "(210)"]
    })

# ═══════════════════════════════════════════════════════════════════
# MODULE 2: PEAK DETECTION
# Detects peaks in XRD intensity data using scipy.signal.find_peaks
# ═══════════════════════════════════════════════════════════════════

def detect_peaks(tt: np.ndarray, intensity: np.ndarray, prominence: float = 50, distance: int = 10) -> pd.DataFrame:
    """
    Detect peaks in the XRD pattern.
    
    Args:
        tt: 2theta array
        intensity: Intensity array
        prominence: Minimum peak prominence
        distance: Minimum distance between peaks in data points
        
    Returns:
        DataFrame with columns: 2theta, intensity, index
    """
    peaks_idx, _ = find_peaks(intensity, prominence=prominence, distance=distance)
    return pd.DataFrame({
        "2theta": tt[peaks_idx],
        "intensity": intensity[peaks_idx],
        "index": peaks_idx
    })

# ═══════════════════════════════════════════════════════════════════
# MODULE 3: PHASE MATCHING
# Matches detected peaks to reference phases and identifies overlaps
# ═══════════════════════════════════════════════════════════════════

def match_phases(detected_peaks: pd.DataFrame, phase_refs: pd.DataFrame, tolerance: float = 0.15) -> pd.DataFrame:
    """
    Match detected peaks to reference phase peaks within a tolerance.
    Identifies overlapping peaks where multiple phases match the same detected peak.
    """
    if detected_peaks.empty or phase_refs.empty:
        return pd.DataFrame()

    matches = []
    for _, peak in detected_peaks.iterrows():
        diff = np.abs(phase_refs["2theta"] - peak["2theta"])
        mask = diff <= tolerance
        matched_refs = phase_refs[mask]
        
        is_overlapped = len(matched_refs) > 1
        phase_names = ", ".join(matched_refs["phase"].unique()) if len(matched_refs) > 0 else "Unmatched"
        hkl_list = ", ".join(matched_refs["hkl"].unique()) if len(matched_refs) > 0 else "-"
        
        matches.append({
            "2theta_obs": peak["2theta"],
            "intensity": peak["intensity"],
            "phases": phase_names,
            "hkls": hkl_list,
            "is_overlapped": is_overlapped,
            "n_matching_phases": len(matched_refs)
        })
    return pd.DataFrame(matches)

def get_phase_tracks(phase_refs: pd.DataFrame, min_2theta: float, max_2theta: float) -> dict:
    """Organize phase reference peaks into separate tracks for plotting."""
    tracks = {}
    for phase in phase_refs["phase"].unique():
        phase_data = phase_refs[phase_refs["phase"] == phase]
        tracks[phase] = phase_data[(phase_data["2theta"] >= min_2theta) & (phase_data["2theta"] <= max_2theta)]
    return tracks

# ═══════════════════════════════════════════════════════════════════
# MODULE 4: PLOTTING
# Generates professional Plotly figures for XRD analysis
# Layout: Top (Obs/Calc) -> Middle (Residual) -> Bottom (Phase Sticks + Overlaps)
# ═══════════════════════════════════════════════════════════════════

def build_xrd_figure(
    obs_df: pd.DataFrame,
    calc_df: pd.DataFrame | None,
    residuals: np.ndarray | None,
    phase_tracks: dict,
    overlap_df: pd.DataFrame | None,
    theme: str = "light",
    show_labels: bool = True,
    show_overlaps: bool = True,
    font_size: int = 14,
    marker_height: float = 1.0
) -> go.Figure:
    """
    Builds a clean, publication-ready 3-panel XRD figure.
    """
    has_residuals = residuals is not None and len(residuals) > 0
    n_rows = 3 if has_residuals else 2
    heights = [0.6, 0.15, 0.25] if has_residuals else [0.7, 0.3]
    
    fig = make_subplots(rows=n_rows, cols=1, row_heights=heights, shared_xaxes=True, vertical_spacing=0.05)
    
    # Theme configs
    is_dark = theme == "dark"
    bg_color = "#111827" if is_dark else "#ffffff"
    grid_color = "#374151" if is_dark else "#e5e7eb"
    text_color = "#e5e7eb" if is_dark else "#111827"
    
    # 1. Top Panel: Observed + Calculated
    fig.add_trace(go.Scatter(
        x=obs_df["2theta"], y=obs_df["intensity"],
        mode="lines", name="Observed", line=dict(color="#2563eb", width=2),
        hovertemplate="<b>2θ:</b> %{x:.3f}°<br><b>Intensity:</b> %{y:.1f}<extra></extra>"
    ), row=1, col=1)
    
    if calc_df is not None and not calc_df.empty:
        fig.add_trace(go.Scatter(
            x=calc_df["2theta"], y=calc_df["intensity"],
            mode="lines", name="Calculated", line=dict(color="#10b981", width=1.5, dash="dash"),
            hovertemplate="<b>2θ:</b> %{x:.3f}°<br><b>Intensity:</b> %{y:.1f}<extra></extra>"
        ), row=1, col=1)
        
    # 2. Middle Panel: Residuals
    if has_residuals:
        fig.add_trace(go.Scatter(
            x=obs_df["2theta"], y=residuals,
            mode="lines", name="Residual", fill="tozeroy",
            line=dict(color="#8b5cf6", width=1),
            hovertemplate="<b>2θ:</b> %{x:.3f}°<br><b>ΔI:</b> %{y:.1f}<extra></extra>"
        ), row=2, col=1)
        
    # 3. Bottom Panel: Phase Tracks & Overlaps
    track_row = 3 if has_residuals else 2
    n_tracks = len(phase_tracks)
    y_positions = np.linspace(0.2, 0.8 * marker_height, n_tracks) if n_tracks > 0 else []
    colors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6", "#6366f1"]
    
    for i, (phase, df) in enumerate(phase_tracks.items()):
        if df.empty: continue
        y_val = y_positions[i] if i < len(y_positions) else 0.5
        color = colors[i % len(colors)]
        
        # Stick markers (Triangle Up)
        fig.add_trace(go.Scatter(
            x=df["2theta"], y=[y_val]*len(df),
            mode="markers", marker=dict(symbol="triangle-up", size=10, color=color),
            name=phase, showlegend=True,
            hovertemplate=f"<b>Phase:</b> {phase}<br><b>2θ:</b> %{{x:.3f}}°<br><b>hkl:</b> %{{customdata}}<extra></extra>",
            customdata=df["hkl"]
        ), row=track_row, col=1)
        
        # Labels (skip if too dense to avoid clutter)
        if show_labels and len(df) < 15:
            for _, row_data in df.iterrows():
                fig.add_annotation(
                    x=row_data["2theta"], y=y_val + 0.12 * marker_height,
                    text=f"{phase} ({row_data['hkl']})",
                    showarrow=False, font=dict(size=font_size-2, color=color),
                    xanchor="center", yanchor="bottom"
                )
                
    # Overlap Track (Red 'X' markers)
    if show_overlaps and overlap_df is not None and not overlap_df.empty:
        fig.add_trace(go.Scatter(
            x=overlap_df["2theta_obs"], y=[1.0 * marker_height]*len(overlap_df),
            mode="markers", marker=dict(symbol="x", size=12, color="#dc2626", line=dict(width=2)),
            name="Overlapped Peaks", showlegend=True,
            hovertemplate="<b>Overlap:</b> %{customdata}<br><b>2θ:</b> %{x:.3f}°<extra></extra>",
            customdata=overlap_df["phases"]
        ), row=track_row, col=1)
        
    # Layout & Styling
    fig.update_layout(
        template="plotly_white" if not is_dark else "plotly_dark",
        paper_bgcolor=bg_color, plot_bgcolor=bg_color,
        font=dict(family="Inter, system-ui, -apple-system, sans-serif", size=font_size, color=text_color),
        height=700, margin=dict(l=70, r=40, t=40, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor=grid_color),
        xaxis2=dict(showgrid=True, gridcolor=grid_color) if has_residuals else None,
        xaxis3=dict(showgrid=True, gridcolor=grid_color) if n_rows == 3 else None
    )
    
    # Axis formatting
    fig.update_yaxes(title_text="Intensity (a.u.)", gridcolor=grid_color, zerolinecolor=grid_color, row=1, col=1)
    fig.update_xaxes(title_text="2θ (degrees)", gridcolor=grid_color, zerolinecolor=grid_color, row=n_rows, col=1)
    
    if has_residuals:
        fig.update_yaxes(title_text="Residual (ΔI)", gridcolor=grid_color, row=2, col=1)
        
    # Hide y-axis ticks/labels for phase track row to keep it clean
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=track_row, col=1)
        
    return fig

# ═══════════════════════════════════════════════════════════════════
# MODULE 5: APP ENTRY POINT
# Streamlit UI, state management, and main logic
# ═══════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="XRD Analyzer Pro", page_icon="🔬", layout="wide")

    # Session state initialization
    for key in ["obs_data", "calc_data", "phase_refs", "peak_matches", "phase_tracks", "overlap_data"]:
        if key not in st.session_state:
            st.session_state[key] = None

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Controls")
        
        st.subheader("📁 Data Input")
        use_dummy = st.checkbox("Load Dummy Sample Data", value=False)
        
        if use_dummy:
            st.session_state.obs_data = generate_dummy_observed()
            st.session_state.phase_refs = generate_dummy_phases()
            st.session_state.calc_data = pd.DataFrame() # No dummy calc data for now
            st.success("✅ Dummy data loaded")
        else:
            obs_file = st.file_uploader("Upload Observed Data (CSV)", type=["csv"])
            calc_file = st.file_uploader("Upload Calculated Pattern (CSV, optional)", type=["csv"])
            phase_file = st.file_uploader("Upload Phase References (CSV)", type=["csv"])
            
            if obs_file is not None:
                st.session_state.obs_data = load_observed_data(obs_file)
                if not st.session_state.obs_data.empty: st.success("✅ Observed data loaded")
                
            if calc_file is not None:
                st.session_state.calc_data = load_observed_data(calc_file)
                if not st.session_state.calc_data.empty: st.success("✅ Calculated data loaded")
                
            if phase_file is not None:
                st.session_state.phase_refs = load_phase_data(phase_file)
                if not st.session_state.phase_refs.empty: st.success("✅ Phase references loaded")

        st.subheader("🔍 Peak Detection")
        prominence = st.slider("Peak Prominence", 10.0, 500.0, 50.0, 10.0)
        distance = st.slider("Min Peak Distance (pts)", 1, 50, 10)
        
        st.subheader("🧩 Phase Matching")
        tolerance = st.slider("Overlap Tolerance (°)", 0.05, 0.5, 0.15, 0.01)
        show_overlaps = st.checkbox("Highlight Overlapped Peaks", value=True)
        show_labels = st.checkbox("Show Phase Labels", value=True)
        
        st.subheader("🎨 Appearance")
        theme = st.selectbox("Theme", ["light", "dark"], index=0)
        font_size = st.slider("Font Size", 10, 18, 13)
        marker_height = st.slider("Marker Track Height", 0.5, 2.0, 1.0)
        
        run_btn = st.button("🔄 Generate Plot", type="primary", use_container_width=True)

    st.title("🔬 XRD Pattern Visualization & Phase Matching")

    if run_btn or (use_dummy and st.session_state.obs_data is not None and not st.session_state.obs_data.empty):
        obs = st.session_state.obs_data
        calc = st.session_state.calc_data
        refs = st.session_state.phase_refs
        
        if obs is not None and not obs.empty:
            # 1. Detect Peaks
            with st.spinner("Detecting peaks..."):
                peaks = detect_peaks(obs["2theta"].values, obs["intensity"].values, prominence, distance)
            
            # 2. Match Phases & Find Overlaps
            matches = pd.DataFrame()
            tracks = {}
            overlaps = None
            
            if refs is not None and not refs.empty:
                matches = match_phases(peaks, refs, tolerance)
                tracks = get_phase_tracks(refs, obs["2theta"].min(), obs["2theta"].max())
                overlaps = matches[matches["is_overlapped"]] if show_overlaps and not matches.empty else None
            
            # 3. Calculate Residuals
            residuals = None
            if calc is not None and not calc.empty:
                try:
                    calc_interp = np.interp(obs["2theta"], calc["2theta"], calc["intensity"])
                    residuals = obs["intensity"] - calc_interp
                except Exception as e:
                    st.warning(f"Could not calculate residuals: {e}")
            
            # 4. Build Plot
            fig = build_xrd_figure(
                obs_df=obs, 
                calc_df=calc, 
                residuals=residuals,
                phase_tracks=tracks,
                overlap_df=overlaps,
                theme=theme, 
                show_labels=show_labels,
                show_overlaps=show_overlaps, 
                font_size=font_size,
                marker_height=marker_height
            )
            
            st.plotly_chart(fig, use_container_width=True, key="main_plot")
            
            # 5. Export Controls
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button("⬇️ Export Observed CSV", obs.to_csv(index=False), "observed.csv", "text/csv")
            with c2:
                if not matches.empty:
                    st.download_button("⬇️ Export Matches CSV", matches.to_csv(index=False), "peak_matches.csv", "text/csv")
            with c3:
                try:
                    # Try using kaleido for PNG export
                    img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
                    st.download_button("⬇️ Export Plot PNG", img_bytes, "xrd_plot.png", "image/png")
                except Exception:
                    st.caption("⚠️ Install `kaleido` (`pip install kaleido`) to enable PNG download, or use Plotly's camera icon to save.")
                
            # Show details
            if not matches.empty:
                with st.expander("📊 Detected Peaks & Phase Matches"):
                    st.dataframe(matches, use_container_width=True)
            else:
                st.info("No phase matches found. Try adjusting the overlap tolerance or uploading phase data.")
        else:
            st.info("👈 Upload data or enable dummy data in the sidebar to begin.")
    else:
        st.info("👈 Configure parameters and click **Generate Plot** or load dummy data.")

if __name__ == "__main__":
    main()
