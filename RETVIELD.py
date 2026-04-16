"""
╔══════════════════════════════════════════════════════════════════╗
║ Co-Cr Dental Alloy · Full Rietveld XRD Refinement (Improved)    ║
║ Enhanced pattern visualization - clear phases & peaks           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import time
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import least_squares

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title="Co-Cr XRD · Rietveld",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================= GITHUB CONFIG (unchanged) =========================
GITHUB_REPO = "Maryamslm/RETVIELD-XRD"
GITHUB_COMMIT = "e9716f8c3d4654fcba8eddde065d0472b1db69e9"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_COMMIT}/samples/"

AVAILABLE_FILES = { ... }  # Keep your existing AVAILABLE_FILES dictionary

# ========================= THEME (unchanged) =========================
def apply_theme(bg_theme: str, font_size: float, primary_color: str):
    # ... (keep your existing apply_theme function unchanged)
    themes = {
        "Dark Mode": {"bg": "#020617", "text": "#e2e8f0", "sidebar": "#030712", "panel": "#080e1a", "border": "#1e293b"},
        "Light Mode": {"bg": "#f8fafc", "text": "#0f172a", "sidebar": "#ffffff", "panel": "#f1f5f9", "border": "#cbd5e1"},
        "High Contrast": {"bg": "#000000", "text": "#00ff00", "sidebar": "#0a0a0a", "panel": "#111111", "border": "#00ff0044"}
    }
    t = themes.get(bg_theme, themes["Dark Mode"])
    st.markdown(f"""
    <style>
        /* Your existing CSS - unchanged */
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif !important; font-size: {font_size}rem !important; }}
        code, pre {{ font-family: 'IBM Plex Mono', monospace !important; }}
        [data-testid="stAppViewContainer"] > .main {{ background-color: {t['bg']} !important; color: {t['text']} !important; }}
        /* ... rest of your CSS remains the same ... */
    </style>
    """, unsafe_allow_html=True)
    return t['border']

# ========================= PHASE DB, CRYSTALLOGRAPHY, PROFILE FUNCTIONS (unchanged) =========================
# Keep all your existing code for:
# - PHASE_DB, AtomSite, Phase class
# - _build_phase_db()
# - All crystallography functions (_d_cubic, _F2, generate_reflections, etc.)
# - Profile functions (gaussian, lorentzian, pseudo_voigt, caglioti, lp_factor, chebyshev_bg, phase_pattern)
# - RietveldRefiner class
# - Demo & file parsing functions

# ... [Paste all your existing code from PHASE_DB down to the RietveldRefiner class unchanged] ...

# ========================= IMPROVED PATTERN PLOTTING =========================
def create_improved_fit_plot(tt, Iobs, results, refiner, show_hkl_labels, hkl_font_size, hkl_label_offset, hkl_color, bg_theme, border_color, wavelength):
    r = results
    z_shift = float(r.get("z_shift", 0.0))
    _, _, pp_vec = _unpack(refiner.x0, refiner.n_bg, refiner.n_ph)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.78, 0.22],
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("Rietveld Fit", "Difference Plot")
    )

    # Observed data
    fig.add_trace(go.Scatter(
        x=tt, y=Iobs,
        mode="lines",
        name="Observed (I_obs)",
        line=dict(color="#94a3b8", width=1.8)
    ), row=1, col=1)

    # Background
    fig.add_trace(go.Scatter(
        x=tt, y=r["Ibg"],
        mode="lines",
        name="Background",
        line=dict(color="#475569", width=1.2, dash="dot"),
        fill="tozeroy",
        fillcolor="rgba(71,85,105,0.15)"
    ), row=1, col=1)

    # Individual phase contributions (stacked on background)
    for key, Iph in r["contribs"].items():
        ph = PHASE_DB[key]
        wf = r["wf"].get(key, 0) * 100
        fig.add_trace(go.Scatter(
            x=tt,
            y=Iph + r["Ibg"],
            mode="lines",
            name=f"{ph.name} ({wf:.1f} wt%)",
            line=dict(color=ph.color, width=1.7, dash="dash"),
            opacity=0.85
        ), row=1, col=1)

    # Calculated pattern
    fig.add_trace(go.Scatter(
        x=tt, y=r["Icalc"],
        mode="lines",
        name="Calculated (I_calc)",
        line=dict(color="#fbbf24", width=2.4)
    ), row=1, col=1)

    # Difference plot
    fig.add_trace(go.Scatter(
        x=tt, y=r["diff"],
        mode="lines",
        name="Δ (obs - calc)",
        line=dict(color="#818cf8", width=1.4),
        fill="tozeroy",
        fillcolor="rgba(129,140,248,0.18)"
    ), row=2, col=1)

    fig.add_hline(y=0, line=dict(color="#475569", width=1, dash="dash"), row=2, col=1)

    # ==================== PEAK TICKS & LABELS ====================
    if show_hkl_labels:
        y_max = float(Iobs.max())
        y_range = y_max - float(Iobs.min())
        base_label_y = y_max + (y_range * hkl_label_offset / 100)

        for i, ph_obj in enumerate(refiner.phases):
            a_ref, c_ref = float(pp_vec[i][1]), float(pp_vec[i][2])
            ph_ref = _make_refined_phase(ph_obj, a_ref, c_ref)
            pks = generate_reflections(ph_ref, wl=wavelength,
                                       tt_min=float(tt.min()), tt_max=float(tt.max()))

            # Vertical tick marks at peak positions
            y_tick = float(Iobs.min()) - 0.06 * y_range
            fig.add_trace(go.Scatter(
                x=[p["tt"] + z_shift for p in pks],
                y=[y_tick] * len(pks),
                mode="markers",
                marker=dict(symbol="line-ns", size=12, color=ph_obj.color, line=dict(width=2.5, color=ph_obj.color)),
                name=f"{ph_obj.name} peaks",
                showlegend=False
            ), row=1, col=1)

            # (hkl) labels with better spacing
            label_color = ph_obj.color if hkl_color == "phase" else hkl_color
            used_positions = []

            for pk in sorted(pks, key=lambda x: x["tt"]):
                tt_pos = pk["tt"] + z_shift
                # Avoid label overlap by slight vertical staggering
                stagger = 0
                for used in used_positions:
                    if abs(used - tt_pos) < 0.8:
                        stagger += 1.2
                label_y = base_label_y + stagger * (y_range * 0.04)

                hkl_text = f"({pk['h']}{pk['k']}{pk['l']})"

                fig.add_annotation(
                    x=tt_pos,
                    y=label_y,
                    text=hkl_text,
                    showarrow=False,
                    font=dict(size=hkl_font_size, color=label_color, family="IBM Plex Mono"),
                    xanchor="center",
                    yanchor="bottom",
                    bordercolor=border_color,
                    borderwidth=1,
                    borderpad=3,
                    bgcolor="rgba(15,23,42,0.85)" if bg_theme == "Dark Mode" else "rgba(255,255,255,0.85)"
                )
                used_positions.append(tt_pos)

    fig.update_layout(
        template="plotly_dark" if bg_theme == "Dark Mode" else "plotly_white",
        height=680,
        legend=dict(font=dict(size=11), orientation="h", y=1.02, x=0.01),
        margin=dict(l=70, r=40, t=40, b=60),
        font=dict(family="IBM Plex Sans")
    )

    fig.update_xaxes(title_text="2θ (°)", row=2, col=1, showgrid=True, gridwidth=1, gridcolor="#334155")
    fig.update_yaxes(title_text="Intensity (counts)", row=1, col=1, showgrid=True, gridwidth=1, gridcolor="#334155")
    fig.update_yaxes(title_text="Difference", row=2, col=1, showgrid=True, gridwidth=1, gridcolor="#334155")

    return fig

# ========================= MAIN APP (only the plotting part is changed) =========================
# Keep everything else exactly the same until the tab_fit section

with tab_fit:
    if st.session_state["results"] is None:
        # ... your existing placeholder plot ...
        st.info("👈 Select a file and press **▶ Run Rietveld Refinement**.")
    else:
        r, refiner, tt, Iobs, elapsed = (st.session_state["results"], st.session_state["refiner"],
                                         st.session_state["tt"], st.session_state["Iobs"], st.session_state["elapsed"])

        # Quality metrics (unchanged)
        rwp, rp, gof, chi2 = r["Rwp"], r["Rp"], r["GOF"], r["chi2"]
        qc = q_color(rwp)
        st.markdown(f"""<div class="mstrip"> ... your existing metrics ... </div>""", unsafe_allow_html=True)

        # === IMPROVED PLOT ===
        fig = create_improved_fit_plot(
            tt, Iobs, r, refiner,
            show_hkl_labels, hkl_font_size, hkl_label_offset, hkl_color,
            bg_theme, border_color, wavelength
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download CSV (unchanged)
        df_pat = pd.DataFrame({
            "two_theta": tt,
            "I_obs": Iobs,
            "I_calc": r["Icalc"],
            "I_background": r["Ibg"],
            "difference": r["diff"],
            **{f"I_{k}": v for k, v in r["contribs"].items()}
        })
        st.download_button("⬇ Download pattern CSV", data=df_pat.to_csv(index=False),
                           file_name="rietveld_pattern.csv", mime="text/csv")

# ========================= REST OF THE APP (tabs, etc.) =========================
# Keep all other tabs (tab_phase, tab_peaks, tab_params, tab_report, tab_about) exactly as they were.

# Final footer (unchanged)
st.markdown(f"""
<hr style="border:none;border-top:1px solid {border_color};margin-top:48px;">
<p style="text-align:center;color:#1e293b;font-size:.72rem;margin-top:6px;">
  Co-Cr XRD Rietveld · Improved pattern visualization · Phases & peaks now clearly visible
</p>
""", unsafe_allow_html=True)
