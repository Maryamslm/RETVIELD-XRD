"""
╔══════════════════════════════════════════════════════════════════╗
║ Co-Cr Dental Alloy · Full Rietveld XRD Refinement (Improved)    ║
║ Clear phases & peaks · Enhanced visualization                   ║
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

# ========================= GITHUB CONFIG =========================
GITHUB_REPO = "Maryamslm/RETVIELD-XRD"
GITHUB_COMMIT = "e9716f8c3d4654fcba8eddde065d0472b1db69e9"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_COMMIT}/samples/"

AVAILABLE_FILES = {
    "CH0": ["CH0_1.ASC", "CH0_1.xrdml", "CH0.ASC", "CH0.xrdml"],
    "CH45": ["CH45_2.ASC", "CH45_2.xrdml", "CH45.ASC", "CH45.xrdml"],
    "CNH0": ["CNH0_3.ASC", "CNH0_3.xrdml", "CNH0.ASC", "CNH0.xrdml"],
    "CNH45": ["CNH45_4.ASC", "CNH45_4.xrdml", "CNH45.ASC", "CNH45.xrdml"],
    "PH0": ["PH0.ASC", "PH0.xrdml", "PH0_1.ASC", "PH0_1.xrdml"],
    "PH45": ["PH45.ASC", "PH45.xrdml", "PH45_1.ASC", "PH45_1.xrdml"],
    "PNH0": ["PNH0.ASC", "PNH0.xrdml", "PNH0_1.ASC", "PNH0_1.xrdml"],
    "PNH45": ["PNH45.ASC", "PNH45.xrdml", "PNH45_1.ASC", "PNH45_1.xrdml"],
    "MEDILOY_powder": ["MEDILOY_powder.xrdml", "MEDILOY_powder.ASC"],
}

# ========================= THEME =========================
def apply_theme(bg_theme: str, font_size: float, primary_color: str):
    themes = {
        "Dark Mode": {"bg": "#020617", "text": "#e2e8f0", "sidebar": "#030712", "panel": "#080e1a", "border": "#1e293b"},
        "Light Mode": {"bg": "#f8fafc", "text": "#0f172a", "sidebar": "#ffffff", "panel": "#f1f5f9", "border": "#cbd5e1"},
        "High Contrast": {"bg": "#000000", "text": "#00ff00", "sidebar": "#0a0a0a", "panel": "#111111", "border": "#00ff0044"}
    }
    t = themes.get(bg_theme, themes["Dark Mode"])
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif !important; font-size: {font_size}rem !important; }}
        code, pre {{ font-family: 'IBM Plex Mono', monospace !important; }}
        [data-testid="stAppViewContainer"] > .main {{ background-color: {t['bg']} !important; color: {t['text']} !important; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{ background: {t['sidebar']} !important; border-right: 1px solid {t['border']}; }}
        .stButton > button {{ border-radius: 8px !important; font-weight: 600 !important; }}
        .hero {{ background: linear-gradient(135deg, {t['bg']} 0%, {t['panel']} 45%, {t['bg']} 100%); border: 1px solid {t['border']}; border-radius: 14px; padding: 28px 36px 22px; margin-bottom: 22px; }}
        .hero h1 {{ font-size: 1.9rem; font-weight: 700; background: linear-gradient(100deg, {primary_color} 0%, #818cf8 50%, #34d399 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .mc {{ background: {t['panel']}; border: 1px solid {t['border']}; border-radius: 10px; padding: 12px 18px; flex: 1; min-width: 110px; }}
        .sh {{ font-size: .7rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: #334155; border-bottom: 1px solid {t['border']}; padding-bottom: 4px; margin: 16px 0 10px; }}
    </style>
    """, unsafe_allow_html=True)
    return t['border']

# ========================= PHASE DATABASE (unchanged) =========================
@dataclass
class AtomSite:
    element: str
    wyckoff: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    Biso: float = 0.5

@dataclass
class Phase:
    key: str
    name: str
    formula: str
    pdf_card: str
    crystal_system: str
    space_group: str
    sg_number: int
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    atoms: List[AtomSite] = field(default_factory=list)
    wf_init: float = 0.5
    color: str = "#60a5fa"
    group: str = "Primary"
    description: str = ""

    @property
    def volume(self) -> float:
        al, be, ga = map(np.radians, [self.alpha, self.beta, self.gamma])
        return self.a * self.b * self.c * np.sqrt(1 - np.cos(al)**2 - np.cos(be)**2 - np.cos(ga)**2 + 2*np.cos(al)*np.cos(be)*np.cos(ga))

def _build_phase_db() -> Dict[str, Phase]:
    db = {}
    # (All your phases - gamma_Co, epsilon_Co, sigma, etc. - kept exactly as original)
    db["gamma_Co"] = Phase(key="gamma_Co", name="γ-Co (FCC)", formula="Co", pdf_card="PDF 15-0806",
        crystal_system="cubic", space_group="Fm-3m", sg_number=225, a=3.5447,
        atoms=[AtomSite("Co", "4a", 0, 0, 0, 1.0, 0.40)], wf_init=0.70, color="#38bdf8")
    db["epsilon_Co"] = Phase(key="epsilon_Co", name="ε-Co (HCP)", formula="Co", pdf_card="PDF 05-0727",
        crystal_system="hexagonal", space_group="P63/mmc", sg_number=194, a=2.5071, c=4.0686,
        gamma=120, atoms=[AtomSite("Co", "2c", 1/3, 2/3, 0.25, 1.0, 0.40)], wf_init=0.15, color="#fb923c")
    # ... Add the rest of your phases (sigma, Cr_bcc, Mo_bcc, Co3Mo, M23C6, etc.) here ...
    # For brevity in this response, assume you paste all phases from your original code
    # (I kept only two for space; copy all from your original _build_phase_db)
    return db

PHASE_DB: Dict[str, Phase] = _build_phase_db()

# ========================= CRYSTALLOGRAPHY & PROFILE FUNCTIONS (unchanged) =========================
# Paste all your original functions here:
# _d_cubic, _d_hex, _d_tet, _allow_*, _f0, _F2, generate_reflections, _make_refined_phase,
# gaussian_profile, lorentzian_profile, pseudo_voigt_profile, get_profile_function,
# caglioti, lp_factor, chebyshev_bg, phase_pattern, _pack, _unpack, hill_howard, r_factors,
# RietveldRefiner class, make_demo_pattern, parse_file_content, fetch_github_xrd, q_color

# (To save space, I'm omitting the full block here — copy these sections exactly from your original code)

# ========================= IMPROVED PLOT FUNCTION =========================
def create_improved_fit_plot(tt, Iobs, results, refiner, show_hkl_labels, hkl_font_size,
                            hkl_label_offset, hkl_color, bg_theme, border_color, wavelength):
    r = results
    z_shift = float(r.get("z_shift", 0.0))
    _, _, pp_vec = _unpack(refiner.x0, refiner.n_bg, refiner.n_ph)

    fig = make_subplots(rows=2, cols=1, row_heights=[0.78, 0.22], shared_xaxes=True,
                        vertical_spacing=0.03, subplot_titles=("Rietveld Fit", "Difference Plot"))

    # Observed
    fig.add_trace(go.Scatter(x=tt, y=Iobs, mode="lines", name="Observed",
                             line=dict(color="#94a3b8", width=1.8)), row=1, col=1)

    # Background
    fig.add_trace(go.Scatter(x=tt, y=r["Ibg"], mode="lines", name="Background",
                             line=dict(color="#475569", width=1.2, dash="dot"),
                             fill="tozeroy", fillcolor="rgba(71,85,105,0.15)"), row=1, col=1)

    # Phase contributions
    for key, Iph in r["contribs"].items():
        ph = PHASE_DB[key]
        wf = r["wf"].get(key, 0) * 100
        fig.add_trace(go.Scatter(x=tt, y=Iph + r["Ibg"], mode="lines",
                                 name=f"{ph.name} ({wf:.1f}%)",
                                 line=dict(color=ph.color, width=1.7, dash="dash"),
                                 opacity=0.85), row=1, col=1)

    # Calculated
    fig.add_trace(go.Scatter(x=tt, y=r["Icalc"], mode="lines", name="Calculated",
                             line=dict(color="#fbbf24", width=2.4)), row=1, col=1)

    # Difference
    fig.add_trace(go.Scatter(x=tt, y=r["diff"], mode="lines", name="Difference",
                             line=dict(color="#818cf8", width=1.4),
                             fill="tozeroy", fillcolor="rgba(129,140,248,0.18)"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#475569", dash="dash"), row=2, col=1)

    # Peak ticks and labels
    if show_hkl_labels:
        y_max = float(Iobs.max())
        y_range = y_max - float(Iobs.min())
        base_y = y_max + y_range * hkl_label_offset / 100

        for i, ph_obj in enumerate(refiner.phases):
            a_ref = float(pp_vec[i][1])
            c_ref = float(pp_vec[i][2])
            ph_ref = _make_refined_phase(ph_obj, a_ref, c_ref)
            pks = generate_reflections(ph_ref, wl=wavelength, tt_min=float(tt.min()), tt_max=float(tt.max()))

            # Tick marks
            y_tick = float(Iobs.min()) - 0.06 * y_range
            fig.add_trace(go.Scatter(
                x=[p["tt"] + z_shift for p in pks],
                y=[y_tick] * len(pks),
                mode="markers",
                marker=dict(symbol="line-ns", size=13, color=ph_obj.color, line=dict(width=2.8)),
                showlegend=False
            ), row=1, col=1)

            # Labels with stagger to avoid overlap
            used = []
            label_col = ph_obj.color if hkl_color == "phase" else hkl_color
            for pk in sorted(pks, key=lambda x: x["tt"]):
                pos = pk["tt"] + z_shift
                stagger = sum(1 for u in used if abs(u - pos) < 1.0)
                label_y = base_y + stagger * (y_range * 0.045)

                fig.add_annotation(
                    x=pos, y=label_y,
                    text=f"({pk['h']}{pk['k']}{pk['l']})",
                    showarrow=False,
                    font=dict(size=hkl_font_size, color=label_col, family="IBM Plex Mono"),
                    xanchor="center", yanchor="bottom",
                    bordercolor=border_color, borderwidth=1, borderpad=3,
                    bgcolor="rgba(15,23,42,0.9)" if bg_theme == "Dark Mode" else "rgba(255,255,255,0.9)"
                )
                used.append(pos)

    fig.update_layout(
        height=700,
        legend=dict(font=dict(size=11), orientation="h", y=1.05),
        margin=dict(l=70, r=30, t=50, b=70),
        template="plotly_dark" if bg_theme == "Dark Mode" else "plotly_white"
    )
    fig.update_xaxes(title_text="2θ (°)", row=2, col=1)
    fig.update_yaxes(title_text="Intensity (counts)", row=1, col=1)
    fig.update_yaxes(title_text="Δ (obs−calc)", row=2, col=1)

    return fig

# ========================= SESSION STATE =========================
for key in ("results", "refiner", "tt", "Iobs", "elapsed", "selected_sample", "source_info"):
    if key not in st.session_state:
        st.session_state[key] = None

# ========================= SIDEBAR =========================
with st.sidebar:
    st.markdown("## ⚙️ Setup")
    bg_theme = st.selectbox("Background Theme", ["Dark Mode", "Light Mode", "High Contrast"])
    font_size = st.slider("Font Size Scale", 0.8, 1.3, 1.0, 0.05)
    primary_color = st.color_picker("Primary Accent Color", "#38bdf8")
    plot_theme = st.selectbox("Plot Theme", ["plotly_dark", "plotly_white", "plotly_light"])
    border_color = apply_theme(bg_theme, font_size, primary_color)

    st.markdown('<div class="sh">🏷️ Peak Labels</div>', unsafe_allow_html=True)
    show_hkl_labels = st.checkbox("Show (hkl) labels on peaks", value=True)
    hkl_font_size = st.slider("Label font size", 7, 18, 10)
    hkl_label_offset = st.slider("Label vertical offset (%)", 0, 60, 18)
    hkl_label_color = st.radio("Label color", ["Phase color", "White", "Black", "Custom"], index=0)
    if hkl_label_color == "Custom":
        hkl_color = st.color_picker("Custom label color", "#ffffff")
    elif hkl_label_color == "White":
        hkl_color = "#ffffff"
    elif hkl_label_color == "Black":
        hkl_color = "#000000"
    else:
        hkl_color = "phase"

    st.markdown('<div class="sh">📈 Profile Function</div>', unsafe_allow_html=True)
    profile_type = st.selectbox("Peak Profile", ["Pseudo-Voigt", "Gaussian", "Lorentzian"], index=0)

    # ... (Keep all your file loading, wavelength, phase selection, refinement flags exactly as original) ...

    run = st.button("▶ Run Rietveld Refinement", type="primary", use_container_width=True)

# ========================= HERO =========================
st.markdown(f"""
<div class="hero">
  <h1>🔬 Co-Cr Dental Alloy · Rietveld XRD Refinement</h1>
  <p>Full-profile Rietveld refinement with improved phase and peak visualization</p>
</div>
""", unsafe_allow_html=True)

# ========================= TABS =========================
tab_fit, tab_phase, tab_peaks, tab_params, tab_report, tab_about = st.tabs([
    "📈 Pattern Fit", "⚖️ Phase Analysis", "📋 Peak List",
    "🔧 Refined Parameters", "📄 Report", "ℹ️ About"
])

with tab_fit:
    if st.session_state.get("results") is None:
        st.info("👈 Load a file and run the refinement to see the improved fit plot.")
    else:
        r = st.session_state["results"]
        refiner = st.session_state["refiner"]
        tt = st.session_state["tt"]
        Iobs = st.session_state["Iobs"]
        elapsed = st.session_state["elapsed"]

        rwp = r["Rwp"]
        qc = q_color(rwp)
        st.markdown(f"""<div class="mstrip">... your metrics cards ...</div>""", unsafe_allow_html=True)

        fig = create_improved_fit_plot(
            tt, Iobs, r, refiner, show_hkl_labels, hkl_font_size,
            hkl_label_offset, hkl_color, bg_theme, border_color, wavelength
        )
        st.plotly_chart(fig, use_container_width=True)

# Keep the rest of your tabs (tab_phase, tab_peaks, etc.) exactly as in your original code

st.caption("Improved version — phases and peaks are now clearly visible")
