"""
╔══════════════════════════════════════════════════════════════════╗
║   Co-Cr Dental Alloy · Full Rietveld XRD Refinement             ║
║   Single-file Streamlit Application with Publication Plotting   ║
║   Supports .ASC/.XRDML · Smoothing · Multiple Peak Profiles     ║
║                                                                  ║
║   Usage:  streamlit run RETVIELD.py                              ║
║   Deps:   pip install streamlit numpy scipy pandas plotly requests matplotlib
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
from scipy.signal import savgol_filter
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO

# Set non-interactive backend for matplotlib to avoid GUI warnings on servers
matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Co-Cr XRD · Rietveld", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

# ═══════════════════════════════════════════════════════════════════
# GITHUB REPO CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════
# PUBLICATION PLOT GENERATOR
# ═══════════════════════════════════════════════════════════════════
def create_publication_figure(tt, I_obs, I_calc, I_bg, I_diff, phase_ticks, 
                              Rwp, Rp, chi2, GOF, sample_name, wavelength, smooth=True):
    """Generate publication-quality Rietveld plot using Matplotlib"""
    if smooth:
        try: I_obs_smooth = savgol_filter(I_obs, window_length=7, polyorder=3)
        except: I_obs_smooth = I_obs
    else:
        I_obs_smooth = I_obs

    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10, 'axes.linewidth': 1.2, 'axes.grid': False,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True, 'figure.dpi': 300
    })

    fig, (ax_main, ax_diff) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})
    
    # Main pattern
    ax_main.plot(tt, I_calc, color='#D32F2F', linewidth=1.5, label='Calculated')
    ax_main.plot(tt, I_obs_smooth, 'ko', markersize=2.5, alpha=0.7, label='Observed', rasterized=True)
    ax_main.plot(tt, I_bg, color='#999999', linewidth=0.8, linestyle='--', label='Background')
    
    # Bragg ticks
    y_min = np.min(I_bg) * 0.8
    colors = ['#0044AA', '#008844', '#AA4400', '#880088', '#4444AA', '#AA8800']
    for i, (name, ticks) in enumerate(phase_ticks.items()):
        if not ticks: continue
        row_y = y_min - (i + 1) * (abs(y_min) * 0.08)
        color = colors[i % len(colors)]
        ax_main.vlines(ticks, row_y - abs(y_min)*0.04, row_y + abs(y_min)*0.04, color=color, linewidth=1.8)
        ax_main.annotate(name, xy=(ticks[0]-0.5, row_y), color=color, fontsize=8.5, fontweight='bold', ha='right', va='center', fontfamily='serif')
    
    ax_main.set_ylim(y_min - len(phase_ticks)*abs(y_min)*0.1, np.max(I_obs)*1.1)
    ax_main.set_ylabel('Intensity (a.u.)', fontsize=11, fontweight='bold')
    ax_main.tick_params(axis='both', labelsize=10, width=1.2, length=4)
    ax_main.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=9)
    
    # Difference
    ax_diff.plot(tt, I_diff, color='#388E3C', linewidth=1.2)
    ax_diff.axhline(0, color='black', linewidth=0.8)
    ax_diff.fill_between(tt, 0, I_diff, color='#66BB6A', alpha=0.2)
    ax_diff.set_xlabel('2θ (degrees)', fontsize=11, fontweight='bold')
    ax_diff.set_ylabel('ΔI', fontsize=10)
    ax_diff.tick_params(axis='both', labelsize=10, width=1.2, length=4)
    ax_diff.set_xlim(ax_main.get_xlim())
    
    # Info box
    info = f"Sample: {sample_name}\nRadiation: {wavelength}\n$R_{{wp}}$: {Rwp:.2f}%\n$R_{{p}}$: {Rp:.2f}%\nχ²: {chi2:.3f}\nGOF: {GOF:.3f}"
    props = dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA', edgecolor='#333333', alpha=0.95, linewidth=1)
    ax_main.text(0.98, 0.98, info, transform=ax_main.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right', fontfamily='serif', bbox=props, zorder=10)
    
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf

# ═══════════════════════════════════════════════════════════════════
# APPEARANCE & THEME CONFIG
# ═══════════════════════════════════════════════════════════════════
def apply_theme(bg_theme: str, font_size: float, primary_color: str):
    themes = {"Dark Mode": {"bg": "#020617", "text": "#e2e8f0", "sidebar": "#030712", "panel": "#080e1a", "border": "#1e293b"},
              "Light Mode": {"bg": "#f8fafc", "text": "#0f172a", "sidebar": "#ffffff", "panel": "#f1f5f9", "border": "#cbd5e1"},
              "High Contrast": {"bg": "#000000", "text": "#00ff00", "sidebar": "#0a0a0a", "panel": "#111111", "border": "#00ff0044"}}
    t = themes.get(bg_theme, themes["Dark Mode"])
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif !important; font-size: {font_size}rem !important; }}
        code, pre {{ font-family: 'IBM Plex Mono', monospace !important; }}
        [data-testid="stAppViewContainer"] > .main {{ background-color: {t['bg']} !important; color: {t['text']} !important; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{ background: {t['sidebar']} !important; border-right: 1px solid {t['border']}; }}
        [data-testid="stSidebar"] * {{ color: {t['text']} !important; }}
        [data-testid="stSidebar"] .stSlider label, [data-testid="stSidebar"] .stCheckbox label {{ color: #94a3b8 !important; }}
        .stButton > button {{ border-radius: 8px !important; font-weight: 600 !important; letter-spacing: .03em !important; }}
        .stButton > button[kind="primary"] {{ background: linear-gradient(135deg, {primary_color}, #7c3aed) !important; border: none !important; color: white !important; }}
        .hero {{ background: linear-gradient(135deg, {t['bg']} 0%, {t['panel']} 45%, {t['bg']} 100%); border: 1px solid {t['border']}; border-radius: 14px; padding: 28px 36px 22px; margin-bottom: 22px; position: relative; overflow: hidden; }}
        .hero h1 {{ font-size: 1.9rem; font-weight: 700; letter-spacing: -.02em; background: linear-gradient(100deg, {primary_color} 0%, #818cf8 50%, #34d399 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 6px; }}
        .hero p {{ color: #64748b; margin: 0; font-size: .88rem; font-weight: 300; line-height: 1.5; }}
        .badge {{ display:inline-block; font-size:.7rem; font-weight:600; letter-spacing:.06em; padding:2px 9px; border-radius:99px; margin-right:6px; margin-top:10px; border:1px solid; }}
        .badge-cu {{ color:#f59e0b; border-color:#f59e0b44; background:#f59e0b10; }}
        .badge-iso {{ color:#34d399; border-color:#34d39944; background:#34d39910; }}
        .badge-slm {{ color:#818cf8; border-color:#818cf844; background:#818cf810; }}
        .mstrip {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:18px; }}
        .mc {{ background: {t['panel']}; border:1px solid {t['border']}; border-radius:10px; padding:12px 18px; flex:1; min-width:110px; }}
        .mc .lbl {{ font-size:.68rem; color:#475569; letter-spacing:.1em; text-transform:uppercase; }}
        .mc .val {{ font-size:1.45rem; font-weight:700; color: {t['text']}; font-family:'IBM Plex Mono', monospace; }}
        .mc .sub {{ font-size:.7rem; color:#334155; }}
        .sh {{ font-size:.7rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:#334155; border-bottom:1px solid {t['border']}; padding-bottom:4px; margin:16px 0 10px; }}
    </style>""", unsafe_allow_html=True)
    return t['border']

# ═══════════════════════════════════════════════════════════════════
# CRYSTAL STRUCTURE LIBRARY
# ═══════════════════════════════════════════════════════════════════
@dataclass
class AtomSite:
    element: str; wyckoff: str; x: float; y: float; z: float; occupancy: float = 1.0; Biso: float = 0.5

@dataclass
class Phase:
    key: str; name: str; formula: str; pdf_card: str; crystal_system: str; space_group: str; sg_number: int
    a: float; b: float; c: float; alpha: float = 90.0; beta: float = 90.0; gamma: float = 90.0
    atoms: List[AtomSite] = field(default_factory=list); wf_init: float = 0.5; color: str = "#60a5fa"; group: str = "Primary"; description: str = ""
    @property
    def volume(self) -> float:
        al, be, ga = map(np.radians, [self.alpha, self.beta, self.gamma])
        return self.a * self.b * self.c * np.sqrt(1 - np.cos(al)**2 - np.cos(be)**2 - np.cos(ga)**2 + 2*np.cos(al)*np.cos(be)*np.cos(ga))

def _build_phase_db() -> Dict[str, Phase]:
    db = {}
    db["gamma_Co"] = Phase(key="gamma_Co", name="γ-Co (FCC)", formula="Co", pdf_card="PDF 15-0806", crystal_system="cubic", space_group="Fm-3m", sg_number=225, a=3.5447, b=3.5447, c=3.5447, atoms=[AtomSite("Co", "4a", 0, 0, 0, 1.0, 0.40)], wf_init=0.70, color="#38bdf8", group="Primary", description="FCC cobalt — primary austenitic matrix.")
    db["epsilon_Co"] = Phase(key="epsilon_Co", name="ε-Co (HCP)", formula="Co", pdf_card="PDF 05-0727", crystal_system="hexagonal", space_group="P63/mmc", sg_number=194, a=2.5071, b=2.5071, c=4.0686, alpha=90, beta=90, gamma=120, atoms=[AtomSite("Co", "2c", 1/3, 2/3, 0.25, 1.0, 0.40)], wf_init=0.15, color="#fb923c", group="Primary", description="HCP cobalt — martensitic transform.")
    db["sigma"] = Phase(key="sigma", name="σ-phase (CoCr)", formula="CoCr", pdf_card="PDF 29-0490", crystal_system="tetragonal", space_group="P42/mnm", sg_number=136, a=8.7960, b=8.7960, c=4.5750, atoms=[AtomSite("Co", "2a", 0, 0, 0, 0.5, 0.50), AtomSite("Cr", "2a", 0, 0, 0, 0.5, 0.50), AtomSite("Co", "4f", 0.398, 0.398, 0, 0.5, 0.50), AtomSite("Cr", "4f", 0.398, 0.398, 0, 0.5, 0.50), AtomSite("Co", "8i", 0.464, 0.132, 0, 0.5, 0.50), AtomSite("Cr", "8i", 0.464, 0.132, 0, 0.5, 0.50)], wf_init=0.05, color="#4ade80", group="Secondary", description="Cr-rich intermetallic; appears after prolonged heat treatment.")
    db["Cr_bcc"] = Phase(key="Cr_bcc", name="Cr (BCC)", formula="Cr", pdf_card="PDF 06-0694", crystal_system="cubic", space_group="Im-3m", sg_number=229, a=2.8839, b=2.8839, c=2.8839, atoms=[AtomSite("Cr", "2a", 0, 0, 0, 1.0, 0.40)], wf_init=0.04, color="#f87171", group="Secondary", description="BCC chromium — excess Cr or incomplete alloying.")
    db["Mo_bcc"] = Phase(key="Mo_bcc", name="Mo (BCC)", formula="Mo", pdf_card="PDF 42-1120", crystal_system="cubic", space_group="Im-3m", sg_number=229, a=3.1472, b=3.1472, c=3.1472, atoms=[AtomSite("Mo", "2a", 0, 0, 0, 1.0, 0.45)], wf_init=0.03, color="#c084fc", group="Secondary", description="BCC molybdenum — inter-dendritic segregation.")
    db["Co3Mo"] = Phase(key="Co3Mo", name="Co₃Mo", formula="Co3Mo", pdf_card="PDF 29-0491", crystal_system="hexagonal", space_group="P63/mmc", sg_number=194, a=5.1400, b=5.1400, c=4.1000, alpha=90, beta=90, gamma=120, atoms=[AtomSite("Co", "6h", 1/6, 1/3, 0.25, 1.0, 0.50), AtomSite("Mo", "2c", 1/3, 2/3, 0.25, 1.0, 0.55)], wf_init=0.02, color="#a78bfa", group="Secondary", description="Hexagonal Co₃Mo — high-T annealing precipitate.")
    db["M23C6"] = Phase(key="M23C6", name="M₂₃C₆ Carbide", formula="Cr23C6", pdf_card="PDF 36-0803", crystal_system="cubic", space_group="Fm-3m", sg_number=225, a=10.61, b=10.61, c=10.61, atoms=[AtomSite("Cr", "24e", 0.35, 0, 0, 1.0, 0.50), AtomSite("Cr", "32f", 0.35, 0.35, 0.35, 1.0, 0.50), AtomSite("C", "32f", 0.30, 0.30, 0.30, 1.0, 0.50)], wf_init=0.05, color="#eab308", group="Carbides", description="Cr₂₃C₆ type; very common in cast alloys.")
    db["M6C"] = Phase(key="M6C", name="M₆C Carbide", formula="(Co,Mo)6C", pdf_card="PDF 27-0408", crystal_system="cubic", space_group="Fd-3m", sg_number=227, a=10.99, b=10.99, c=10.99, atoms=[AtomSite("Mo", "16c", 0, 0, 0, 0.5, 0.50), AtomSite("Co", "16d", 0.5, 0.5, 0.5, 0.5, 0.50), AtomSite("C", "48f", 0.375, 0.375, 0.375, 1.0, 0.50)], wf_init=0.05, color="#f97316", group="Carbides", description="Mo/W-rich; found in Mo- or W-containing alloys.")
    db["Laves"] = Phase(key="Laves", name="Laves Phase (Co₂Mo)", formula="Co2Mo", pdf_card="PDF 03-1225", crystal_system="hexagonal", space_group="P63/mmc", sg_number=194, a=4.73, b=4.73, c=7.72, alpha=90, beta=90, gamma=120, atoms=[AtomSite("Co", "2a", 0, 0, 0, 1.0, 0.50), AtomSite("Mo", "2d", 1/3, 2/3, 0.75, 1.0, 0.50), AtomSite("Co", "6h", 0.45, 0.90, 0.25, 1.0, 0.50)], wf_init=0.05, color="#d946ef", group="Laves", description="Hexagonal intermetallic precipitate.")
    db["Cr2O3"] = Phase(key="Cr2O3", name="Cr₂O₃", formula="Cr2O3", pdf_card="PDF 38-1479", crystal_system="trigonal", space_group="R-3m", sg_number=167, a=4.9580, b=4.9580, c=13.5942, alpha=90, beta=90, gamma=120, atoms=[AtomSite("Cr", "12c", 0, 0, 0.348, 1.0, 0.55), AtomSite("O", "18e", 0.306, 0, 0.25, 1.0, 0.60)], wf_init=0.02, color="#f472b6", group="Oxide", description="Chromium sesquioxide.")
    db["CoCr2O4"] = Phase(key="CoCr2O4", name="CoCr₂O₄", formula="CoCr2O4", pdf_card="PDF 22-1084", crystal_system="cubic", space_group="Fm-3m", sg_number=227, a=8.3216, b=8.3216, c=8.3216, atoms=[AtomSite("Co", "8a", 0.125, 0.125, 0.125, 1.0, 0.55), AtomSite("Cr", "16d", 0.5, 0.5, 0.5, 1.0, 0.55), AtomSite("O", "32e", 0.264, 0.264, 0.264, 1.0, 0.65)], wf_init=0.01, color="#22d3ee", group="Oxide", description="Cobalt-chromium spinel oxide.")
    return db

PHASE_DB = _build_phase_db()
PRIMARY_KEYS, SECONDARY_KEYS, CARBIDE_KEYS, LAVES_KEYS, OXIDE_KEYS = ["gamma_Co", "epsilon_Co"], ["sigma", "Cr_bcc", "Mo_bcc", "Co3Mo"], ["M23C6", "M6C"], ["Laves"], ["Cr2O3", "CoCr2O4"]

# ═══════════════════════════════════════════════════════════════════
# CRYSTALLOGRAPHY UTILITIES
# ═══════════════════════════════════════════════════════════════════
def _d_cubic(a, h, k, l): s = h*h + k*k + l*l; return a / np.sqrt(s) if s else np.inf
def _d_hex(a, c, h, k, l): t = (4/3)*((h*h + h*k + k*k) / a**2) + (l/c)**2; return 1/np.sqrt(t) if t > 0 else np.inf
def _d_tet(a, c, h, k, l): t = (h*h + k*k) / a**2 + l*l / c**2; return 1/np.sqrt(t) if t > 0 else np.inf
def _allow_fcc(h, k, l): return len({h%2, k%2, l%2}) == 1
def _allow_bcc(h, k, l): return (h+k+l) % 2 == 0
def _allow_hcp(h, k, l): return not (l%2 != 0 and (h-k)%3 == 0)
def _allow_sig(h, k, l): return (h+k+l) % 2 == 0
def _allow_all(h, k, l): return True
def _allow_fd3m(h, k, l): return not ((h%2 != k%2) or (k%2 != l%2) or ((h%2 == 0) and (h+k+l)%4 != 0))
_ALLOW = {"Fm-3m": _allow_fcc, "Im-3m": _allow_bcc, "P63/mmc": _allow_hcp, "P42/mnm": _allow_sig, "R-3m": _allow_all, "Fd-3m": _allow_fd3m}

_CM = {"Co": ([2.7686,2.2087,1.6079,1.0000],[14.178,3.398,0.124,41.698],0.9768), "Cr": ([2.3070,2.2940,0.8167,0.0000],[10.798,1.173,11.002,132.79],1.1003), "Mo": ([3.7025,2.3517,1.5442,0.8534],[12.943,2.658,0.157,39.714],0.6670), "O": ([0.4548,0.9177,0.4719,0.0000],[23.780,7.622,0.165,0.000],0.0000), "C": ([2.31, 1.02, 1.59, 0.0], [20.84, 10.21, 0.57, 51.65], 0.20), "W": ([4.0, 3.0, 2.0, 1.0], [10.0, 3.0, 0.5, 50.0], 0.5)}
def _f0(el, stl):
    if el not in _CM: return max({"Co":27,"Cr":24,"Mo":42,"O":8,"C":6}.get(el, 20) - stl*4, 1.0)
    a,b,c = _CM[el]; return c + sum(ai * np.exp(-bi * stl**2) for ai, bi in zip(a, b))

def _calc_d(ph, h, k, l):
    cs = ph.crystal_system.lower()
    if cs == "cubic": return _d_cubic(ph.a, h, k, l)
    elif cs in ("hexagonal", "trigonal"): return _d_hex(ph.a, ph.c, h, k, l)
    elif cs == "tetragonal": return _d_tet(ph.a, ph.c, h, k, l)
    return _d_cubic(ph.a, h, k, l)

def _F2(ph, h, k, l, wl=1.54056):
    d = _calc_d(ph, h, k, l); stl = 1.0/(2.0*d) if d > 0 else 0.0; Fr = Fi = 0.0
    for at in ph.atoms: f = _f0(at.element, stl); DW = np.exp(-at.Biso * stl**2); pa = 2*np.pi*(h*at.x + k*at.y + l*at.z); Fr += at.occupancy * f * DW * np.cos(pa); Fi += at.occupancy * f * DW * np.sin(pa)
    return Fr*Fr + Fi*Fi

def generate_reflections(ph, wl=1.54056, tt_min=10.0, tt_max=100.0, n=7):
    afn = _ALLOW.get(ph.space_group, _allow_all); seen = {}
    for h in range(-n, n+1):
        for k in range(-n, n+1):
            for l in range(-n, n+1):
                if h == k == l == 0 or not afn(h, k, l): continue
                d = _calc_d(ph, h, k, l)
                if d < 0.5 or d > 20: continue
                st = wl / (2.0*d)
                if abs(st) > 1: continue
                tt = 2.0 * np.degrees(np.arcsin(st))
                if not (tt_min <= tt <= tt_max): continue
                dk = round(d, 4)
                if dk in seen: seen[dk]["mult"] += 1
                else: seen[dk] = {"h":h, "k":k, "l":l, "d":d, "tt":tt, "mult":1}
    return sorted(seen.values(), key=lambda x: x["tt"])

def _make_refined_phase(ph, a_ref, c_ref):
    return Phase(key=ph.key, name=ph.name, formula=ph.formula, pdf_card=ph.pdf_card, crystal_system=ph.crystal_system, space_group=ph.space_group, sg_number=ph.sg_number, a=a_ref, b=(a_ref if ph.b == ph.a else ph.b), c=c_ref, alpha=ph.alpha, beta=ph.beta, gamma=ph.gamma, atoms=ph.atoms, color=ph.color)

# ═══════════════════════════════════════════════════════════════════
# PROFILE FUNCTIONS & REFINER
# ═══════════════════════════════════════════════════════════════════
def gaussian_profile(tt, tt_k, fwhm): sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))); return np.exp(-0.5 * ((tt - tt_k) / sigma)**2)
def lorentzian_profile(tt, tt_k, fwhm): gamma = fwhm / 2.0; return (gamma**2) / ((tt - tt_k)**2 + gamma**2)
def pseudo_voigt_profile(tt, tt_k, fwhm, eta): return eta * lorentzian_profile(tt, tt_k, fwhm) + (1.0 - eta) * gaussian_profile(tt, tt_k, fwhm)

def get_profile_function(pt): return {"Gaussian": lambda t, tk, fw, e: gaussian_profile(t, tk, fw), "Lorentzian": lambda t, tk, fw, e: lorentzian_profile(t, tk, fw), "Pseudo-Voigt": pseudo_voigt_profile}.get(pt, pseudo_voigt_profile)

def caglioti(tt_deg, U, V, W): th = np.radians(tt_deg / 2.0); return np.sqrt(max(U*np.tan(th)**2 + V*np.tan(th) + W, 1e-8))
def lp_factor(tt_deg): th = np.radians(tt_deg / 2.0); c2t = np.cos(2.0*th); c2m = np.cos(np.radians(26.6)); den = np.sin(th)**2 * np.cos(th); return (1.0 + c2t**2 * c2m**2) / den if den > 0 else 1.0
def chebyshev_bg(tt, coeffs, tt0, tt1):
    x = 2.0*(tt - tt0)/(tt1 - tt0) - 1.0; bg = np.zeros_like(tt); Tp, Tc = np.ones_like(x), x.copy()
    if len(coeffs) > 0: bg += coeffs[0]*Tp
    if len(coeffs) > 1: bg += coeffs[1]*Tc
    for c in coeffs[2:]: Tn = 2.0*x*Tc - Tp; bg += c*Tn; Tp, Tc = Tc, Tn
    return bg

def phase_pattern(tt, ph, a, c, scale, U, V, W, eta0, z_shift, wl, pt="Pseudo-Voigt"):
    ph_r = _make_refined_phase(ph, a, c); refls = generate_reflections(ph_r, wl=wl, tt_min=max(float(tt.min())-5.0, 0.1), tt_max=float(tt.max())+5.0)
    avg_biso = np.mean([at.Biso for at in ph.atoms]) if ph.atoms else 0.5; I = np.zeros_like(tt); pf = get_profile_function(pt)
    for r in refls:
        tt_k = r["tt"] + z_shift; F2 = _F2(ph_r, r["h"], r["k"], r["l"], wl); lp = lp_factor(tt_k); fwhm = caglioti(tt_k, U, V, W)
        stl = np.sin(np.radians(tt_k / 2.0)) / wl; DW = np.exp(-avg_biso * stl**2)
        I += scale * r["mult"] * F2 * lp * DW * pf(tt, tt_k, fwhm, eta0)
    return I

N_PP = 7
def _pack(z, bg, pp): return np.array([z, *bg, *[v for p in pp for v in p]], dtype=float)
def _unpack(v, n_bg, n_ph): z=float(v[0]); bg=v[1:1+n_bg]; pp=[v[1+n_bg+i*N_PP:1+n_bg+(i+1)*N_PP] for i in range(n_ph)]; return z, bg, pp

_MASS = {"Co":58.933,"Cr":51.996,"Mo":95.950,"O":15.999,"C":12.011,"W":183.84}
def hill_howard(phases, pp):
    totals = {}; gt=0
    for ph, p in zip(phases, pp): scale=float(p[0]); uc=sum(_MASS.get(at.element, 50.0)*at.occupancy for at in ph.atoms) or 1.0; totals[ph.key]=scale*uc*ph.volume; gt+=totals[ph.key]
    return {k: v/(gt or 1.0) for k, v in totals.items()}

def r_factors(I_obs, I_calc, w):
    num=float(np.sum(w*(I_obs-I_calc)**2)); den=float(np.sum(w*I_obs**2)); Rwp=np.sqrt(num/den) if den>0 else 99.0
    Rp=float(np.sum(np.abs(I_obs-I_calc))/np.sum(np.abs(I_obs))); chi2=num/max(len(I_obs)-1,1); Re=np.sqrt((len(I_obs)-1)/den) if den>0 else 1.0
    return dict(Rwp=Rwp, Rp=Rp, chi2=chi2, Re=Re, GOF=Rwp/Re if Re>0 else 99.0)

class RietveldRefiner:
    def __init__(self, tt, I_obs, phase_keys, wl=1.54056, n_bg=5, pt="Pseudo-Voigt"):
        self.tt=tt.astype(float); self.Iobs=np.maximum(I_obs.astype(float), 0.0); self.wl=float(wl); self.n_bg=int(n_bg); self.pt=pt
        self.phases=[PHASE_DB[k] for k in phase_keys]; self.n_ph=len(self.phases); self.w=1.0/np.maximum(self.Iobs, 1.0); self._init_x0()
    def _init_x0(self):
        Ipeak=float(np.percentile(self.Iobs, 95)); Imin=float(np.percentile(self.Iobs, 10)); bg0=[Imin]+[0.0]*(self.n_bg-1)
        pp=[[ph.wf_init*Ipeak*1e-4, ph.a, ph.c, 0.02, -0.01, 0.005, 0.5] for ph in self.phases]; self.x0=_pack(0.0, bg0, pp)
    def _calc(self, v):
        z, bg_c, pp = _unpack(v, self.n_bg, self.n_ph); bg=chebyshev_bg(self.tt, bg_c, self.tt.min(), self.tt.max()); Icalc=bg.copy(); contribs={}
        for ph, p in zip(self.phases, pp): sc,a,c,U,V,W,et=(float(x) for x in p); Iph=phase_pattern(self.tt, ph, a, c, sc, U, V, W, et, z, self.wl, self.pt); contribs[ph.key]=Iph; Icalc+=Iph
        return Icalc, bg, contribs
    def _res(self, v): Icalc, _, _ = self._calc(v); return np.sqrt(self.w)*(self.Iobs-Icalc)
    def _bounds(self, flags):
        n=len(self.x0); lo,hi=np.full(n,-np.inf), np.full(n,np.inf); x=self.x0
        def frz(i): lo[i],hi[i]=x[i]-1e-10, x[i]+1e-10
        def fre(i,lb,ub): lo[i],hi[i]=lb,ub
        fre(0, -1.0, 1.0) if flags.get("zero") else frz(0)
        for j in range(1,1+self.n_bg): fre(j,-1e7,1e7) if flags.get("bg") else frz(j)
        for i,ph in enumerate(self.phases): b=1+self.n_bg+i*N_PP; fre(b,0,1e12) if flags.get("scale") else frz(b); fre(b+1,ph.a*0.95,ph.a*1.05) if flags.get("lattice") else frz(b+1); fre(b+2,ph.c*0.95,ph.c*1.05) if flags.get("lattice") else frz(b+2)
        if flags.get("profile"): fre(b+3,0,0.5); fre(b+4,-0.1,0); fre(b+5,1e-4,0.1); fre(b+6,0,1)
        else: [frz(b+j) for j in range(3,7)]
        return lo,hi
    def refine(self, flags, max_iter=400):
        lo,hi=self._bounds(flags); m=(lo==hi); hi[m]+=1e-9
        try: res=least_squares(self._res, self.x0, bounds=(lo,hi), method="trf", max_nfev=max_iter, ftol=1e-7, xtol=1e-7, gtol=1e-7, verbose=0); self.x0=res.x
        except Exception as e: st.warning(f"Optimisation note: {e}")
        Icalc,bg,contribs=self._calc(self.x0); rf=r_factors(self.Iobs, Icalc, self.w)
        z,bg_c,pp=_unpack(self.x0,self.n_bg,self.n_ph); wf=hill_howard(self.phases,pp); lat={}
        for ph,p in zip(self.phases,pp): sc,a,c,U,V,W,et=(float(x) for x in p); lat[ph.key]={"a_init":ph.a,"c_init":ph.c,"a_ref":a,"c_ref":c,"da":a-ph.a,"dc":c-ph.c,"U":U,"V":V,"W":W,"eta":et,"scale":sc}
        return {**rf,"Icalc":Icalc,"Ibg":bg,"contribs":contribs,"diff":self.Iobs-Icalc,"wf":wf,"lat":lat,"z_shift":z}

# ═══════════════════════════════════════════════════════════════════
# DEMO & FILE PARSER
# ═══════════════════════════════════════════════════════════════════
@st.cache_data
def make_demo_pattern(noise=0.025, seed=7):
    rng=np.random.default_rng(seed); tt=np.linspace(10,100,4500); wf={"gamma_Co":0.68,"epsilon_Co":0.15,"sigma":0.08,"Cr_bcc":0.05,"Mo_bcc":0.04}
    bg_c=np.array([280.,-60.,25.,-8.,4.]); I=chebyshev_bg(tt,bg_c,tt.min(),tt.max())
    for k,w in wf.items(): ph=PHASE_DB[k]; I+=phase_pattern(tt,ph,ph.a,ph.c,w*7500,0.025,-0.012,0.006,0.45,0.0,1.54056)
    I=np.maximum(I,0.0); I=rng.poisson(I).astype(float)+rng.normal(0,noise*I.max(),size=I.shape); return tt,np.maximum(I,0.0)

def parse_file_content(content, filename):
    name=filename.lower()
    if name.endswith(".xrdml"):
        try:
            root=ET.fromstring(content)
            cn=root.find(".//{*}counts") or root.find(".//counts")
            if cn is None: raise ValueError("No counts node found.")
            I=np.array(cn.text.split(), dtype=float); sp=ep=None
            for p in root.findall(".//{*}positions"):
                if "2Theta" in p.get("axis",""):
                    try: sp=float(p.find("{*}startPosition").text); ep=float(p.find("{*}endPosition").text)
                    except: pass
            tt=np.linspace(sp or 10.0, ep or 100.0, len(I))
            return tt,I
        except ET.ParseError as e: raise ValueError(f"XML error: {e}")
            
    lines=[ln.strip() for ln in content.splitlines() if ln.strip() and ln.strip()[0] not in "#!/'\";"]
    data=[]
    # Correct indentation here to ensure try/except is inside the loop
    for ln in lines: 
        parts = ln.replace(",", " ").split()
        try:
            if len(parts)>=2: data.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue # Now inside the loop
            
    if not data: raise ValueError("Cannot parse.")
    arr = np.array(data)
    tt, I = arr[:, 0], arr[:, 1]
    if tt.max() < 5: tt = np.degrees(tt)
    if not np.all(tt[:-1] <= tt[1:]):
        idx = np.argsort(tt); tt, I = tt[idx], I[idx]
    return tt, I

def fetch_github_xrd(sample, ext=".ASC"):
    if sample not in AVAILABLE_FILES: raise ValueError(f"Sample '{sample}' not found.")
    # First pass: try exact match
    for fn in AVAILABLE_FILES[sample]:
        if fn.endswith(ext):
            try:
                r=requests.get(f"{GITHUB_RAW_BASE}{fn}", timeout=30)
                r.raise_for_status()
                return parse_file_content(r.text, fn) + (fn,)
            except: continue
    # Second pass: try any extension
    for fn in AVAILABLE_FILES[sample]:
        try:
            r=requests.get(f"{GITHUB_RAW_BASE}{fn}", timeout=30)
            r.raise_for_status()
            return parse_file_content(r.text, fn) + (fn,)
        except: continue
    raise ValueError(f"Could not fetch '{sample}'.")

def q_color(rwp): return "#4ade80" if rwp<0.05 else ("#fbbf24" if rwp<0.10 else "#f87171")

# ═══════════════════════════════════════════════════════════════════
# SESSION STATE & SIDEBAR
# ═══════════════════════════════════════════════════════════════════
for _k in ("results","refiner","tt","Iobs","elapsed","selected_sample","source_info"):
    if _k not in st.session_state: st.session_state[_k]=None

with st.sidebar:
    st.markdown("## ⚙️ Setup")
    st.markdown('<div class="sh">🎨 Appearance</div>', unsafe_allow_html=True)
    bg_theme=st.selectbox("Background Theme", ["Dark Mode","Light Mode","High Contrast"])
    font_size=st.slider("Font Size Scale", 0.8, 1.3, 1.0, 0.05)
    primary_color=st.color_picker("Accent Color", "#38bdf8")
    plot_theme=st.selectbox("Plot Theme", ["plotly_dark","plotly_white","plotly_light"])
    border_color=apply_theme(bg_theme, font_size, primary_color)
    
    st.markdown('<div class="sh">🏷️ Peak Labels</div>', unsafe_allow_html=True)
    show_labels=st.checkbox("Show (hkl) labels", value=True)
    label_font=st.slider("Label font size", 8, 16, 10)
    label_offset=st.slider("Vertical offset (%)", 0, 50, 15)
    label_col_mode=st.radio("Label color", ["Phase","White","Black","Custom"])
    if label_col_mode=="Custom": label_color=st.color_picker("Custom", "#ffffff")
    elif label_col_mode=="White": label_color="#ffffff"
    elif label_col_mode=="Black": label_color="#000000"
    else: label_color="phase"
    
    st.markdown('<div class="sh">📈 Profile</div>', unsafe_allow_html=True)
    profile_type=st.selectbox("Peak Function", ["Pseudo-Voigt","Gaussian","Lorentzian"])
    
    st.markdown('<div class="sh">📁 GitHub Files</div>', unsafe_allow_html=True)
    selected_sample=st.selectbox("Select Sample", list(AVAILABLE_FILES.keys()))
    file_ext=st.radio("Format", [".ASC",".xrdml"], horizontal=True)
    fetch_btn=st.button("🔄 Load from GitHub", type="primary", use_container_width=True)
    tt_raw=I_raw=None; source_info=""
    if fetch_btn:
        with st.spinner("Fetching..."):
            try: tt_raw,I_raw,fn=fetch_github_xrd(selected_sample, file_ext); source_info=f"✓ {fn} ({len(tt_raw)} pts)"; st.success(source_info); st.session_state.selected_sample=selected_sample; st.session_state.source_info=source_info
            except Exception as e: st.error(str(e))
    with st.expander("🔁 Demo / Upload", expanded=False):
        src=st.radio("", ["Demo","Upload"], label_visibility="collapsed")
        if src=="Demo": tt_raw,I_raw=make_demo_pattern(); source_info="Demo loaded"
        else:
            up=st.file_uploader("Upload", type=["xy","dat","txt","csv","xrdml","asc"])
            if up:
                try: tt_raw,I_raw=parse_file_content(up.read().decode(), up.name); source_info=f"✓ {up.name}"
                except Exception as e: st.error(str(e))
                
    st.markdown('<div class="sh">⚙️ Instrument & Window</div>', unsafe_allow_html=True)
    wl_opt={"Cu Kα₁":1.54056,"Cu Kα":1.54184,"Mo Kα₁":0.70932,"Ag Kα₁":0.56087,"Co Kα₁":1.78900}
    wl_label=st.selectbox("Wavelength", list(wl_opt.keys())); wavelength=wl_opt[wl_label]
    zero_seed=st.slider("Zero-shift (°)", -1.0, 1.0, 0.0, 0.01)
    tt_lo,tt_hi=st.slider("2θ Range", 10.0, 120.0, (15.0, 95.0), 0.5)
    
    st.markdown('<div class="sh">🧊 Phases</div>', unsafe_allow_html=True)
    sel_keys=[]; groups=[("Primary",PRIMARY_KEYS,True),("Secondary",SECONDARY_KEYS,True),("Carbides",CARBIDE_KEYS,True),("Laves",LAVES_KEYS,True),("Oxides",OXIDE_KEYS,False)]
    for grp,keys,exp in groups:
        with st.expander(grp, expanded=exp):
            for k in keys:
                ph=PHASE_DB[k]
                if st.checkbox(f"{ph.name} ({ph.formula})", value=k in PRIMARY_KEYS+SECONDARY_KEYS[:2], help=ph.description): sel_keys.append(k)
    if not sel_keys: st.warning("Select ≥1 phase.")
    
    st.markdown('<div class="sh">🔧 Flags</div>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    fl_scale=c1.checkbox("Scale",True); fl_lattice=c2.checkbox("Lattice",True); fl_bg=c1.checkbox("BG",True); fl_prof=c2.checkbox("Profile",True); fl_zero=st.checkbox("Zero",False)
    n_bg=st.slider("BG terms", 2, 8, 5); max_it=st.slider("Iter", 50, 1000, 350, 50)
    run=st.button("▶ Run Refinement", type="primary", use_container_width=True, disabled=(tt_raw is None or not sel_keys))

# ═══════════════════════════════════════════════════════════════════
# HERO & RUN
# ═══════════════════════════════════════════════════════════════════
st.markdown(f"""<div class="hero"><h1>🔬 Co-Cr Dental Alloy · Rietveld</h1><p>Full-profile refinement for 3D-printed Co-Cr alloys</p><span class="badge badge-cu">Cu/Mo/Co Kα</span><span class="badge badge-iso">ISO 22674</span><span class="badge badge-slm">SLM/DMLS</span></div>""", unsafe_allow_html=True)
if st.session_state.selected_sample and st.session_state.source_info: st.caption(f"📊 **{st.session_state.selected_sample}** — {st.session_state.source_info}")

if run and tt_raw is not None and sel_keys:
    mask=(tt_raw>=tt_lo)&(tt_raw<=tt_hi); tt_c,I_c=tt_raw[mask],I_raw[mask]
    if len(tt_c)<50: st.error("Widen 2θ window.")
    else:
        prog=st.progress(0,"Init…"); t0=time.time()
        refiner=RietveldRefiner(tt_c,I_c,sel_keys,wavelength,n_bg,profile_type); refiner.x0[0]=float(zero_seed)
        prog.progress(15,"Optimizing…")
        res=refiner.refine(dict(scale=fl_scale,lattice=fl_lattice,bg=fl_bg,profile=fl_prof,zero=fl_zero), max_iter=max_it); elapsed=time.time()-t0
        prog.progress(100,f"Done ({elapsed:.1f}s)"); time.sleep(0.3); prog.empty()
        st.session_state.update(results=res,refiner=refiner,tt=tt_c,Iobs=I_c,elapsed=elapsed)

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tabs=st.tabs(["📈 Fit","⚖️ Phases","📋 Peaks","🔧 Params","📄 Report","ℹ️ About"])

with tabs[0]:
    if st.session_state["results"] is None:
        if tt_raw is not None:
            mask=(tt_raw>=tt_lo)&(tt_raw<=tt_hi); fig=go.Figure(go.Scatter(x=tt_raw[mask],y=I_raw[mask],line=dict(color=primary_color,width=1))); fig.update_layout(template=plot_theme, xaxis_title="2θ", yaxis_title="I", height=350)
            st.plotly_chart(fig, use_container_width=True)
        st.info("👈 Load data & run refinement.")
    else:
        r,refiner,tt,Iobs,elapsed=st.session_state["results"],st.session_state["refiner"],st.session_state["tt"],st.session_state["Iobs"],st.session_state["elapsed"]
        z_shift=float(r.get("z_shift",0.0)); _,_,pp_vec=_unpack(refiner.x0,refiner.n_bg,refiner.n_ph)
        rwp,rp,gof,chi2=r["Rwp"],r["Rp"],r["GOF"],r["chi2"]; qc=q_color(rwp)
        st.markdown(f"""<div class="mstrip"><div class="mc"><div class="lbl">R_wp</div><div class="val" style="color:{qc}">{rwp*100:.2f}</div><div class="sub">%</div></div><div class="mc"><div class="lbl">R_p</div><div class="val">{rp*100:.2f}</div><div class="sub">%</div></div><div class="mc"><div class="lbl">GOF</div><div class="val">{gof:.3f}</div></div><div class="mc"><div class="lbl">χ²</div><div class="val">{chi2:.4f}</div></div><div class="mc"><div class="lbl">Pts</div><div class="val">{len(tt)}</div></div><div class="mc"><div class="lbl">Time</div><div class="val">{elapsed:.1f}s</div></div></div>""", unsafe_allow_html=True)
        
        fig=make_subplots(rows=2,cols=1,row_heights=[0.78,0.22],shared_xaxes=True,vertical_spacing=0.02)
        fig.add_trace(go.Scatter(x=tt,y=Iobs,mode="lines",name="Observed",line=dict(color="#94a3b8",width=1.3)),1,1)
        fig.add_trace(go.Scatter(x=tt,y=r["Ibg"],mode="lines",name="BG",line=dict(color="#334155",width=1,dash="dot"),fill="tozeroy"),1,1)
        for k,Iph in r["contribs"].items(): fig.add_trace(go.Scatter(x=tt,y=Iph+r["Ibg"],mode="lines",name=f"{PHASE_DB[k].name} ({r['wf'][k]*100:.1f}%)",line=dict(color=PHASE_DB[k].color,width=1.6,dash="dash"),opacity=0.8),1,1)
        fig.add_trace(go.Scatter(x=tt,y=r["Icalc"],mode="lines",name="Calc",line=dict(color="#fbbf24",width=2.2)),1,1)
        fig.add_trace(go.Scatter(x=tt,y=r["diff"],mode="lines",name="Δ",line=dict(color="#818cf8",width=1),fill="tozeroy"),2,1)
        fig.add_hline(y=0,line=dict(color="#334155",dash="dash"),row=2,col=1)
        
        if show_labels:
            y_max,y_min=float(Iobs.max()),float(Iobs.min()); rng=y_max-y_min if y_max>y_min else 1000; ly=y_max+(rng*label_offset/100)
            for i,ph in enumerate(refiner.phases):
                a_r,c_r=float(pp_vec[i][1]),float(pp_vec[i][2]); pks=generate_reflections(_make_refined_phase(ph,a_r,c_r),wl=wavelength,tt_min=float(tt.min()),tt_max=float(tt.max()))
                yt=y_min-(rng*0.08); fig.add_trace(go.Scatter(x=[p["tt"]+z_shift for p in pks],y=[yt]*len(pks),mode="markers",marker=dict(symbol="triangle-up",size=12,color=ph.color),name=f"{ph.name} ticks",showlegend=True),1,1)
                lc=ph.color if label_color=="phase" else label_color
                for pk in pks: fig.add_annotation(x=pk["tt"]+z_shift, y=ly, text=f"({pk['h']} {pk['k']} {pk['l']})", showarrow=False, font=dict(size=label_font,color=lc,family="IBM Plex Mono"), xanchor="center", yanchor="bottom", bgcolor="rgba(0,0,0,0.4)" if bg_theme=="Dark Mode" else "rgba(255,255,255,0.8)")
        fig.update_layout(template=plot_theme, height=650, margin=dict(l=60,r=20,t=20,b=50), legend=dict(font=dict(size=10)), xaxis=dict(title="2θ (°)"), yaxis=dict(title="Intensity"), xaxis2=dict(title="2θ (°)"), yaxis2=dict(title="Δ"))
        st.plotly_chart(fig, use_container_width=True)
        
        # 🖨️ PUBLICATION PLOT SECTION
        with st.expander("🖨️ Export Publication-Quality Figure", expanded=False):
            c1,c2,c3=st.columns(3)
            with c1: smooth=st.checkbox("Savitzky-Golay smoothing", True); sw=st.slider("Window", 3, 21, 7, 2)
            with c2: fig_w=st.slider("Width (in)", 6.0, 12.0, 8.0, 0.5); fig_h=st.slider("Height (in)", 4.0, 10.0, 6.0, 0.5)
            with c3: fmt=st.selectbox("Format", ["PNG","PDF","SVG"])
            
            if st.button("🎨 Generate & Download"):
                plt.rcParams['figure.figsize']=(fig_w, fig_h)
                phase_ticks={}
                for i,ph in enumerate(refiner.phases):
                    a_r,c_r=float(pp_vec[i][1]),float(pp_vec[i][2]); pks=generate_reflections(_make_refined_phase(ph,a_r,c_r),wl=wavelength,tt_min=float(tt.min()),tt_max=float(tt.max()))
                    phase_ticks[ph.name]=[p["tt"]+z_shift for p in pks]
                buf=create_publication_figure(tt, Iobs, r["Icalc"], r["Ibg"], r["diff"], phase_ticks, rwp*100, rp*100, chi2, gof, st.session_state.selected_sample or "Co-Cr", f"{wl_label} ({wavelength}Å)", smooth)
                st.image(buf)
                st.download_button(f"📥 Download .{fmt.lower()}", buf, f"rietveld_plot.{fmt.lower()}", f"image/{fmt.lower()}")
        
        df=pd.DataFrame({"2θ":tt,"I_obs":Iobs,"I_calc":r["Icalc"],"I_bg":r["Ibg"],"Diff":r["diff"],**{f"I_{k}":v for k,v in r["contribs"].items()}})
        st.download_button("⬇ Data CSV", df.to_csv(index=False), "rietveld_data.csv", "text/csv")

with tabs[1]:
    if st.session_state["results"] is None: st.info("Run first.")
    else:
        r,wf=st.session_state["results"],st.session_state["results"]["wf"]
        c1,c2=st.columns(2)
        with c1:
            fig=go.Figure(go.Pie(labels=[PHASE_DB[k].name for k in wf], values=[wf[k]*100 for k in wf], hole=0.6, textinfo="label+percent", marker=dict(colors=[PHASE_DB[k].color for k in wf])))
            fig.update_layout(template=plot_theme, height=350); st.plotly_chart(fig, use_container_width=True)
        with c2:
            idx=np.argsort([wf[k]*100 for k in wf])[::-1]; fig=go.Figure(go.Bar(x=[wf[k]*100 for k in [list(wf.keys())[i] for i in idx]], y=[PHASE_DB[k].name for k in [list(wf.keys())[i] for i in idx]], orientation="h", text=[f"{wf[k]*100:.2f}%" for k in [list(wf.keys())[i] for i in idx]], textposition="inside"))
            fig.update_layout(template=plot_theme, height=350, xaxis_title="wt %"); st.plotly_chart(fig, use_container_width=True)
        df=pd.DataFrame([{"Phase":PHASE_DB[k].name,"Formula":PHASE_DB[k].formula,"wt%":f"{wf[k]*100:.2f}","a":f"{r['lat'][k]['a_ref']:.4f}"} for k in wf])
        st.dataframe(df, use_container_width=True, hide_index=True)

with tabs[2]:
    if st.session_state["refiner"] is None: st.info("Run first.")
    else:
        refiner,tt,r=st.session_state["refiner"],st.session_state["tt"],st.session_state["results"]; z_shift=float(r.get("z_shift",0.0)); _,_,pp_vec=_unpack(refiner.x0,refiner.n_bg,refiner.n_ph)
        show=st.multiselect("Phases", [ph.key for ph in refiner.phases], default=[ph.key for ph in refiner.phases], format_func=lambda k: PHASE_DB[k].name)
        rows=[]
        for i,ph in enumerate(refiner.phases):
            if ph.key not in show: continue
            a_r,c_r=float(pp_vec[i][1]),float(pp_vec[i][2])
            for ref in generate_reflections(_make_refined_phase(ph,a_r,c_r),wl=wavelength,tt_min=float(tt.min()),tt_max=float(tt.max())):
                rows.append({"Phase":ph.name,"hkl":f"({ref['h']} {ref['k']} {ref['l']})","d(Å)":f"{ref['d']:.4f}","2θ(°)":f"{ref['tt']+z_shift:.3f}","Mult":ref["mult"]})
        if rows: st.dataframe(pd.DataFrame(rows).sort_values("2θ(°)"), use_container_width=True, height=500)
        else: st.warning("No peaks in range.")

with tabs[3]:
    if st.session_state["results"] is None: st.info("Run first.")
    else:
        r,refiner,tt=st.session_state["results"],st.session_state["refiner"],st.session_state["tt"]
        st.dataframe(pd.DataFrame([{"Phase":ph.name,"a":f"{r['lat'][ph.key]['a_ref']:.4f}","c":f"{r['lat'][ph.key]['c_ref']:.4f}" if ph.c!=ph.a else "—","U":f"{r['lat'][ph.key]['U']:.4f}","V":f"{r['lat'][ph.key]['V']:.4f}","W":f"{r['lat'][ph.key]['W']:.4f}","η":f"{r['lat'][ph.key]['eta']:.3f}"} for ph in refiner.phases]), use_container_width=True, hide_index=True)
        z,bg,_=_unpack(refiner.x0,refiner.n_bg,refiner.n_ph)
        st.code(f"Zero: {z:+.4f}°\nWL: {wavelength}Å\nBG: {' '.join(f'{float(v):.1f}' for v in bg)}", language="text")

with tabs[4]:
    if st.session_state["results"] is None: st.info("Run first.")
    else:
        r,tt,elapsed=st.session_state["results"],st.session_state["tt"],st.session_state["elapsed"]
        md=["# Rietveld Report", f"**Date:** {pd.Timestamp.now():%Y-%m-%d}", f"**WL:** {wl_label} ({wavelength}Å)", f"**Pts:** {len(tt)}", f"**2θ:** {tt.min():.1f}-{tt.max():.1f}°", f"**Time:** {elapsed:.1f}s", "---", "|R_wp|Rp|χ²|GOF|","|---|---|---|---|",f"|{r['Rwp']*100:.2f}%|{r['Rp']*100:.2f}%|{r['chi2']:.3f}|{r['GOF']:.3f}|","---"]
        md.append("|Phase|wt%|a(Å)|c(Å)|"); md.append("|---|---|---|---|")
        for k,w in r["wf"].items(): md.append(f"|{PHASE_DB[k].name}|{w*100:.2f}|{r['lat'][k]['a_ref']:.4f}|{r['lat'][k]['c_ref']:.4f if PHASE_DB[k].c!=PHASE_DB[k].a else '—'}|")
        st.markdown("\n".join(md)); st.download_button("📄 Report MD", "\n".join(md), "report.md", "text/markdown")

with tabs[5]:
    st.markdown("## ℹ️ About\nFull-profile Rietveld refinement for SLM/DMLS Co-Cr alloys.\n**Features:** GitHub loading, Gaussian/Lorentzian/PV profiles, Savitzky-Golay smoothing, publication export, phase labeling.\n```bash\npip install streamlit numpy scipy pandas plotly requests matplotlib\nstreamlit run RETVIELD.py\n```")

st.markdown("<hr style='border:none;border-top:1px solid #0f172a;margin-top:48px;'>", unsafe_allow_html=True)
