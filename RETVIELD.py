"""
╔══════════════════════════════════════════════════════════════════╗
║   Co-Cr Dental Alloy · Full Rietveld XRD Refinement             ║
║   Single-file Streamlit Application with GitHub File Selector   ║
║   Supports .ASC (two-column text) and .XRDML (Panalytical XML)  ║
║                                                                  ║
║   Usage:  streamlit run app_full.py                              ║
║   Deps:   pip install streamlit numpy scipy pandas plotly requests
╚══════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════
import time
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import requests
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import least_squares

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Co-Cr XRD · Rietveld",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# GITHUB REPO CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
GITHUB_REPO = "Maryamslm/RETVIELD-XRD"
GITHUB_COMMIT = "e9716f8c3d4654fcba8eddde065d0472b1db69e9"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_COMMIT}/samples/"

# Available files: user-friendly names → possible repository filenames
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
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
code, pre { font-family: 'IBM Plex Mono', monospace; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#030712 0%,#0c1524 100%);
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stCheckbox label { color:#94a3b8 !important; }

.hero {
    background: linear-gradient(135deg,#020617 0%,#0f2a4a 45%,#020617 100%);
    border: 1px solid #1e3a5f55; border-radius: 14px;
    padding: 28px 36px 22px; margin-bottom: 22px;
    position: relative; overflow: hidden;
}
.hero::after {
    content:""; position:absolute; inset:0;
    background: radial-gradient(ellipse 70% 60% at 20% 50%,#1d4ed812,transparent);
    pointer-events:none;
}
.hero h1 {
    font-size:1.9rem; font-weight:700; letter-spacing:-.02em;
    background: linear-gradient(100deg,#38bdf8 0%,#818cf8 50%,#34d399 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin:0 0 6px;
}
.hero p { color:#64748b; margin:0; font-size:.88rem; font-weight:300; line-height:1.5; }
.badge {
    display:inline-block; font-size:.7rem; font-weight:600; letter-spacing:.06em;
    padding:2px 9px; border-radius:99px; margin-right:6px; margin-top:10px; border:1px solid;
}
.badge-cu  { color:#f59e0b; border-color:#f59e0b44; background:#f59e0b10; }
.badge-iso { color:#34d399; border-color:#34d39944; background:#34d39910; }
.badge-slm { color:#818cf8; border-color:#818cf844; background:#818cf810; }

.mstrip { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:18px; }
.mc { background:#080e1a; border:1px solid #1e293b; border-radius:10px;
      padding:12px 18px; flex:1; min-width:110px; }
.mc .lbl { font-size:.68rem; color:#475569; letter-spacing:.1em; text-transform:uppercase; }
.mc .val { font-size:1.45rem; font-weight:700; color:#f1f5f9; font-family:'IBM Plex Mono',monospace; }
.mc .sub { font-size:.7rem; color:#334155; }

.sh { font-size:.7rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
      color:#334155; border-bottom:1px solid #1e293b; padding-bottom:4px; margin:16px 0 10px; }

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#1d4ed8,#7c3aed) !important;
    border:none !important; border-radius:8px !important;
    font-weight:600 !important; letter-spacing:.03em !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# CRYSTAL STRUCTURE LIBRARY
# ═══════════════════════════════════════════════════════════════════

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
    beta:  float = 90.0
    gamma: float = 90.0
    atoms: List[AtomSite] = field(default_factory=list)
    wf_init: float = 0.5
    color: str = "#60a5fa"
    group: str = "Primary"
    description: str = ""

    @property
    def volume(self) -> float:
        al, be, ga = map(np.radians, [self.alpha, self.beta, self.gamma])
        return self.a * self.b * self.c * np.sqrt(
            1 - np.cos(al)**2 - np.cos(be)**2 - np.cos(ga)**2
            + 2*np.cos(al)*np.cos(be)*np.cos(ga)
        )


def _build_phase_db() -> Dict[str, Phase]:
    db: Dict[str, Phase] = {}

    db["gamma_Co"] = Phase(
        key="gamma_Co", name="γ-Co  (FCC)", formula="Co",
        pdf_card="PDF 15-0806", crystal_system="cubic",
        space_group="Fm-3m", sg_number=225,
        a=3.5447, b=3.5447, c=3.5447,
        atoms=[AtomSite("Co", "4a", 0, 0, 0, 1.0, 0.40)],
        wf_init=0.70, color="#38bdf8", group="Primary",
        description="FCC cobalt — primary austenitic matrix in SLM Co-Cr.",
    )
    db["epsilon_Co"] = Phase(
        key="epsilon_Co", name="ε-Co  (HCP)", formula="Co",
        pdf_card="PDF 05-0727", crystal_system="hexagonal",
        space_group="P63/mmc", sg_number=194,
        a=2.5071, b=2.5071, c=4.0686,
        alpha=90, beta=90, gamma=120,
        atoms=[AtomSite("Co", "2c", 1/3, 2/3, 0.25, 1.0, 0.40)],
        wf_init=0.15, color="#fb923c", group="Primary",
        description="HCP cobalt — martensitic transform; stress indicator.",
    )
    db["sigma"] = Phase(
        key="sigma", name="σ-phase  (CoCr)", formula="CoCr",
        pdf_card="PDF 29-0490", crystal_system="tetragonal",
        space_group="P42/mnm", sg_number=136,
        a=8.7960, b=8.7960, c=4.5750,
        atoms=[
            AtomSite("Co", "2a", 0,     0,     0,     0.5, 0.50),
            AtomSite("Cr", "2a", 0,     0,     0,     0.5, 0.50),
            AtomSite("Co", "4f", 0.398, 0.398, 0,     0.5, 0.50),
            AtomSite("Cr", "4f", 0.398, 0.398, 0,     0.5, 0.50),
            AtomSite("Co", "8i", 0.464, 0.132, 0,     0.5, 0.50),
            AtomSite("Cr", "8i", 0.464, 0.132, 0,     0.5, 0.50),
        ],
        wf_init=0.05, color="#4ade80", group="Secondary",
        description="Brittle tetragonal intermetallic; appears after aging 700-900 °C.",
    )
    db["Cr_bcc"] = Phase(
        key="Cr_bcc", name="Cr  (BCC)", formula="Cr",
        pdf_card="PDF 06-0694", crystal_system="cubic",
        space_group="Im-3m", sg_number=229,
        a=2.8839, b=2.8839, c=2.8839,
        atoms=[AtomSite("Cr", "2a", 0, 0, 0, 1.0, 0.40)],
        wf_init=0.04, color="#f87171", group="Secondary",
        description="BCC chromium — excess Cr or incomplete alloying.",
    )
    db["Mo_bcc"] = Phase(
        key="Mo_bcc", name="Mo  (BCC)", formula="Mo",
        pdf_card="PDF 42-1120", crystal_system="cubic",
        space_group="Im-3m", sg_number=229,
        a=3.1472, b=3.1472, c=3.1472,
        atoms=[AtomSite("Mo", "2a", 0, 0, 0, 1.0, 0.45)],
        wf_init=0.03, color="#c084fc", group="Secondary",
        description="BCC molybdenum — inter-dendritic segregation.",
    )
    db["Co3Mo"] = Phase(
        key="Co3Mo", name="Co₃Mo", formula="Co3Mo",
        pdf_card="PDF 29-0491", crystal_system="hexagonal",
        space_group="P63/mmc", sg_number=194,
        a=5.1400, b=5.1400, c=4.1000,
        alpha=90, beta=90, gamma=120,
        atoms=[
            AtomSite("Co", "6h", 1/6, 1/3, 0.25, 1.0, 0.50),
            AtomSite("Mo", "2c", 1/3, 2/3, 0.25, 1.0, 0.55),
        ],
        wf_init=0.02, color="#a78bfa", group="Secondary",
        description="Hexagonal Co₃Mo — high-T annealing precipitate.",
    )
    db["Cr2O3"] = Phase(
        key="Cr2O3", name="Cr₂O₃  (Eskolaite)", formula="Cr2O3",
        pdf_card="PDF 38-1479", crystal_system="trigonal",
        space_group="R-3m", sg_number=167,
        a=4.9580, b=4.9580, c=13.5942,
        alpha=90, beta=90, gamma=120,
        atoms=[
            AtomSite("Cr", "12c", 0,     0,     0.348, 1.0, 0.55),
            AtomSite("O",  "18e", 0.306, 0,     0.25,  1.0, 0.60),
        ],
        wf_init=0.02, color="#f472b6", group="Oxide",
        description="Chromium sesquioxide — passive oxide layer.",
    )
    db["CoCr2O4"] = Phase(
        key="CoCr2O4", name="CoCr₂O₄  (Spinel)", formula="CoCr2O4",
        pdf_card="PDF 22-1084", crystal_system="cubic",
        space_group="Fm-3m", sg_number=227,
        a=8.3216, b=8.3216, c=8.3216,
        atoms=[
            AtomSite("Co", "8a",  0.125, 0.125, 0.125, 1.0, 0.55),
            AtomSite("Cr", "16d", 0.5,   0.5,   0.5,   1.0, 0.55),
            AtomSite("O",  "32e", 0.264, 0.264, 0.264, 1.0, 0.65),
        ],
        wf_init=0.01, color="#22d3ee", group="Oxide",
        description="Cobalt-chromium spinel oxide.",
    )
    return db


PHASE_DB: Dict[str, Phase] = _build_phase_db()
PRIMARY_KEYS   = ["gamma_Co", "epsilon_Co"]
SECONDARY_KEYS = ["sigma", "Cr_bcc", "Mo_bcc", "Co3Mo"]
OXIDE_KEYS     = ["Cr2O3", "CoCr2O4"]


# ═══════════════════════════════════════════════════════════════════
# CRYSTALLOGRAPHY UTILITIES
# ═══════════════════════════════════════════════════════════════════

def _d_cubic(a, h, k, l):
    s = h*h + k*k + l*l
    return a / np.sqrt(s) if s else np.inf

def _d_hex(a, c, h, k, l):
    t = (4/3)*((h*h + h*k + k*k) / a**2) + (l/c)**2
    return 1/np.sqrt(t) if t > 0 else np.inf

def _d_tet(a, c, h, k, l):
    t = (h*h + k*k) / a**2 + l*l / c**2
    return 1/np.sqrt(t) if t > 0 else np.inf

def _allow_fcc(h, k, l): return len({h%2, k%2, l%2}) == 1
def _allow_bcc(h, k, l): return (h+k+l) % 2 == 0
def _allow_hcp(h, k, l): return not (l%2 != 0 and (h-k)%3 == 0)
def _allow_sig(h, k, l): return (h+k+l) % 2 == 0
def _allow_all(h, k, l): return True

_ALLOW = {
    "Fm-3m":   _allow_fcc,
    "Im-3m":   _allow_bcc,
    "P63/mmc": _allow_hcp,
    "P42/mnm": _allow_sig,
    "R-3m":    _allow_all,
    "Pm-3m":   _allow_all,
}

_CM: Dict[str, Tuple] = {
    "Co": ([2.7686,2.2087,1.6079,1.0000],[14.178,3.398,0.124,41.698],0.9768),
    "Cr": ([2.3070,2.2940,0.8167,0.0000],[10.798,1.173,11.002,132.79],1.1003),
    "Mo": ([3.7025,2.3517,1.5442,0.8534],[12.943,2.658,0.157,39.714],0.6670),
    "O":  ([0.4548,0.9177,0.4719,0.0000],[23.780,7.622,0.165,0.000], 0.0000),
    "W":  ([4.000, 3.000, 2.000, 1.000], [10.0,  3.0,  0.5, 50.0],  0.5000),
}

def _f0(el: str, stl: float) -> float:
    if el not in _CM:
        return max({"Co":27,"Cr":24,"Mo":42,"O":8}.get(el, 20) - stl*4, 1.0)
    a, b, c = _CM[el]
    return c + sum(ai * np.exp(-bi * stl**2) for ai, bi in zip(a, b))


def _calc_d(ph: Phase, h: int, k: int, l: int) -> float:
    cs = ph.crystal_system.lower()
    if cs == "cubic":
        return _d_cubic(ph.a, h, k, l)
    elif cs in ("hexagonal", "trigonal"):
        return _d_hex(ph.a, ph.c, h, k, l)
    elif cs == "tetragonal":
        return _d_tet(ph.a, ph.c, h, k, l)
    return _d_cubic(ph.a, h, k, l)


def _F2(ph: Phase, h: int, k: int, l: int, wl: float = 1.54056) -> float:
    d = _calc_d(ph, h, k, l)
    stl = 1.0/(2.0*d) if d > 0 else 0.0
    Fr = Fi = 0.0
    for at in ph.atoms:
        f  = _f0(at.element, stl)
        DW = np.exp(-at.Biso * stl**2)
        pa = 2*np.pi*(h*at.x + k*at.y + l*at.z)
        Fr += at.occupancy * f * DW * np.cos(pa)
        Fi += at.occupancy * f * DW * np.sin(pa)
    return Fr*Fr + Fi*Fi


def generate_reflections(ph: Phase, wl: float = 1.54056,
                         tt_min: float = 10.0, tt_max: float = 100.0,
                         n: int = 7) -> List[Dict]:
    """Return list of allowed reflections sorted by 2θ."""
    afn = _ALLOW.get(ph.space_group, _allow_all)
    seen: Dict[float, Dict] = {}
    for h in range(-n, n+1):
        for k in range(-n, n+1):
            for l in range(-n, n+1):
                if h == k == l == 0:
                    continue
                if not afn(h, k, l):
                    continue
                d = _calc_d(ph, h, k, l)
                if d < 0.5 or d > 20:
                    continue
                st = wl / (2.0*d)
                if abs(st) > 1:
                    continue
                tt = 2.0 * np.degrees(np.arcsin(st))
                if not (tt_min <= tt <= tt_max):
                    continue
                dk = round(d, 4)
                if dk in seen:
                    seen[dk]["mult"] += 1
                    seen[dk]["hkl_list"].append((h, k, l))
                else:
                    seen[dk] = {"h":h, "k":k, "l":l,
                                "hkl_list":[(h,k,l)],
                                "d":d, "tt":tt, "mult":1}
    return sorted(seen.values(), key=lambda x: x["tt"])


def _make_refined_phase(ph: Phase, a_ref: float, c_ref: float) -> Phase:
    """Return a copy of ph with refined lattice parameters a and c."""
    return Phase(
        key=ph.key, name=ph.name, formula=ph.formula,
        pdf_card=ph.pdf_card, crystal_system=ph.crystal_system,
        space_group=ph.space_group, sg_number=ph.sg_number,
        a=a_ref, b=(a_ref if ph.b == ph.a else ph.b), c=c_ref,
        alpha=ph.alpha, beta=ph.beta, gamma=ph.gamma,
        atoms=ph.atoms, color=ph.color,
    )


# ═══════════════════════════════════════════════════════════════════
# PROFILE & BACKGROUND FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def pseudo_voigt(tt: np.ndarray, tt_k: float, fwhm: float, eta: float) -> np.ndarray:
    eta = np.clip(eta, 0.0, 1.0)
    dx  = tt - tt_k
    sig = fwhm / (2.0*np.sqrt(2.0*np.log(2.0)))
    G   = np.exp(-0.5*(dx/sig)**2) / (sig*np.sqrt(2.0*np.pi))
    gam = fwhm / 2.0
    L   = (1.0/np.pi) * gam / (dx**2 + gam**2)
    return eta*L + (1.0-eta)*G


def caglioti(tt_deg: float, U: float, V: float, W: float) -> float:
    th  = np.radians(tt_deg / 2.0)
    tan = np.tan(th)
    return np.sqrt(max(U*tan**2 + V*tan + W, 1e-8))


def lp_factor(tt_deg: float) -> float:
    th  = np.radians(tt_deg / 2.0)
    c2t = np.cos(2.0*th)
    c2m = np.cos(np.radians(26.6))
    den = np.sin(th)**2 * np.cos(th)
    return (1.0 + c2t**2 * c2m**2) / den if den > 0 else 1.0


def chebyshev_bg(tt: np.ndarray, coeffs: np.ndarray,
                 tt0: float, tt1: float) -> np.ndarray:
    x  = 2.0*(tt - tt0)/(tt1 - tt0) - 1.0
    bg = np.zeros_like(tt)
    Tp = np.ones_like(x)
    Tc = x.copy()
    if len(coeffs) > 0: bg += coeffs[0] * Tp
    if len(coeffs) > 1: bg += coeffs[1] * Tc
    for c in coeffs[2:]:
        Tn = 2.0*x*Tc - Tp
        bg += c * Tn
        Tp, Tc = Tc, Tn
    return bg


def phase_pattern(tt: np.ndarray, ph: Phase,
                  a: float, c: float, scale: float,
                  U: float, V: float, W: float, eta0: float,
                  z_shift: float, wl: float) -> np.ndarray:
    """Compute calculated intensity contribution of one phase."""
    ph_r  = _make_refined_phase(ph, a, c)
    refls = generate_reflections(
        ph_r, wl=wl,
        tt_min=max(float(tt.min()) - 5.0, 0.1),
        tt_max=float(tt.max()) + 5.0,
    )
    avg_biso = np.mean([at.Biso for at in ph.atoms]) if ph.atoms else 0.5
    I = np.zeros_like(tt)
    for r in refls:
        tt_k = r["tt"] + z_shift
        F2   = _F2(ph_r, r["h"], r["k"], r["l"], wl)
        lp   = lp_factor(tt_k)
        fwhm = caglioti(tt_k, U, V, W)
        stl  = np.sin(np.radians(tt_k / 2.0)) / wl
        DW   = np.exp(-avg_biso * stl**2)
        I   += scale * r["mult"] * F2 * lp * DW * pseudo_voigt(tt, tt_k, fwhm, eta0)
    return I


# ═══════════════════════════════════════════════════════════════════
# PARAMETER VECTOR
# ═══════════════════════════════════════════════════════════════════
N_PP = 7  # params per phase: scale,a,c,U,V,W,eta

def _pack(z, bg, per_phase) -> np.ndarray:
    return np.array([z, *bg, *[v for p in per_phase for v in p]], dtype=float)

def _unpack(v: np.ndarray, n_bg: int, n_ph: int):
    z  = float(v[0])
    bg = v[1 : 1+n_bg]
    pp = [v[1+n_bg+i*N_PP : 1+n_bg+(i+1)*N_PP] for i in range(n_ph)]
    return z, bg, pp


# ═══════════════════════════════════════════════════════════════════
# HILL-HOWARD WEIGHT FRACTIONS
# ═══════════════════════════════════════════════════════════════════
_MASS = {"Co":58.933,"Cr":51.996,"Mo":95.950,"O":15.999,"W":183.84,"N":14.007}

def hill_howard(phases: List[Phase], pp: List[np.ndarray]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for ph, p in zip(phases, pp):
        scale   = float(p[0])
        uc_mass = sum(_MASS.get(at.element, 50.0) * at.occupancy for at in ph.atoms) or 1.0
        totals[ph.key] = scale * uc_mass * ph.volume
    gt = sum(totals.values()) or 1.0
    return {k: v/gt for k, v in totals.items()}


# ═══════════════════════════════════════════════════════════════════
# R-FACTORS
# ═══════════════════════════════════════════════════════════════════
def r_factors(I_obs, I_calc, w) -> Dict[str, float]:
    num  = float(np.sum(w * (I_obs - I_calc)**2))
    den  = float(np.sum(w * I_obs**2))
    Rwp  = np.sqrt(num/den)  if den > 0 else 99.0
    Rp   = float(np.sum(np.abs(I_obs - I_calc)) / np.sum(np.abs(I_obs)))
    chi2 = num / max(len(I_obs) - 1, 1)
    Re   = np.sqrt((len(I_obs) - 1) / den) if den > 0 else 1.0
    GOF  = float(Rwp / Re) if Re > 0 else 99.0
    return dict(Rwp=float(Rwp), Rp=float(Rp), chi2=float(chi2),
                Re=float(Re), GOF=float(GOF))


# ═══════════════════════════════════════════════════════════════════
# RIETVELD REFINER
# ═══════════════════════════════════════════════════════════════════
class RietveldRefiner:

    def __init__(self, tt: np.ndarray, I_obs: np.ndarray,
                 phase_keys: List[str],
                 wavelength: float = 1.54056,
                 n_bg: int = 5):
        self.tt     = tt.astype(float)
        self.Iobs   = np.maximum(I_obs.astype(float), 0.0)
        self.wl     = float(wavelength)
        self.n_bg   = int(n_bg)
        self.phases = [PHASE_DB[k] for k in phase_keys]
        self.n_ph   = len(self.phases)
        self.w      = 1.0 / np.maximum(self.Iobs, 1.0)
        self._init_x0()

    def _init_x0(self):
        Ipeak = float(np.percentile(self.Iobs, 95))
        Imin  = float(np.percentile(self.Iobs, 10))
        bg0   = [Imin] + [0.0]*(self.n_bg - 1)
        pp    = []
        for ph in self.phases:
            pp.append([ph.wf_init * Ipeak * 1e-4,
                       ph.a, ph.c, 0.02, -0.01, 0.005, 0.5])
        self.x0 = _pack(0.0, bg0, pp)

    def _calc(self, v: np.ndarray):
        z, bg_c, pp = _unpack(v, self.n_bg, self.n_ph)
        bg    = chebyshev_bg(self.tt, bg_c, self.tt.min(), self.tt.max())
        Icalc = bg.copy()
        contribs: Dict[str, np.ndarray] = {}
        for ph, p in zip(self.phases, pp):
            sc, a, c, U, V, W, et = (float(x) for x in p)
            Iph = phase_pattern(self.tt, ph, a, c, sc, U, V, W, et, z, self.wl)
            contribs[ph.key] = Iph
            Icalc = Icalc + Iph
        return Icalc, bg, contribs

    def _res(self, v):
        Icalc, _, _ = self._calc(v)
        return np.sqrt(self.w) * (self.Iobs - Icalc)

    def _bounds(self, flags: Dict[str, bool]):
        n  = len(self.x0)
        lo = np.full(n, -np.inf)
        hi = np.full(n,  np.inf)
        x  = self.x0

        def freeze(i):
            lo[i] = x[i] - 1e-10
            hi[i] = x[i] + 1e-10

        def free(i, lb, ub):
            lo[i] = lb
            hi[i] = ub

        if flags.get("zero", False):
            free(0, -1.0, 1.0)
        else:
            freeze(0)

        for j in range(1, 1+self.n_bg):
            if flags.get("bg", True):
                free(j, -1e7, 1e7)
            else:
                freeze(j)

        for i, ph in enumerate(self.phases):
            b = 1 + self.n_bg + i*N_PP
            if flags.get("scale",   True): free(b,   0.0,       1e12)
            else:                          freeze(b)
            if flags.get("lattice", True): free(b+1, ph.a*0.95, ph.a*1.05)
            else:                          freeze(b+1)
            if flags.get("lattice", True): free(b+2, ph.c*0.95, ph.c*1.05)
            else:                          freeze(b+2)
            if flags.get("profile", True):
                free(b+3, 0.0,  0.5)
                free(b+4, -0.1, 0.0)
                free(b+5, 1e-4, 0.1)
                free(b+6, 0.0,  1.0)
            else:
                for j in range(3, 7):
                    freeze(b+j)

        return lo, hi

    def refine(self, flags: Dict[str, bool], max_iter: int = 400) -> Dict:
        lo, hi = self._bounds(flags)
        mask   = (lo == hi)
        hi[mask] += 1e-9

        try:
            res = least_squares(
                self._res, self.x0,
                bounds=(lo, hi), method="trf",
                max_nfev=max_iter,
                ftol=1e-7, xtol=1e-7, gtol=1e-7,
                verbose=0,
            )
            self.x0 = res.x
        except Exception as e:
            st.warning(f"Optimisation note: {e}")

        Icalc, bg, contribs = self._calc(self.x0)
        rf = r_factors(self.Iobs, Icalc, self.w)

        z, bg_c, pp = _unpack(self.x0, self.n_bg, self.n_ph)
        wf = hill_howard(self.phases, pp)

        lat: Dict[str, Dict] = {}
        for ph, p in zip(self.phases, pp):
            sc, a, c, U, V, W, et = (float(x) for x in p)
            lat[ph.key] = {
                "a_init": ph.a, "c_init": ph.c,
                "a_ref":  a,    "c_ref":  c,
                "da": a - ph.a, "dc": c - ph.c,
                "U": U, "V": V, "W": W, "eta": et, "scale": sc,
            }

        return {
            **rf,
            "Icalc":    Icalc,
            "Ibg":      bg,
            "contribs": contribs,
            "diff":     self.Iobs - Icalc,
            "wf":       wf,
            "lat":      lat,
            "z_shift":  z,
        }


# ═══════════════════════════════════════════════════════════════════
# SYNTHETIC DEMO PATTERN
# ═══════════════════════════════════════════════════════════════════
@st.cache_data
def make_demo_pattern(noise: float = 0.025, seed: int = 7):
    rng = np.random.default_rng(seed)
    tt  = np.linspace(10, 100, 4500)
    wf_demo = {
        "gamma_Co":   0.68,
        "epsilon_Co": 0.15,
        "sigma":      0.08,
        "Cr_bcc":     0.05,
        "Mo_bcc":     0.04,
    }
    bg_c = np.array([280., -60., 25., -8., 4.])
    I    = chebyshev_bg(tt, bg_c, tt.min(), tt.max())
    for key, wf in wf_demo.items():
        ph = PHASE_DB[key]
        I += phase_pattern(tt, ph, ph.a, ph.c,
                           wf*7500, 0.025, -0.012, 0.006, 0.45, 0.0, 1.54056)
    I = np.maximum(I, 0.0)
    I = rng.poisson(I).astype(float)
    I += rng.normal(0, noise*I.max(), size=I.shape)
    return tt, np.maximum(I, 0.0)


# ═══════════════════════════════════════════════════════════════════
# FILE PARSER - SUPPORTS .ASC AND .XRDML
# ═══════════════════════════════════════════════════════════════════
def parse_file_content(content: str, filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse XRD file content and return 2θ, intensity arrays.
    Supports:
    - .ASC, .DAT, .TXT, .CSV: Two-column text (2θ, Intensity)
    - .XRDML: Panalytical XML format
    """
    name = filename.lower()
    
    # ── XRDML (Panalytical XML) Parser ──────────────────────────────
    if name.endswith(".xrdml"):
        try:
            root = ET.fromstring(content)
            # Find counts data
            counts_elem = root.find(".//{*}counts")
            if counts_elem is None:
                # Try without namespace
                counts_elem = root.find(".//counts")
            if counts_elem is None:
                raise ValueError("No <counts> node found in .xrdml file.")
            
            # Parse intensity values
            I = np.array(counts_elem.text.split(), dtype=float)
            
            # Find 2θ positions
            start_pos = end_pos = None
            for pos_elem in root.findall(".//{*}positions"):
                if "2Theta" in pos_elem.get("axis", ""):
                    try:
                        start_pos = float(pos_elem.find("{*}startPosition").text)
                        end_pos = float(pos_elem.find("{*}endPosition").text)
                    except (AttributeError, ValueError):
                        pass
            
            # Generate 2θ array
            if start_pos is not None and end_pos is not None:
                tt = np.linspace(start_pos, end_pos, len(I))
            else:
                # Fallback: assume standard range
                tt = np.linspace(10.0, 100.0, len(I))
            
            return tt, I
            
        except ET.ParseError as e:
            raise ValueError(f"XML parsing error in .xrdml file: {e}")
    
    # ── Two-column text format parser (.ASC, .DAT, .TXT, .CSV) ─────
    lines = [ln.strip() for ln in content.splitlines()
             if ln.strip() and ln.strip()[0] not in "#!/'\";"]
    
    data = []
    for ln in lines:
        # Handle comma or space/tab separated
        parts = ln.replace(",", " ").split()
        try:
            if len(parts) >= 2:
                two_theta = float(parts[0])
                intensity = float(parts[1])
                data.append((two_theta, intensity))
        except ValueError:
            continue  # Skip malformed lines
    
    if not data:
        raise ValueError("Cannot parse file — expected 2 columns: 2θ and Intensity.")
    
    arr = np.array(data)
    tt, I = arr[:, 0], arr[:, 1]
    
    # Convert radians to degrees if values are very small
    if tt.max() < 5:
        tt = np.degrees(tt)
    
    # Sort by 2θ if needed
    if not np.all(tt[:-1] <= tt[1:]):
        sort_idx = np.argsort(tt)
        tt, I = tt[sort_idx], I[sort_idx]
    
    return tt, I


def fetch_github_xrd(sample_name: str, file_ext: str = ".ASC") -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Fetch XRD data from GitHub repository.
    Returns: (two_theta_array, intensity_array, actual_filename_used)
    """
    if sample_name not in AVAILABLE_FILES:
        raise ValueError(f"Sample '{sample_name}' not found in available files.")
    
    possible_files = AVAILABLE_FILES[sample_name]
    
    # Try files with preferred extension first
    for filename in possible_files:
        if filename.endswith(file_ext):
            url = GITHUB_RAW_BASE + filename
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                content = response.text
                tt, I = parse_file_content(content, filename)
                return tt, I, filename
            except requests.RequestException as e:
                continue
            except Exception as e:
                continue
    
    # If preferred extension fails, try all possible files
    for filename in possible_files:
        url = GITHUB_RAW_BASE + filename
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            content = response.text
            tt, I = parse_file_content(content, filename)
            return tt, I, filename
        except requests.RequestException:
            continue
        except Exception:
            continue
    
    raise ValueError(f"Could not fetch any file for sample '{sample_name}'. "
                    f"Tried extensions: .ASC, .xrdml with filenames: {possible_files}")


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
def q_color(rwp: float) -> str:
    if rwp < 0.05: return "#4ade80"
    if rwp < 0.10: return "#fbbf24"
    return "#f87171"


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════
for _k in ("results", "refiner", "tt", "Iobs", "elapsed", "selected_sample", "source_info"):
    if _k not in st.session_state:
        st.session_state[_k] = None


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Setup")

    st.markdown('<div class="sh">📁 GitHub Repository Files</div>', unsafe_allow_html=True)
    
    # Dropdown selector for samples
    sample_options = list(AVAILABLE_FILES.keys())
    selected_sample = st.selectbox(
        "Select XRD Sample",
        options=sample_options,
        index=0,
        help="Choose a sample from the RETVIELD-XRD repository"
    )
    
    # File extension preference
    file_ext = st.radio(
        "Preferred file format",
        options=[".ASC", ".xrdml"],
        index=0,
        horizontal=True,
        help=".ASC = two-column text format | .XRDML = Panalytical XML"
    )
    
    fetch_btn = st.button("🔄 Load Selected File from GitHub", type="primary", use_container_width=True)
    
    tt_raw = I_raw = None
    source_info = ""
    
    if fetch_btn:
        with st.spinner(f"Fetching {selected_sample} from GitHub..."):
            try:
                tt_raw, I_raw, actual_file = fetch_github_xrd(selected_sample, file_ext)
                source_info = f"✓ Loaded: {actual_file} ({len(tt_raw)} pts, {tt_raw.min():.1f}°–{tt_raw.max():.1f}°)"
                st.success(source_info)
                st.session_state.selected_sample = selected_sample
                st.session_state.source_info = source_info
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Try switching file format or check your internet connection.")
    
    # Alternative data sources
    with st.expander("🔁 Alternative: Demo Pattern or Local Upload", expanded=False):
        src = st.radio("", ["Demo pattern (synthetic)", "Upload my XRD file"],
                       label_visibility="collapsed", key="alt_src")
        
        if src.startswith("Demo"):
            tt_raw, I_raw = make_demo_pattern()
            st.info("Synthetic SLM Co-Cr-Mo · Cu Kα₁")
            source_info = "Demo pattern loaded"
        else:
            up = st.file_uploader(
                "Drag & drop XRD file",
                type=["xy","dat","txt","csv","xrdml","asc"],
                label_visibility="collapsed",
            )
            if up:
                try:
                    content = up.read().decode("utf-8", errors="replace")
                    tt_raw, I_raw = parse_file_content(content, up.name)
                    source_info = f"✓ {up.name}: {len(tt_raw)} pts · {tt_raw.min():.1f}° – {tt_raw.max():.1f}°"
                    st.success(source_info)
                except Exception as e:
                    st.error(str(e))

    st.markdown('<div class="sh">⚙️ Instrument</div>', unsafe_allow_html=True)
    WL_OPTIONS = {
        "Cu Kα₁  (1.54056 Å)": 1.54056,
        "Cu Kα   (1.54184 Å)": 1.54184,
        "Mo Kα₁  (0.70932 Å)": 0.70932,
        "Ag Kα₁  (0.56087 Å)": 0.56087,
        "Co Kα₁  (1.78900 Å)": 1.78900,
    }
    wl_label   = st.selectbox("Wavelength", list(WL_OPTIONS.keys()), index=0)
    wavelength = WL_OPTIONS[wl_label]
    zero_seed  = st.slider("Zero-shift seed (°)", -1.0, 1.0, 0.0, 0.01)

    st.markdown('<div class="sh">📐 2θ Window</div>', unsafe_allow_html=True)
    tt_lo, tt_hi = st.slider("", 10.0, 120.0, (15.0, 95.0), 0.5)

    st.markdown('<div class="sh">🧊 Phase Selection</div>', unsafe_allow_html=True)
    sel_keys: List[str] = []
    for grp, keys, exp in [
        ("Primary phases",   PRIMARY_KEYS,   True),
        ("Secondary phases", SECONDARY_KEYS, True),
        ("Oxide phases",     OXIDE_KEYS,     False),
    ]:
        with st.expander(grp, expanded=exp):
            for k in keys:
                ph = PHASE_DB[k]
                default = k in (PRIMARY_KEYS + SECONDARY_KEYS[:2])
                if st.checkbox(f"{ph.name}  ·  {ph.formula}",
                               value=default, key=f"ck_{k}",
                               help=ph.description):
                    sel_keys.append(k)

    if not sel_keys:
        st.warning("⚠️ Select at least one phase.")

    st.markdown('<div class="sh">🔧 Refinement Flags</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    fl_scale   = c1.checkbox("Scale",      value=True)
    fl_lattice = c2.checkbox("Lattice",    value=True)
    fl_bg      = c1.checkbox("Background", value=True)
    fl_profile = c2.checkbox("Profile",    value=True)
    fl_zero    = st.checkbox("Zero-shift", value=False)
    n_bg       = st.slider("Background terms", 2, 8, 5)
    max_it     = st.slider("Max iterations",  50, 1000, 350, 50)

    st.markdown("")
    run = st.button(
        "▶  Run Rietveld Refinement",
        type="primary",
        use_container_width=True,
        disabled=(tt_raw is None or not sel_keys),
    )


# ═══════════════════════════════════════════════════════════════════
# HERO BANNER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🔬 Co-Cr Dental Alloy · Rietveld XRD Refinement</h1>
  <p>Full-profile Rietveld refinement for 3D-printed (SLM / DMLS) cobalt-chromium dental alloys<br>
     Phase identification · Weight fractions · Lattice parameters · Peak profile analysis</p>
  <span class="badge badge-cu">Cu Kα / Mo Kα / Co Kα / Ag Kα</span>
  <span class="badge badge-iso">ISO 22674 · ASTM F75</span>
  <span class="badge badge-slm">SLM · DMLS · Casting</span>
</div>
""", unsafe_allow_html=True)

# Show current sample info
if st.session_state.selected_sample and st.session_state.source_info:
    st.caption(f"📊 Current sample: **{st.session_state.selected_sample}** — {st.session_state.source_info}")


# ═══════════════════════════════════════════════════════════════════
# RUN REFINEMENT
# ═══════════════════════════════════════════════════════════════════
if run and tt_raw is not None and sel_keys:
    mask = (tt_raw >= tt_lo) & (tt_raw <= tt_hi)
    tt_c = tt_raw[mask]
    I_c  = I_raw[mask]

    if len(tt_c) < 50:
        st.error("Too few data points in selected 2θ range — widen the window.")
    else:
        prog = st.progress(0, "Initialising refiner …")
        t0   = time.time()

        refiner = RietveldRefiner(tt_c, I_c, sel_keys, wavelength, n_bg)
        refiner.x0[0] = float(zero_seed)
        prog.progress(15, "Running least-squares optimisation …")

        flags   = dict(scale=fl_scale, lattice=fl_lattice,
                       bg=fl_bg, profile=fl_profile, zero=fl_zero)
        results = refiner.refine(flags, max_iter=max_it)
        elapsed = time.time() - t0

        prog.progress(100, f"Done in {elapsed:.1f} s")
        time.sleep(0.3)
        prog.empty()

        st.session_state.update(
            results=results, refiner=refiner,
            tt=tt_c, Iobs=I_c, elapsed=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tab_fit, tab_phase, tab_peaks, tab_params, tab_report, tab_about = st.tabs([
    "📈 Pattern Fit",
    "⚖️ Phase Analysis",
    "📋 Peak List",
    "🔧 Refined Parameters",
    "📄 Report",
    "ℹ️ About",
])


# ───────────────────────────────────────────────────────────────────
# TAB 1 · PATTERN FIT
# ───────────────────────────────────────────────────────────────────
with tab_fit:
    if st.session_state["results"] is None:
        if tt_raw is not None:
            mask = (tt_raw >= tt_lo) & (tt_raw <= tt_hi)
            fig  = go.Figure(go.Scatter(
                x=tt_raw[mask], y=I_raw[mask],
                mode="lines", line=dict(color="#38bdf8", width=1), name="I_obs",
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#030712", plot_bgcolor="#030712",
                xaxis_title="2θ (°)", yaxis_title="Intensity (counts)",
                height=350, margin=dict(l=60,r=20,t=20,b=50),
            )
            st.plotly_chart(fig, use_container_width=True)
        st.info("👈 Select a file from the dropdown above and press **▶ Run Rietveld Refinement**.")
    else:
        r       = st.session_state["results"]
        refiner = st.session_state["refiner"]
        tt      = st.session_state["tt"]
        Iobs    = st.session_state["Iobs"]
        elapsed = st.session_state["elapsed"]
        z_shift = float(r.get("z_shift", 0.0))
        _, _, pp_vec = _unpack(refiner.x0, refiner.n_bg, refiner.n_ph)

        # Metric cards
        rwp = r["Rwp"]; rp = r["Rp"]; gof = r["GOF"]; chi2 = r["chi2"]
        qc  = q_color(rwp)
        st.markdown(f"""
        <div class="mstrip">
          <div class="mc"><div class="lbl">R_wp</div>
            <div class="val" style="color:{qc}">{rwp*100:.2f}</div>
            <div class="sub">% &nbsp;(target &lt; 10 %)</div></div>
          <div class="mc"><div class="lbl">R_p</div>
            <div class="val">{rp*100:.2f}</div><div class="sub">%</div></div>
          <div class="mc"><div class="lbl">GOF</div>
            <div class="val">{gof:.3f}</div><div class="sub">target ≈ 1</div></div>
          <div class="mc"><div class="lbl">χ²</div>
            <div class="val">{chi2:.4f}</div><div class="sub"></div></div>
          <div class="mc"><div class="lbl">Points</div>
            <div class="val">{len(tt)}</div><div class="sub">data pts</div></div>
          <div class="mc"><div class="lbl">Time</div>
            <div class="val">{elapsed:.1f}</div><div class="sub">s</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Rietveld plot
        fig = make_subplots(
            rows=2, cols=1, row_heights=[0.78, 0.22],
            shared_xaxes=True, vertical_spacing=0.02,
        )

        fig.add_trace(go.Scatter(
            x=tt, y=Iobs, mode="lines", name="I_obs (exp)",
            line=dict(color="#94a3b8", width=1.3),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=tt, y=r["Ibg"], mode="lines", name="Background",
            line=dict(color="#334155", width=1, dash="dot"),
            fill="tozeroy", fillcolor="rgba(51,65,85,0.12)",
        ), row=1, col=1)

        for key, Iph in r["contribs"].items():
            ph  = PHASE_DB[key]
            wf  = r["wf"].get(key, 0) * 100
            fig.add_trace(go.Scatter(
                x=tt, y=Iph + r["Ibg"], mode="lines",
                name=f"{ph.name}  ({wf:.1f}%)",
                line=dict(color=ph.color, width=1.6, dash="dash"),
                opacity=0.8,
            ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=tt, y=r["Icalc"], mode="lines", name="I_calc",
            line=dict(color="#fbbf24", width=2.2),
        ), row=1, col=1)

        # hkl tick marks
        y_tick = float(Iobs.min()) - 0.05*(float(Iobs.max()) - float(Iobs.min()))
        for i, ph_obj in enumerate(refiner.phases):
            a_ref = float(pp_vec[i][1])
            c_ref = float(pp_vec[i][2])
            ph_ref = _make_refined_phase(ph_obj, a_ref, c_ref)
            pks = generate_reflections(
                ph_ref, wl=wavelength,
                tt_min=float(tt.min()), tt_max=float(tt.max()),
            )
            fig.add_trace(go.Scatter(
                x=[p["tt"] + z_shift for p in pks],
                y=[y_tick] * len(pks),
                mode="markers",
                marker=dict(symbol="line-ns", size=11, color=ph_obj.color,
                            line=dict(width=2, color=ph_obj.color)),
                name=f"{ph_obj.name} hkl",
                showlegend=False,
            ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=tt, y=r["diff"], mode="lines", name="Δ obs−calc",
            line=dict(color="#818cf8", width=1),
            fill="tozeroy", fillcolor="rgba(129,140,248,0.12)",
        ), row=2, col=1)
        fig.add_hline(y=0, line=dict(color="#334155", width=1, dash="dash"), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#030712", plot_bgcolor="#030712",
            height=650,
            legend=dict(bgcolor="#080e1a", bordercolor="#1e293b",
                        font=dict(size=11), x=1.01, y=1),
            margin=dict(l=65, r=210, t=15, b=55),
            font=dict(family="IBM Plex Sans"),
        )
        fig.update_xaxes(title_text="2θ (°)", row=2, col=1,
                         gridcolor="#0f172a", zerolinecolor="#0f172a")
        fig.update_yaxes(title_text="Intensity (counts)", row=1, col=1,
                         gridcolor="#0f172a", zerolinecolor="#0f172a")
        fig.update_yaxes(title_text="Δ", row=2, col=1,
                         gridcolor="#0f172a", zerolinecolor="#0f172a")
        st.plotly_chart(fig, use_container_width=True)

        df_pat = pd.DataFrame({
            "two_theta":    tt,
            "I_obs":        Iobs,
            "I_calc":       r["Icalc"],
            "I_background": r["Ibg"],
            "difference":   r["diff"],
            **{f"I_{k}": v for k, v in r["contribs"].items()},
        })
        st.download_button("⬇ Download pattern CSV",
                           data=df_pat.to_csv(index=False),
                           file_name="rietveld_pattern.csv", mime="text/csv")


# ───────────────────────────────────────────────────────────────────
# TAB 2 · PHASE ANALYSIS
# ───────────────────────────────────────────────────────────────────
with tab_phase:
    if st.session_state["results"] is None:
        st.info("Run refinement first.")
    else:
        r        = st.session_state["results"]
        wf_dict  = r["wf"]
        ph_names = [PHASE_DB[k].name  for k in wf_dict]
        wf_pct   = [wf_dict[k]*100    for k in wf_dict]
        ph_cols  = [PHASE_DB[k].color for k in wf_dict]

        c1, c2 = st.columns(2)

        with c1:
            fig_pie = go.Figure(go.Pie(
                labels=ph_names, values=wf_pct, hole=0.58,
                marker=dict(colors=ph_cols, line=dict(color="#030712", width=2.5)),
                textinfo="label+percent",
                textfont=dict(size=11.5, family="IBM Plex Sans"),
                hovertemplate="<b>%{label}</b><br>%{value:.2f} wt%<extra></extra>",
            ))
            fig_pie.add_annotation(
                text="Weight<br>Fraction", x=0.5, y=0.5,
                font=dict(size=12, color="#64748b"), showarrow=False,
            )
            fig_pie.update_layout(
                template="plotly_dark", paper_bgcolor="#030712",
                showlegend=False, height=360,
                margin=dict(l=10,r=10,t=30,b=10),
                title=dict(text="Phase Composition", font=dict(size=13, color="#64748b")),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            idx = list(np.argsort(wf_pct)[::-1])
            fig_bar = go.Figure(go.Bar(
                x=[wf_pct[i] for i in idx],
                y=[ph_names[i] for i in idx],
                orientation="h",
                marker=dict(color=[ph_cols[i] for i in idx]),
                text=[f"{wf_pct[i]:.2f} %" for i in idx],
                textposition="inside",
                insidetextfont=dict(color="white", size=11, family="IBM Plex Mono"),
            ))
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="#030712", plot_bgcolor="#030712",
                xaxis_title="wt %", height=360,
                margin=dict(l=10,r=10,t=30,b=40),
                title=dict(text="Phase Distribution", font=dict(size=13, color="#64748b")),
                yaxis=dict(gridcolor="#0f172a"),
                xaxis=dict(gridcolor="#0f172a", range=[0, max(wf_pct)*1.15]),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('<div class="sh">Phase Detail</div>', unsafe_allow_html=True)
        lat  = r["lat"]
        rows = []
        for k, wf in wf_dict.items():
            ph = PHASE_DB[k]
            lp = lat.get(k, {})
            rows.append({
                "Phase":         ph.name,
                "Formula":       ph.formula,
                "PDF Card":      ph.pdf_card,
                "System":        ph.crystal_system.title(),
                "S.G.":          ph.space_group,
                "wt %":          f"{wf*100:.2f}",
                "a ref (Å)":     f"{lp.get('a_ref', ph.a):.4f}",
                "c ref (Å)":     f"{lp.get('c_ref', ph.c):.4f}" if ph.c != ph.a else "—",
                "Group":         ph.group,
            })
        df_ph = pd.DataFrame(rows)
        st.dataframe(df_ph, use_container_width=True, hide_index=True)

        with st.expander("🧪 Microstructural Interpretation", expanded=True):
            wm = {k: wf_dict.get(k, 0)*100 for k in PHASE_DB}
            msgs = []
            if wm["gamma_Co"] > 50:
                msgs.append(("✅","γ-Co (FCC)",f"{wm['gamma_Co']:.1f} wt%",
                    "Dominant austenitic matrix — typical of rapid-solidification SLM. "
                    "Good biocompatibility and ductility for dental prostheses."))
            if wm["epsilon_Co"] > 5:
                msgs.append(("⚠️","ε-Co (HCP)",f"{wm['epsilon_Co']:.1f} wt%",
                    "Elevated HCP fraction — martensitic transformation driven by residual stress "
                    "or strain. Monitor fatigue performance."))
            if wm["sigma"] > 3:
                msgs.append(("🔴","σ-phase",f"{wm['sigma']:.1f} wt%",
                    "Brittle intermetallic — risk to ductility and fracture toughness. "
                    "Consider solution anneal at >1100 °C + water quench."))
            if wm["Cr_bcc"] > 2:
                msgs.append(("🔵","Cr (BCC)",f"{wm['Cr_bcc']:.1f} wt%",
                    "Free chromium — may indicate incomplete alloying or Cr-rich segregation."))
            if wm["Mo_bcc"] > 1:
                msgs.append(("🟣","Mo (BCC)",f"{wm['Mo_bcc']:.1f} wt%",
                    "Segregated molybdenum — inter-dendritic artefact; "
                    "homogenisation anneal at 1150 °C / 1 h recommended."))
            if wm.get("Cr2O3",0) > 0.5 or wm.get("CoCr2O4",0) > 0.5:
                msgs.append(("🟠","Oxides",
                    f"Cr₂O₃={wm.get('Cr2O3',0):.1f}%  CoCr₂O₄={wm.get('CoCr2O4',0):.1f}%",
                    "Surface/internal oxide detected — check SLM atmosphere and powder storage."))
            if not msgs:
                msgs.append(("ℹ️","No dominant phases","—",
                             "Check phase selection and 2θ range."))
            for icon, name, pct, msg in msgs:
                st.markdown(f"""
                <div style="background:#080e1a;border:1px solid #1e293b;border-radius:10px;
                            padding:14px 18px;margin-bottom:10px;">
                  <div style="font-weight:700;font-size:.95rem;margin-bottom:4px;">
                    {icon} &nbsp;{name} &nbsp;
                    <span style="font-family:'IBM Plex Mono';font-size:.82rem;color:#64748b;">{pct}</span>
                  </div>
                  <div style="color:#94a3b8;font-size:.85rem;line-height:1.55;">{msg}</div>
                </div>""", unsafe_allow_html=True)

        st.download_button("⬇ Download phase table (.csv)",
                           data=df_ph.to_csv(index=False),
                           file_name="phase_analysis.csv", mime="text/csv")


# ───────────────────────────────────────────────────────────────────
# TAB 3 · PEAK LIST
# ───────────────────────────────────────────────────────────────────
with tab_peaks:
    if st.session_state["refiner"] is None:
        st.info("Run refinement first.")
    else:
        refiner = st.session_state["refiner"]
        tt      = st.session_state["tt"]
        r       = st.session_state["results"]
        z_shift = float(r.get("z_shift", 0.0))
        _, _, pp_vec = _unpack(refiner.x0, refiner.n_bg, refiner.n_ph)

        show = st.multiselect(
            "Phases to display",
            options=[ph.key for ph in refiner.phases],
            default=[ph.key for ph in refiner.phases],
            format_func=lambda k: PHASE_DB[k].name,
        )

        rows = []
        for i, ph_obj in enumerate(refiner.phases):
            if ph_obj.key not in show:
                continue
            a_ref  = float(pp_vec[i][1])
            c_ref  = float(pp_vec[i][2])
            ph_ref = _make_refined_phase(ph_obj, a_ref, c_ref)
            for ref in generate_reflections(
                ph_ref, wl=wavelength,
                tt_min=float(tt.min()), tt_max=float(tt.max()),
            ):
                h, k, l = ref["h"], ref["k"], ref["l"]
                rows.append({
                    "Phase":   ph_obj.name,
                    "hkl":     f"({h} {k} {l})",
                    "d (Å)":   f"{ref['d']:.4f}",
                    "2θ (°)":  f"{ref['tt'] + z_shift:.3f}",
                    "Mult.":   ref["mult"],
                    "Group":   ph_obj.group,
                })

        if rows:
            df_pk = (pd.DataFrame(rows)
                     .sort_values("2θ (°)")
                     .reset_index(drop=True))
            st.dataframe(df_pk, use_container_width=True, hide_index=True, height=500)
            st.download_button("⬇ Download peak list (.csv)",
                               data=df_pk.to_csv(index=False),
                               file_name="peak_list.csv", mime="text/csv")
        else:
            st.warning("No peaks found for selected phases in this 2θ range.")


# ───────────────────────────────────────────────────────────────────
# TAB 4 · REFINED PARAMETERS
# ───────────────────────────────────────────────────────────────────
with tab_params:
    if st.session_state["results"] is None:
        st.info("Run refinement first.")
    else:
        r       = st.session_state["results"]
        refiner = st.session_state["refiner"]
        lat     = r["lat"]
        tt      = st.session_state["tt"]

        st.markdown('<div class="sh">Lattice Parameters</div>', unsafe_allow_html=True)
        lp_rows = []
        for ph_obj in refiner.phases:
            lp = lat.get(ph_obj.key, {})
            lp_rows.append({
                "Phase":      ph_obj.name,
                "System":     ph_obj.crystal_system.title(),
                "S.G.":       ph_obj.space_group,
                "a_init (Å)": f"{lp.get('a_init', ph_obj.a):.4f}",
                "a_ref (Å)":  f"{lp.get('a_ref',  ph_obj.a):.4f}",
                "Δa (Å)":     f"{lp.get('da', 0):+.4f}",
                "c_init (Å)": f"{lp.get('c_init', ph_obj.c):.4f}" if ph_obj.c != ph_obj.a else "—",
                "c_ref (Å)":  f"{lp.get('c_ref',  ph_obj.c):.4f}" if ph_obj.c != ph_obj.a else "—",
                "Δc (Å)":     f"{lp.get('dc', 0):+.4f}"           if ph_obj.c != ph_obj.a else "—",
            })
        st.dataframe(pd.DataFrame(lp_rows), use_container_width=True, hide_index=True)

        st.markdown('<div class="sh">Profile Parameters (U, V, W, η)</div>',
                    unsafe_allow_html=True)
        prof_rows = []
        for ph_obj in refiner.phases:
            lp = lat.get(ph_obj.key, {})
            prof_rows.append({
                "Phase":      ph_obj.name,
                "Scale":      f"{lp.get('scale', 0):.4e}",
                "U":          f"{lp.get('U', 0):.5f}",
                "V":          f"{lp.get('V', 0):.5f}",
                "W":          f"{lp.get('W', 0):.5f}",
                "η (mixing)": f"{lp.get('eta', 0.5):.3f}",
            })
        st.dataframe(pd.DataFrame(prof_rows), use_container_width=True, hide_index=True)

        st.markdown('<div class="sh">Global Parameters</div>', unsafe_allow_html=True)
        z_val, bg_c, _ = _unpack(refiner.x0, refiner.n_bg, refiner.n_ph)
        st.code(
            f"Zero-shift  : {z_val:+.4f} °\n"
            f"Wavelength  : {wavelength:.5f} Å\n"
            f"Background  : {' '.join(f'{float(v):.2f}' for v in bg_c)}"
            f"  (Chebyshev coeff. 0–{len(bg_c)-1})",
            language="text",
        )

        # FWHM vs 2θ
        st.markdown('<div class="sh">FWHM vs 2θ</div>', unsafe_allow_html=True)
        tt_plot = np.linspace(float(tt.min()), float(tt.max()), 300)
        fig_fw  = go.Figure()
        for ph_obj in refiner.phases:
            lp = lat.get(ph_obj.key, {})
            U, V, W = lp.get("U",.02), lp.get("V",-.01), lp.get("W",.005)
            fw = [caglioti(t, U, V, W) for t in tt_plot]
            fig_fw.add_trace(go.Scatter(
                x=tt_plot, y=fw, mode="lines",
                name=ph_obj.name, line=dict(color=ph_obj.color, width=2),
            ))
        fig_fw.update_layout(
            template="plotly_dark",
            paper_bgcolor="#030712", plot_bgcolor="#030712",
            xaxis_title="2θ (°)", yaxis_title="FWHM (°)",
            height=270, margin=dict(l=60,r=20,t=10,b=50),
            legend=dict(bgcolor="#080e1a", bordercolor="#1e293b", font=dict(size=10)),
            xaxis=dict(gridcolor="#0f172a"),
            yaxis=dict(gridcolor="#0f172a"),
        )
        st.plotly_chart(fig_fw, use_container_width=True)


# ───────────────────────────────────────────────────────────────────
# TAB 5 · REPORT
# ───────────────────────────────────────────────────────────────────
with tab_report:
    if st.session_state["results"] is None:
        st.info("Run refinement first.")
    else:
        r       = st.session_state["results"]
        refiner = st.session_state["refiner"]
        elapsed = st.session_state["elapsed"]
        tt      = st.session_state["tt"]

        md = f"""# Rietveld Refinement Report
## Co-Cr Dental Alloy · XRD Phase Analysis

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
**Radiation:** {wl_label}  ·  λ = {wavelength:.5f} Å
**Data points:** {len(tt)}
**2θ range:** {tt.min():.2f}° – {tt.max():.2f}°
**Computation time:** {elapsed:.2f} s

---

## 1 · Refinement Quality

| Indicator | Value | Guidance |
|-----------|-------|----------|
| R_wp | **{r['Rwp']*100:.2f} %** | < 10 % acceptable; < 5 % excellent |
| R_p  | **{r['Rp']*100:.2f} %**  | — |
| χ²   | **{r['chi2']:.4f}** | — |
| GOF  | **{r['GOF']:.4f}** | ≈ 1 ideal |

---

## 2 · Phase Weight Fractions (Hill-Howard)

| Phase | Formula | S.G. | wt % |
|-------|---------|------|------|
"""
        for k, wf in r["wf"].items():
            ph = PHASE_DB[k]
            md += f"| {ph.name} | {ph.formula} | {ph.space_group} | **{wf*100:.2f}** |\n"

        md += "\n---\n\n## 3 · Refined Lattice Parameters\n\n"
        md += "| Phase | a_init (Å) | a_ref (Å) | Δa | c_init (Å) | c_ref (Å) | Δc |\n"
        md += "|-------|-----------|-----------|-----|-----------|-----------|----|\n"
        for k, lp in r["lat"].items():
            ph = PHASE_DB[k]
            if ph.c != ph.a:
                md += (f"| {ph.name} | {lp['a_init']:.4f} | {lp['a_ref']:.4f} | "
                       f"{lp['da']:+.4f} | {lp['c_init']:.4f} | {lp['c_ref']:.4f} | "
                       f"{lp['dc']:+.4f} |\n")
            else:
                md += (f"| {ph.name} | {lp['a_init']:.4f} | {lp['a_ref']:.4f} | "
                       f"{lp['da']:+.4f} | — | — | — |\n")

        md += "\n---\n\n## 4 · Profile Parameters\n\n"
        md += "| Phase | Scale | U | V | W | η |\n"
        md += "|-------|-------|---|---|---|---|\n"
        for k, lp in r["lat"].items():
            ph = PHASE_DB[k]
            md += (f"| {ph.name} | {lp['scale']:.3e} | {lp['U']:.5f} "
                   f"| {lp['V']:.5f} | {lp['W']:.5f} | {lp['eta']:.3f} |\n")

        md += f"""
---

## 5 · Methodology

- **Profile:** Thompson-Cox-Hastings pseudo-Voigt · Caglioti FWHM (U, V, W, η)
- **Background:** Chebyshev polynomial ({n_bg} terms)
- **LP correction:** Graphite monochromator (2θ_mono = 26.6°)
- **Structure factors:** Cromer-Mann + isotropic Debye-Waller
- **Weight fractions:** Hill-Howard formula
- **Optimiser:** scipy.optimize.least_squares (TRF / Levenberg-Marquardt · {max_it} iter.)

---

## 6 · References

1. Rietveld (1969) *J. Appl. Cryst.* **2**, 65–71
2. Hill & Howard (1987) *J. Appl. Cryst.* **20**, 467–474
3. Thompson et al. (1987) *J. Appl. Cryst.* **20**, 79–83
4. Takaichi et al. (2013) *Acta Biomaterialia* **9**, 7901–7910
5. ISO 22674:2016 · ASTM F75-18
"""
        st.markdown(md)
        st.download_button("⬇ Download Report (.md)",
                           data=md, file_name="rietveld_report.md",
                           mime="text/markdown")


# ───────────────────────────────────────────────────────────────────
# TAB 6 · ABOUT
# ───────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
## About

Full-profile **Rietveld refinement** for X-ray diffraction patterns from
**3D-printed (SLM/DMLS) Co-Cr dental alloys** — entirely in the browser,
no crystallography software required.

---

### ✨ GitHub Integration

- **Dropdown selector** for samples: `CH0`, `CH45`, `CNH0`, `CNH45`, `PH0`, `PH45`, `PNH0`, `PNH45`
- **Auto-fetches** `.ASC` or `.XRDML` files from: `Maryamslm/RETVIELD-XRD`
- **Fallback** to local upload or synthetic demo pattern

---

### 📚 Phase Library

| Phase | Formula | Space Group | Group |
|-------|---------|-------------|-------|
| γ-Co (FCC) | Co | Fm-3m | Primary |
| ε-Co (HCP) | Co | P6₃/mmc | Primary |
| σ-phase | CoCr | P4₂/mnm | Secondary |
| Cr (BCC) | Cr | Im-3m | Secondary |
| Mo (BCC) | Mo | Im-3m | Secondary |
| Co₃Mo | Co₃Mo | P6₃/mmc | Secondary |
| Cr₂O₃ | Cr₂O₃ | R-3m | Oxide |
| CoCr₂O₄ | CoCr₂O₄ | Fm-3m | Oxide |

---

### 🔬 Algorithm
