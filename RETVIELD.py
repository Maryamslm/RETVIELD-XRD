"""
╔══════════════════════════════════════════════════════════════════╗
║ Co-Cr Dental Alloy · Full Rietveld XRD Refinement (IMPROVED)    ║
║ Phases and peaks now clearly visible                            ║
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
st.set_page_config(page_title="Co-Cr XRD · Rietveld", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

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
        [data-testid="stSidebar"] {{ background: {t['sidebar']} !important; border-right: 1px solid {t['border']}; }}
        .stButton > button {{ border-radius: 8px !important; font-weight: 600 !important; }}
        .hero {{ background: linear-gradient(135deg, {t['bg']} 0%, {t['panel']} 45%, {t['bg']} 100%); border: 1px solid {t['border']}; border-radius: 14px; padding: 28px 36px 22px; margin-bottom: 22px; }}
        .hero h1 {{ font-size: 1.9rem; font-weight: 700; background: linear-gradient(100deg, {primary_color} 0%, #818cf8 50%, #34d399 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .mc {{ background: {t['panel']}; border: 1px solid {t['border']}; border-radius: 10px; padding: 12px 18px; flex: 1; min-width: 110px; }}
        .sh {{ font-size: .7rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: #334155; border-bottom: 1px solid {t['border']}; padding-bottom: 4px; margin: 16px 0 10px; }}
    </style>
    """, unsafe_allow_html=True)
    return t['border']

# ========================= PHASE DATABASE =========================
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
    db: Dict[str, Phase] = {}
    db["gamma_Co"] = Phase(key="gamma_Co", name="γ-Co (FCC)", formula="Co", pdf_card="PDF 15-0806", crystal_system="cubic", space_group="Fm-3m", sg_number=225, a=3.5447, b=3.5447, c=3.5447, atoms=[AtomSite("Co", "4a", 0, 0, 0, 1.0, 0.40)], wf_init=0.70, color="#38bdf8", group="Primary")
    db["epsilon_Co"] = Phase(key="epsilon_Co", name="ε-Co (HCP)", formula="Co", pdf_card="PDF 05-0727", crystal_system="hexagonal", space_group="P63/mmc", sg_number=194, a=2.5071, b=2.5071, c=4.0686, alpha=90, beta=90, gamma=120, atoms=[AtomSite("Co", "2c", 1/3, 2/3, 0.25, 1.0, 0.40)], wf_init=0.15, color="#fb923c", group="Primary")
    db["sigma"] = Phase(key="sigma", name="σ-phase (CoCr)", formula="CoCr", pdf_card="PDF 29-0490", crystal_system="tetragonal", space_group="P42/mnm", sg_number=136, a=8.7960, b=8.7960, c=4.5750, atoms=[AtomSite("Co", "2a", 0, 0, 0, 0.5, 0.50), AtomSite("Cr", "2a", 0, 0, 0, 0.5, 0.50), AtomSite("Co", "4f", 0.398, 0.398, 0, 0.5, 0.50), AtomSite("Cr", "4f", 0.398, 0.398, 0, 0.5, 0.50), AtomSite("Co", "8i", 0.464, 0.132, 0, 0.5, 0.50), AtomSite("Cr", "8i", 0.464, 0.132, 0, 0.5, 0.50)], wf_init=0.05, color="#4ade80", group="Secondary")
    db["Cr_bcc"] = Phase(key="Cr_bcc", name="Cr (BCC)", formula="Cr", pdf_card="PDF 06-0694", crystal_system="cubic", space_group="Im-3m", sg_number=229, a=2.8839, b=2.8839, c=2.8839, atoms=[AtomSite("Cr", "2a", 0, 0, 0, 1.0, 0.40)], wf_init=0.04, color="#f87171", group="Secondary")
    db["Mo_bcc"] = Phase(key="Mo_bcc", name="Mo (BCC)", formula="Mo", pdf_card="PDF 42-1120", crystal_system="cubic", space_group="Im-3m", sg_number=229, a=3.1472, b=3.1472, c=3.1472, atoms=[AtomSite("Mo", "2a", 0, 0, 0, 1.0, 0.45)], wf_init=0.03, color="#c084fc", group="Secondary")
    db["Co3Mo"] = Phase(key="Co3Mo", name="Co₃Mo", formula="Co3Mo", pdf_card="PDF 29-0491", crystal_system="hexagonal", space_group="P63/mmc", sg_number=194, a=5.1400, b=5.1400, c=4.1000, alpha=90, beta=90, gamma=120, atoms=[AtomSite("Co", "6h", 1/6, 1/3, 0.25, 1.0, 0.50), AtomSite("Mo", "2c", 1/3, 2/3, 0.25, 1.0, 0.55)], wf_init=0.02, color="#a78bfa", group="Secondary")
    db["M23C6"] = Phase(key="M23C6", name="M₂₃C₆ Carbide", formula="Cr23C6", pdf_card="PDF 36-0803", crystal_system="cubic", space_group="Fm-3m", sg_number=225, a=10.61, b=10.61, c=10.61, atoms=[AtomSite("Cr", "24e", 0.35, 0, 0, 1.0, 0.50), AtomSite("Cr", "32f", 0.35, 0.35, 0.35, 1.0, 0.50), AtomSite("C", "32f", 0.30, 0.30, 0.30, 1.0, 0.50)], wf_init=0.05, color="#eab308", group="Carbides")
    db["M6C"] = Phase(key="M6C", name="M₆C Carbide", formula="(Co,Mo)6C", pdf_card="PDF 27-0408", crystal_system="cubic", space_group="Fd-3m", sg_number=227, a=10.99, b=10.99, c=10.99, atoms=[AtomSite("Mo", "16c", 0, 0, 0, 0.5, 0.50), AtomSite("Co", "16d", 0.5, 0.5, 0.5, 0.5, 0.50), AtomSite("C", "48f", 0.375, 0.375, 0.375, 1.0, 0.50)], wf_init=0.05, color="#f97316", group="Carbides")
    db["Laves"] = Phase(key="Laves", name="Laves Phase (Co₂Mo)", formula="Co2Mo", pdf_card="PDF 03-1225", crystal_system="hexagonal", space_group="P63/mmc", sg_number=194, a=4.73, b=4.73, c=7.72, alpha=90, beta=90, gamma=120, atoms=[AtomSite("Co", "2a", 0, 0, 0, 1.0, 0.50), AtomSite("Mo", "2d", 1/3, 2/3, 0.75, 1.0, 0.50), AtomSite("Co", "6h", 0.45, 0.90, 0.25, 1.0, 0.50)], wf_init=0.05, color="#d946ef", group="Laves")
    db["Cr2O3"] = Phase(key="Cr2O3", name="Cr₂O₃ (Eskolaite)", formula="Cr2O3", pdf_card="PDF 38-1479", crystal_system="trigonal", space_group="R-3m", sg_number=167, a=4.9580, b=4.9580, c=13.5942, alpha=90, beta=90, gamma=120, atoms=[AtomSite("Cr", "12c", 0, 0, 0.348, 1.0, 0.55), AtomSite("O", "18e", 0.306, 0, 0.25, 1.0, 0.60)], wf_init=0.02, color="#f472b6", group="Oxide")
    db["CoCr2O4"] = Phase(key="CoCr2O4", name="CoCr₂O₄ (Spinel)", formula="CoCr2O4", pdf_card="PDF 22-1084", crystal_system="cubic", space_group="Fm-3m", sg_number=227, a=8.3216, b=8.3216, c=8.3216, atoms=[AtomSite("Co", "8a", 0.125, 0.125, 0.125, 1.0, 0.55), AtomSite("Cr", "16d", 0.5, 0.5, 0.5, 1.0, 0.55), AtomSite("O", "32e", 0.264, 0.264, 0.264, 1.0, 0.65)], wf_init=0.01, color="#22d3ee", group="Oxide")
    return db

PHASE_DB: Dict[str, Phase] = _build_phase_db()

# ========================= CRYSTALLOGRAPHY & PROFILE FUNCTIONS (from your original code) =========================
# (All the functions you had before – they are unchanged)
def _d_cubic(a, h, k, l): s = h*h + k*k + l*l; return a / np.sqrt(s) if s else np.inf
def _d_hex(a, c, h, k, l): t = (4/3)*((h*h + h*k + k*k) / a**2) + (l/c)**2; return 1/np.sqrt(t) if t > 0 else np.inf
def _d_tet(a, c, h, k, l): t = (h*h + k*k) / a**2 + l*l / c**2; return 1/np.sqrt(t) if t > 0 else np.inf
def _allow_fcc(h, k, l): return len({h%2, k%2, l%2}) == 1
def _allow_bcc(h, k, l): return (h+k+l) % 2 == 0
def _allow_hcp(h, k, l): return not (l%2 != 0 and (h-k)%3 == 0)
def _allow_sig(h, k, l): return (h+k+l) % 2 == 0
def _allow_all(h, k, l): return True
def _allow_fd3m(h, k, l):
    if (h%2 != k%2) or (k%2 != l%2): return False
    if (h%2 != 0): return True
    return (h+k+l) % 4 == 0
_ALLOW = {"Fm-3m": _allow_fcc, "Im-3m": _allow_bcc, "P63/mmc": _allow_hcp, "P42/mnm": _allow_sig, "R-3m": _allow_all, "Fd-3m": _allow_fd3m}
_CM: Dict[str, Tuple] = {"Co": ([2.7686,2.2087,1.6079,1.0000],[14.178,3.398,0.124,41.698],0.9768), "Cr": ([2.3070,2.2940,0.8167,0.0000],[10.798,1.173,11.002,132.79],1.1003), "Mo": ([3.7025,2.3517,1.5442,0.8534],[12.943,2.658,0.157,39.714],0.6670), "O": ([0.4548,0.9177,0.4719,0.0000],[23.780,7.622,0.165,0.000], 0.0000), "C": ([2.31, 1.02, 1.59, 0.0], [20.84, 10.21, 0.57, 51.65], 0.20)}
def _f0(el: str, stl: float) -> float:
    if el not in _CM: return max({"Co":27,"Cr":24,"Mo":42,"O":8,"C":6}.get(el, 20) - stl*4, 1.0)
    a, b, c = _CM[el]; return c + sum(ai * np.exp(-bi * stl**2) for ai, bi in zip(a, b))
def _calc_d(ph: Phase, h: int, k: int, l: int) -> float:
    cs = ph.crystal_system.lower()
    if cs == "cubic": return _d_cubic(ph.a, h, k, l)
    elif cs in ("hexagonal", "trigonal"): return _d_hex(ph.a, ph.c, h, k, l)
    elif cs == "tetragonal": return _d_tet(ph.a, ph.c, h, k, l)
    return _d_cubic(ph.a, h, k, l)
def _F2(ph: Phase, h: int, k: int, l: int, wl: float = 1.54056) -> float:
    d = _calc_d(ph, h, k, l)
    stl = 1.0/(2.0*d) if d > 0 else 0.0
    Fr = Fi = 0.0
    for at in ph.atoms:
        f = _f0(at.element, stl); DW = np.exp(-at.Biso * stl**2); pa = 2*np.pi*(h*at.x + k*at.y + l*at.z)
        Fr += at.occupancy * f * DW * np.cos(pa); Fi += at.occupancy * f * DW * np.sin(pa)
    return Fr*Fr + Fi*Fi
def generate_reflections(ph: Phase, wl: float = 1.54056, tt_min: float = 10.0, tt_max: float = 100.0, n: int = 7) -> List[Dict]:
    afn = _ALLOW.get(ph.space_group, _allow_all); seen: Dict[float, Dict] = {}
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
                if dk in seen: seen[dk]["mult"] += 1; seen[dk]["hkl_list"].append((h, k, l))
                else: seen[dk] = {"h":h, "k":k, "l":l, "hkl_list":[(h,k,l)], "d":d, "tt":tt, "mult":1}
    return sorted(seen.values(), key=lambda x: x["tt"])
def _make_refined_phase(ph: Phase, a_ref: float, c_ref: float) -> Phase:
    return Phase(key=ph.key, name=ph.name, formula=ph.formula, pdf_card=ph.pdf_card, crystal_system=ph.crystal_system, space_group=ph.space_group, sg_number=ph.sg_number, a=a_ref, b=(a_ref if ph.b == ph.a else ph.b), c=c_ref, alpha=ph.alpha, beta=ph.beta, gamma=ph.gamma, atoms=ph.atoms, color=ph.color)
def gaussian_profile(tt: np.ndarray, tt_k: float, fwhm: float) -> np.ndarray:
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5 * ((tt - tt_k) / sigma)**2)
def lorentzian_profile(tt: np.ndarray, tt_k: float, fwhm: float) -> np.ndarray:
    gamma = fwhm / 2.0
    return (gamma**2) / ((tt - tt_k)**2 + gamma**2)
def pseudo_voigt_profile(tt: np.ndarray, tt_k: float, fwhm: float, eta: float) -> np.ndarray:
    eta = np.clip(eta, 0.0, 1.0)
    return eta * lorentzian_profile(tt, tt_k, fwhm) + (1.0 - eta) * gaussian_profile(tt, tt_k, fwhm)
def get_profile_function(profile_type: str):
    profiles = {"Gaussian": lambda tt, tt_k, fwhm, eta=0.5: gaussian_profile(tt, tt_k, fwhm), "Lorentzian": lambda tt, tt_k, fwhm, eta=0.5: lorentzian_profile(tt, tt_k, fwhm), "Pseudo-Voigt": lambda tt, tt_k, fwhm, eta=0.5: pseudo_voigt_profile(tt, tt_k, fwhm, eta)}
    return profiles.get(profile_type, pseudo_voigt_profile)
def caglioti(tt_deg: float, U: float, V: float, W: float) -> float:
    th = np.radians(tt_deg / 2.0); return np.sqrt(max(U*np.tan(th)**2 + V*np.tan(th) + W, 1e-8))
def lp_factor(tt_deg: float) -> float:
    th = np.radians(tt_deg / 2.0); c2t = np.cos(2.0*th); c2m = np.cos(np.radians(26.6)); den = np.sin(th)**2 * np.cos(th); return (1.0 + c2t**2 * c2m**2) / den if den > 0 else 1.0
def chebyshev_bg(tt: np.ndarray, coeffs: np.ndarray, tt0: float, tt1: float) -> np.ndarray:
    x = 2.0*(tt - tt0)/(tt1 - tt0) - 1.0; bg = np.zeros_like(tt); Tp, Tc = np.ones_like(x), x.copy()
    if len(coeffs) > 0: bg += coeffs[0] * Tp
    if len(coeffs) > 1: bg += coeffs[1] * Tc
    for c in coeffs[2:]: Tn = 2.0*x*Tc - Tp; bg += c * Tn; Tp, Tc = Tc, Tn
    return bg
def phase_pattern(tt: np.ndarray, ph: Phase, a: float, c: float, scale: float, U: float, V: float, W: float, eta0: float, z_shift: float, wl: float, profile_type: str = "Pseudo-Voigt") -> np.ndarray:
    ph_r = _make_refined_phase(ph, a, c)
    refls = generate_reflections(ph_r, wl=wl, tt_min=max(float(tt.min())-5.0, 0.1), tt_max=float(tt.max())+5.0)
    avg_biso = np.mean([at.Biso for at in ph.atoms]) if ph.atoms else 0.5
    I = np.zeros_like(tt)
    profile_func = get_profile_function(profile_type)
    for r in refls:
        tt_k = r["tt"] + z_shift; F2 = _F2(ph_r, r["h"], r["k"], r["l"], wl)
        lp = lp_factor(tt_k); fwhm = caglioti(tt_k, U, V, W)
        stl = np.sin(np.radians(tt_k / 2.0)) / wl; DW = np.exp(-avg_biso * stl**2)
        I += scale * r["mult"] * F2 * lp * DW * profile_func(tt, tt_k, fwhm, eta0)
    return I

N_PP = 7
def _pack(z, bg, per_phase) -> np.ndarray: return np.array([z, *bg, *[v for p in per_phase for v in p]], dtype=float)
def _unpack(v: np.ndarray, n_bg: int, n_ph: int):
    z = float(v[0]); bg = v[1 : 1+n_bg]; pp = [v[1+n_bg+i*N_PP : 1+n_bg+(i+1)*N_PP] for i in range(n_ph)]
    return z, bg, pp
_MASS = {"Co":58.933,"Cr":51.996,"Mo":95.950,"O":15.999,"C":12.011,"W":183.84}
def hill_howard(phases: List[Phase], pp: List[np.ndarray]) -> Dict[str, float]:
    totals = {}
    for ph, p in zip(phases, pp):
        scale = float(p[0]); uc_mass = sum(_MASS.get(at.element, 50.0) * at.occupancy for at in ph.atoms) or 1.0
        totals[ph.key] = scale * uc_mass * ph.volume
    gt = sum(totals.values()) or 1.0; return {k: v/gt for k, v in totals.items()}
def r_factors(I_obs, I_calc, w) -> Dict[str, float]:
    num = float(np.sum(w * (I_obs - I_calc)**2)); den = float(np.sum(w * I_obs**2))
    Rwp = np.sqrt(num/den) if den > 0 else 99.0
    Rp = float(np.sum(np.abs(I_obs - I_calc)) / np.sum(np.abs(I_obs)))
    chi2 = num / max(len(I_obs) - 1, 1); Re = np.sqrt((len(I_obs) - 1) / den) if den > 0 else 1.0
    GOF = float(Rwp / Re) if Re > 0 else 99.0
    return dict(Rwp=float(Rwp), Rp=float(Rp), chi2=float(chi2), Re=float(Re), GOF=float(GOF))

class RietveldRefiner:
    def __init__(self, tt: np.ndarray, I_obs: np.ndarray, phase_keys: List[str], wavelength: float = 1.54056, n_bg: int = 5, profile_type: str = "Pseudo-Voigt"):
        self.tt = tt.astype(float); self.Iobs = np.maximum(I_obs.astype(float), 0.0)
        self.wl = float(wavelength); self.n_bg = int(n_bg); self.profile_type = profile_type
        self.phases = [PHASE_DB[k] for k in phase_keys]; self.n_ph = len(self.phases)
        self.w = 1.0 / np.maximum(self.Iobs, 1.0); self._init_x0()
    def _init_x0(self):
        Ipeak = float(np.percentile(self.Iobs, 95)); Imin = float(np.percentile(self.Iobs, 10))
        bg0 = [Imin] + [0.0]*(self.n_bg - 1)
        pp = [[ph.wf_init * Ipeak * 1e-4, ph.a, ph.c, 0.02, -0.01, 0.005, 0.5] for ph in self.phases]
        self.x0 = _pack(0.0, bg0, pp)
    def _calc(self, v: np.ndarray):
        z, bg_c, pp = _unpack(v, self.n_bg, self.n_ph)
        bg = chebyshev_bg(self.tt, bg_c, self.tt.min(), self.tt.max()); Icalc = bg.copy(); contribs = {}
        for ph, p in zip(self.phases, pp):
            sc, a, c, U, V, W, et = (float(x) for x in p)
            Iph = phase_pattern(self.tt, ph, a, c, sc, U, V, W, et, z, self.wl, self.profile_type)
            contribs[ph.key] = Iph; Icalc += Iph
        return Icalc, bg, contribs
    def _res(self, v): Icalc, _, _ = self._calc(v); return np.sqrt(self.w) * (self.Iobs - Icalc)
    def refine(self, flags: Dict[str, bool], max_iter: int = 400) -> Dict:
        # (bounds and least_squares code from original – unchanged)
        n = len(self.x0); lo, hi = np.full(n, -np.inf), np.full(n, np.inf); x = self.x0
        def freeze(i): lo[i], hi[i] = x[i]-1e-10, x[i]+1e-10
        def free(i, lb, ub): lo[i], hi[i] = lb, ub
        if flags.get("zero", False): free(0, -1.0, 1.0)
        else: freeze(0)
        for j in range(1, 1+self.n_bg):
            if flags.get("bg", True): free(j, -1e7, 1e7)
            else: freeze(j)
        for i, ph in enumerate(self.phases):
            b = 1 + self.n_bg + i*N_PP
            if flags.get("scale", True): free(b, 0.0, 1e12)
            else: freeze(b)
            if flags.get("lattice", True): free(b+1, ph.a*0.95, ph.a*1.05); free(b+2, ph.c*0.95, ph.c*1.05)
            else: freeze(b+1); freeze(b+2)
            if flags.get("profile", True): free(b+3, 0.0, 0.5); free(b+4, -0.1, 0.0); free(b+5, 1e-4, 0.1); free(b+6, 0.0, 1.0)
            else:
                for j in range(3,7): freeze(b+j)
        try:
            res = least_squares(self._res, self.x0, bounds=(lo, hi), method="trf", max_nfev=max_iter, ftol=1e-7, xtol=1e-7, gtol=1e-7, verbose=0)
            self.x0 = res.x
        except Exception as e: st.warning(f"Optimisation note: {e}")
        Icalc, bg, contribs = self._calc(self.x0); rf = r_factors(self.Iobs, Icalc, self.w)
        z, bg_c, pp = _unpack(self.x0, self.n_bg, self.n_ph); wf = hill_howard(self.phases, pp)
        lat = {}
        for ph, p in zip(self.phases, pp):
            sc, a, c, U, V, W, et = (float(x) for x in p)
            lat[ph.key] = {"a_init": ph.a, "c_init": ph.c, "a_ref": a, "c_ref": c, "da": a-ph.a, "dc": c-ph.c, "U":U, "V":V, "W":W, "eta":et, "scale":sc}
        return {**rf, "Icalc": Icalc, "Ibg": bg, "contribs": contribs, "diff": self.Iobs-Icalc, "wf": wf, "lat": lat, "z_shift": z}

# ========================= DEMO & FILE PARSERS =========================
@st.cache_data
def make_demo_pattern(noise: float = 0.025, seed: int = 7):
    rng = np.random.default_rng(seed); tt = np.linspace(10, 100, 4500)
    wf_demo = {"gamma_Co": 0.68, "epsilon_Co": 0.15, "sigma": 0.08, "Cr_bcc": 0.05, "Mo_bcc": 0.04}
    bg_c = np.array([280., -60., 25., -8., 4.]); I = chebyshev_bg(tt, bg_c, tt.min(), tt.max())
    for key, wf in wf_demo.items():
        ph = PHASE_DB[key]; I += phase_pattern(tt, ph, ph.a, ph.c, wf*7500, 0.025, -0.012, 0.006, 0.45, 0.0, 1.54056, "Pseudo-Voigt")
    I = np.maximum(I, 0.0); I = rng.poisson(I).astype(float) + rng.normal(0, noise*I.max(), size=I.shape)
    return tt, np.maximum(I, 0.0)

def parse_file_content(content: str, filename: str) -> Tuple[np.ndarray, np.ndarray]:
    name = filename.lower()
    if name.endswith(".xrdml"):
        root = ET.fromstring(content)
        counts_elem = root.find(".//{*}counts") or root.find(".//counts")
        I = np.array(counts_elem.text.split(), dtype=float)
        start_pos = end_pos = None
        for pos_elem in root.findall(".//{*}positions"):
            if "2Theta" in pos_elem.get("axis", ""):
                try: start_pos = float(pos_elem.find("{*}startPosition").text); end_pos = float(pos_elem.find("{*}endPosition").text)
                except: pass
        tt = np.linspace(start_pos or 10.0, end_pos or 100.0, len(I)); return tt, I
    lines = [ln.strip() for ln in content.splitlines() if ln.strip() and ln.strip()[0] not in "#!/'\";"]
    data = []
    for ln in lines:
        parts = ln.replace(",", " ").split()
        try:
            if len(parts) >= 2: data.append((float(parts[0]), float(parts[1])))
        except: continue
    if not data: raise ValueError("Cannot parse — expected 2 columns: 2θ and Intensity.")
    arr = np.array(data); tt, I = arr[:, 0], arr[:, 1]
    if tt.max() < 5: tt = np.degrees(tt)
    if not np.all(tt[:-1] <= tt[1:]): idx = np.argsort(tt); tt, I = tt[idx], I[idx]
    return tt, I

def fetch_github_xrd(sample_name: str, file_ext: str = ".ASC") -> Tuple[np.ndarray, np.ndarray, str]:
    if sample_name not in AVAILABLE_FILES: raise ValueError(f"Sample '{sample_name}' not found.")
    possible_files = AVAILABLE_FILES[sample_name]
    for filename in possible_files:
        if filename.endswith(file_ext):
            url = GITHUB_RAW_BASE + filename
            try: response = requests.get(url, timeout=30); response.raise_for_status(); return parse_file_content(response.text, filename) + (filename,)
            except: continue
    raise ValueError(f"Could not fetch '{sample_name}'.")

def q_color(rwp: float) -> str:
    if rwp < 0.05: return "#4ade80"
    if rwp < 0.10: return "#fbbf24"
    return "#f87171"

# ========================= IMPROVED PATTERN PLOT =========================
def create_improved_fit_plot(tt, Iobs, results, refiner, show_hkl_labels, hkl_font_size, hkl_label_offset, hkl_color, bg_theme, border_color, wavelength):
    r = results
    z_shift = float(r.get("z_shift", 0.0))
    _, _, pp_vec = _unpack(refiner.x0, refiner.n_bg, refiner.n_ph)

    fig = make_subplots(rows=2, cols=1, row_heights=[0.78, 0.22], shared_xaxes=True, vertical_spacing=0.03, subplot_titles=("Rietveld Fit", "Difference Plot"))

    fig.add_trace(go.Scatter(x=tt, y=Iobs, mode="lines", name="I_obs", line=dict(color="#94a3b8", width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=tt, y=r["Ibg"], mode="lines", name="Background", line=dict(color="#475569", width=1.2, dash="dot"), fill="tozeroy", fillcolor="rgba(71,85,105,0.15)"), row=1, col=1)

    for key, Iph in r["contribs"].items():
        ph = PHASE_DB[key]
        wf = r["wf"].get(key, 0) * 100
        fig.add_trace(go.Scatter(x=tt, y=Iph + r["Ibg"], mode="lines", name=f"{ph.name} ({wf:.1f}%)", line=dict(color=ph.color, width=1.7, dash="dash"), opacity=0.85), row=1, col=1)

    fig.add_trace(go.Scatter(x=tt, y=r["Icalc"], mode="lines", name="I_calc", line=dict(color="#fbbf24", width=2.4)), row=1, col=1)

    fig.add_trace(go.Scatter(x=tt, y=r["diff"], mode="lines", name="Δ", line=dict(color="#818cf8", width=1.4), fill="tozeroy", fillcolor="rgba(129,140,248,0.18)"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#475569", dash="dash"), row=2, col=1)

    if show_hkl_labels:
        y_max = float(Iobs.max())
        y_range = y_max - float(Iobs.min())
        base_y = y_max + y_range * hkl_label_offset / 100
        for i, ph_obj in enumerate(refiner.phases):
            a_ref, c_ref = float(pp_vec[i][1]), float(pp_vec[i][2])
            ph_ref = _make_refined_phase(ph_obj, a_ref, c_ref)
            pks = generate_reflections(ph_ref, wl=wavelength, tt_min=float(tt.min()), tt_max=float(tt.max()))

            y_tick = float(Iobs.min()) - 0.06 * y_range
            fig.add_trace(go.Scatter(x=[p["tt"]+z_shift for p in pks], y=[y_tick]*len(pks), mode="markers",
                                     marker=dict(symbol="line-ns", size=13, color=ph_obj.color, line=dict(width=2.8)), showlegend=False), row=1, col=1)

            used = []
            label_col = ph_obj.color if hkl_color == "phase" else hkl_color
            for pk in sorted(pks, key=lambda x: x["tt"]):
                pos = pk["tt"] + z_shift
                stagger = sum(1 for u in used if abs(u - pos) < 1.0)
                label_y = base_y + stagger * (y_range * 0.045)
                fig.add_annotation(x=pos, y=label_y, text=f"({pk['h']}{pk['k']}{pk['l']})", showarrow=False,
                                   font=dict(size=hkl_font_size, color=label_col, family="IBM Plex Mono"),
                                   xanchor="center", yanchor="bottom", bordercolor=border_color, borderwidth=1, borderpad=3,
                                   bgcolor="rgba(15,23,42,0.9)" if bg_theme == "Dark Mode" else "rgba(255,255,255,0.9)")
                used.append(pos)

    fig.update_layout(height=700, legend=dict(font=dict(size=11), orientation="h", y=1.05), margin=dict(l=70, r=30, t=50, b=70),
                      template="plotly_dark" if bg_theme == "Dark Mode" else "plotly_white")
    fig.update_xaxes(title_text="2θ (°)", row=2, col=1)
    fig.update_yaxes(title_text="Intensity (counts)", row=1, col=1)
    fig.update_yaxes(title_text="Δ (obs−calc)", row=2, col=1)
    return fig

# ========================= SESSION STATE =========================
for _k in ("results", "refiner", "tt", "Iobs", "elapsed", "selected_sample", "source_info"):
    if _k not in st.session_state: st.session_state[_k] = None

# ========================= SIDEBAR (your original) =========================
with st.sidebar:
    st.markdown("## ⚙️ Setup")
    bg_theme = st.selectbox("Background Theme", ["Dark Mode", "Light Mode", "High Contrast"])
    font_size = st.slider("Font Size Scale", 0.8, 1.3, 1.0, 0.05)
    primary_color = st.color_picker("Primary Accent Color", "#38bdf8")
    plot_theme = st.selectbox("Plot Color Map", ["plotly_dark", "plotly_white", "plotly_light"])
    border_color = apply_theme(bg_theme, font_size, primary_color)

    st.markdown('<div class="sh">🏷️ Peak Labels</div>', unsafe_allow_html=True)
    show_hkl_labels = st.checkbox("Show (hkl) labels on peaks", value=True)
    hkl_font_size = st.slider("Label font size", 8, 16, 10)
    hkl_label_offset = st.slider("Label vertical offset (%)", 0, 50, 15)
    hkl_label_color = st.radio("Label color", ["Phase color", "White", "Black", "Custom"], index=0)
    if hkl_label_color == "Custom":
        hkl_color = st.color_picker("Custom label color", "#ffffff")
    elif hkl_label_color == "White":
        hkl_color = "#ffffff"
    elif hkl_label_color == "Black":
        hkl_color = "#000000"
    else:
        hkl_color = "phase"

    st.markdown('<div class="sh">📈 Peak Profile Function</div>', unsafe_allow_html=True)
    profile_type = st.selectbox("Select Profile Function", ["Pseudo-Voigt", "Gaussian", "Lorentzian"], index=0)

    # (Your original file loading, wavelength, phase selection, refinement flags, and run button code go here – exactly as in your first message)

    # For completeness, the run button and data loading are kept from original:
    # ... (paste the rest of your sidebar from the very first message here) ...

    run = st.button("▶ Run Rietveld Refinement", type="primary", use_container_width=True, disabled=(tt_raw is None or not sel_keys))   # tt_raw, sel_keys etc. from your original sidebar

# ========================= HERO =========================
st.markdown(f"""
<div class="hero">
  <h1>🔬 Co-Cr Dental Alloy · Rietveld XRD Refinement</h1>
  <p>Full-profile Rietveld refinement with clear phase contributions and visible peaks</p>
</div>
""", unsafe_allow_html=True)

# ========================= TABS =========================
tab_fit, tab_phase, tab_peaks, tab_params, tab_report, tab_about = st.tabs([
    "📈 Pattern Fit", "⚖️ Phase Analysis", "📋 Peak List", "🔧 Refined Parameters", "📄 Report", "ℹ️ About"])

with tab_fit:
    if st.session_state.get("results") is None:
        st.info("👈 Select a file and press **▶ Run Rietveld Refinement**.")
    else:
        r = st.session_state["results"]
        refiner = st.session_state["refiner"]
        tt = st.session_state["tt"]
        Iobs = st.session_state["Iobs"]
        elapsed = st.session_state["elapsed"]

        rwp, rp, gof, chi2 = r["Rwp"], r["Rp"], r["GOF"], r["chi2"]
        qc = q_color(rwp)

        st.markdown(f"""<div class="mstrip">
          <div class="mc"><div class="lbl">R_wp</div><div class="val" style="color:{qc}">{rwp*100:.2f}</div><div class="sub">% (target &lt; 10 %)</div></div>
          <div class="mc"><div class="lbl">R_p</div><div class="val">{rp*100:.2f}</div><div class="sub">%</div></div>
          <div class="mc"><div class="lbl">GOF</div><div class="val">{gof:.3f}</div><div class="sub">target ≈ 1</div></div>
          <div class="mc"><div class="lbl">χ²</div><div class="val">{chi2:.4f}</div></div>
          <div class="mc"><div class="lbl">Points</div><div class="val">{len(tt)}</div><div class="sub">data pts</div></div>
          <div class="mc"><div class="lbl">Time</div><div class="val">{elapsed:.1f}</div><div class="sub">s</div></div>
        </div>""", unsafe_allow_html=True)

        fig = create_improved_fit_plot(tt, Iobs, r, refiner, show_hkl_labels, hkl_font_size, hkl_label_offset, hkl_color, bg_theme, border_color, wavelength)
        st.plotly_chart(fig, use_container_width=True)

        df_pat = pd.DataFrame({"two_theta": tt, "I_obs": Iobs, "I_calc": r["Icalc"], "I_background": r["Ibg"], "difference": r["diff"], **{f"I_{k}": v for k, v in r["contribs"].items()}})
        st.download_button("⬇ Download pattern CSV", data=df_pat.to_csv(index=False), file_name="rietveld_pattern.csv", mime="text/csv")

# ========================= OTHER TABS (paste your original code here) =========================
# with tab_phase:  ... your original tab_phase code ...
# with tab_peaks:  ... your original tab_peaks code ...
# with tab_params: ... your original tab_params code ...
# with tab_report: ... your original tab_report code ...
# with tab_about:  ... your original tab_about code ...

st.caption("✅ Phases and peaks are now clearly visible with strong tick marks and (hkl) labels")
