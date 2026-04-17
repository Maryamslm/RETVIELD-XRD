"""
XRD Rietveld Profile Fitting — Co-Cr Dental Alloy
Tailored for CH0_1.ASC  (2θ: 30–130°, Cu Kα, step 0.026°)
Deploy: GitHub + Streamlit Community Cloud
MODIFIED: Show ALL matching phases on peaks + Font size + 2θ range controls
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import UnivariateSpline
import pandas as pd
import io

# ──────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XRD Rietveld — Co-Cr Dental Alloy",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 XRD Rietveld Profile Fitting")
st.caption("Co-Cr Dental Alloy · Cu Kα radiation · 2θ range 30–130°")

# ──────────────────────────────────────────────────────────
# Known phases for Co-Cr dental alloys (Cu Kα, λ=1.54056 Å)
# ──────────────────────────────────────────────────────────
PHASES = {
    "γ-Co (FCC)": {
        "color": "#1565C0",
        "peaks": [
            (43.70, "111"), (50.83, "200"), (74.78, "220"),
            (90.61, "311"), (95.93, "222"),
        ],
        "description": "FCC Co-Cr solid solution — dominant phase in dental alloys",
        "space_group": "Fm3̄m",
        "a_ang": 3.548,
    },
    "ε-Co (HCP)": {
        "color": "#C62828",
        "peaks": [
            (41.02, "100"), (44.20, "002"), (47.30, "101"),
            (55.10, "102"), (62.15, "110"),
        ],
        "description": "HCP cobalt — metastable phase, forms under stress/cooling",
        "space_group": "P6₃/mmc",
        "a_ang": 2.507,
    },
    "Cr₂₃C₆ carbide": {
        "color": "#2E7D32",
        "peaks": [
            (37.20, "511"), (43.43, "600"), (63.11, "622"),
            (75.85, "731"), (80.28, "800"),
        ],
        "description": "Chromium carbide — precipitates at grain boundaries",
        "space_group": "Fm3̄m",
        "a_ang": 10.659,
    },
    "Mo₂C / Mo-rich": {
        "color": "#6A1B9A",
        "peaks": [
            (34.40, "100"), (38.05, "002"), (39.48, "101"),
            (60.90, "102"), (69.95, "110"),
        ],
        "description": "Molybdenum carbide / Mo-rich phase (common in dental Co-Cr-Mo)",
        "space_group": "P6₃/mmc",
        "a_ang": 3.002,
    },
    "σ-phase (CoCr)": {
        "color": "#E65100",
        "peaks": [
            (42.10, "330"), (45.30, "401"), (53.10, "411"),
            (65.60, "421"), (78.10, "510"),
        ],
        "description": "Intermetallic sigma phase — embrittles the alloy",
        "space_group": "P4₂/mnm",
        "a_ang": 8.800,
    },
}

DETECTED_PEAKS = [44.495, 48.161, 51.801, 55.155, 60.173, 73.589, 89.891, 101.331, 112.459, 115.345]

# ──────────────────────────────────────────────────────────
# Peak profile functions
# ──────────────────────────────────────────────────────────

def caglioti_fwhm(tth, U, V, W):
    t = np.tan(np.radians(tth / 2))
    val = U * t**2 + V * t + W
    return np.sqrt(np.maximum(val, 1e-6))

def gaussian(x, c, fwhm, A):
    s = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return A * np.exp(-0.5 * ((x - c) / s)**2)

def lorentzian(x, c, fwhm, A):
    g = fwhm / 2
    return A * g**2 / ((x - c)**2 + g**2)

def pseudo_voigt(x, c, fwhm, A, eta=0.5):
    return A * (eta * (fwhm/2)**2 / ((x-c)**2 + (fwhm/2)**2)
                + (1-eta) * np.exp(-4*np.log(2)*(x-c)**2/fwhm**2))

def profile(x, c, fwhm, A, shape, eta=0.5):
    if shape == "Gaussian":
        return gaussian(x, c, fwhm, A)
    elif shape == "Lorentzian":
        return lorentzian(x, c, fwhm, A)
    else:
        return pseudo_voigt(x, c, fwhm, A, eta)

def scherrer(fwhm_deg, tth_deg, wl=1.54056, K=0.94):
    beta = np.radians(fwhm_deg)
    theta = np.radians(tth_deg / 2)
    return (K * wl) / (beta * np.cos(theta)) / 10  # nm

def match_peak_to_all_phases(peaks_2theta, phases_dict, tolerance=0.5):
    """
    Match each detected peak to ALL phase peaks within tolerance.
    Returns: list of (detected_pk, [(phase_name, hkl, ref_pk, color, dist), ...])
    """
    matches = []
    for pk in peaks_2theta:
        phase_matches = []
        for pname, pinfo in phases_dict.items():
            for (ref_pk, hkl) in pinfo["peaks"]:
                dist = abs(pk - ref_pk)
                if dist <= tolerance:
                    phase_matches.append((pname, hkl, ref_pk, pinfo["color"], dist))
        phase_matches.sort(key=lambda x: x[4])
        if phase_matches:
            matches.append((pk, phase_matches))
    return matches

# ──────────────────────────────────────────────────────────
# Initialize session state for range presets
# ──────────────────────────────────────────────────────────
if "tth_min" not in st.session_state:
    st.session_state.tth_min = 30.0
if "tth_max" not in st.session_state:
    st.session_state.tth_max = 130.0

# ──────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")

    wl = st.number_input("Wavelength λ (Å)", value=1.54056, format="%.5f",
                         help="Cu Kα1 = 1.54056 Å")

    shape = st.selectbox("Peak profile", ["Pseudo-Voigt", "Gaussian", "Lorentzian"])

    if shape == "Pseudo-Voigt":
        eta = st.slider("η (Lorentzian fraction)", 0.0, 1.0, 0.5, 0.05)
    else:
        eta = 0.5

    st.subheader("Caglioti broadening")
    U = st.slider("U", 0.00, 0.20, 0.04, 0.005, format="%.3f")
    V = st.slider("V", -0.10, 0.00, -0.01, 0.002, format="%.3f")
    W = st.slider("W", 0.00, 0.10, 0.015, 0.002, format="%.3f")

    st.subheader("📐 2θ Range & Display")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        tth_min = st.slider("Min 2θ (°)", 30.0, 60.0, st.session_state.tth_min, 1.0, key="slider_min")
    with col_r2:
        tth_max = st.slider("Max 2θ (°)", 60.0, 130.0, st.session_state.tth_max, 1.0, key="slider_max")
    
    st.session_state.tth_min = tth_min
    st.session_state.tth_max = tth_max

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        if st.button("🔍 Full", help="30–130°"):
            st.session_state.tth_min, st.session_state.tth_max = 30.0, 130.0
            st.rerun()
    with col_p2:
        if st.button("🎯 Peaks", help="Zoom to main peaks 40–80°"):
            st.session_state.tth_min, st.session_state.tth_max = 40.0, 80.0
            st.rerun()
    with col_p3:
        if st.button("📊 High", help="High-angle region 80–130°"):
            st.session_state.tth_min, st.session_state.tth_max = 80.0, 130.0
            st.rerun()

    st.subheader("🔤 Font Settings")
    font_size = st.slider("Peak label font (pt)", 6, 16, 8, 1)
    label_fontsize = st.slider("Axis label font", 8, 18, 11, 1)
    tick_fontsize = st.slider("Tick font", 6, 14, 8, 1)
    legend_fontsize = st.slider("Legend font", 6, 14, 8, 1)

    st.subheader("🏷️ Peak Labels")
    show_all_phases = st.checkbox("Show ALL matching phases on peaks", value=True,
                                   help="Display every phase that matches within tolerance")
    show_phase_labels = st.checkbox("Show phase+hkl labels", value=True)
    show_2theta_labels = st.checkbox("Show 2θ values", value=False)
    peak_tolerance = st.slider("Peak match tolerance (°)", 0.1, 1.5, 0.5, 0.05)
    max_labels_per_peak = st.slider("Max phases to show per peak", 1, 5, 3, 1,
                                     help="Limit clutter when many phases overlap")

    st.subheader("Active phases (for refinement)")
    active = {}
    for ph in PHASES:
        active[ph] = st.checkbox(ph, value=(ph in ["γ-Co (FCC)", "ε-Co (HCP)"]))

    refine_btn = st.button("▶ Run Refinement", type="primary", use_container_width=True)

# ──────────────────────────────────────────────────────────
# File upload
# ──────────────────────────────────────────────────────────
st.subheader("📂 XRD Data")
col_up, col_info = st.columns([1, 1])

with col_up:
    uploaded = st.file_uploader(
        "Upload your .ASC / .xy / .txt / .csv file",
        type=["asc", "xy", "txt", "csv"],
        help="Two-column file: 2θ  Intensity"
    )

two_theta_raw, intensity_raw = None, None

if uploaded:
    try:
        content = uploaded.read().decode("utf-8", errors="replace")
        rows = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
        arr = np.array(rows)
        two_theta_raw, intensity_raw = arr[:, 0], arr[:, 1]
        with col_info:
            st.success(f"✅ Loaded {len(two_theta_raw)} points  |  "
                       f"2θ: {two_theta_raw.min():.2f}° – {two_theta_raw.max():.2f}°  |  "
                       f"I_max: {intensity_raw.max():.0f}")
    except Exception as e:
        st.error(f"Parse error: {e}")

# Use your CH0_1.ASC data as default if no file uploaded
if two_theta_raw is None:
    st.info("No file uploaded — showing example with synthetic pattern matching CH0_1.ASC statistics.")
    two_theta_raw = np.arange(30.013, 130.010, 0.026)
    intensity_raw = np.ones_like(two_theta_raw) * 320.0

    real_peaks = [
        (44.495, 130),  (48.161, 260),  (51.801, 335),
        (55.155, 2320), (60.173, 170),  (73.589, 525),
        (89.891, 175),  (101.331, 330), (112.459, 450), (115.345, 125),
    ]
    for c, amp in real_peaks:
        fwhm = caglioti_fwhm(c, 0.04, -0.01, 0.015)
        intensity_raw += pseudo_voigt(two_theta_raw, c, fwhm, amp)
    intensity_raw += np.random.normal(0, 8, len(two_theta_raw))
    intensity_raw = np.maximum(intensity_raw, 0)

# Apply 2θ range filter
mask = (two_theta_raw >= tth_min) & (two_theta_raw <= tth_max)
two_theta = two_theta_raw[mask]
intensity = intensity_raw[mask]

# ──────────────────────────────────────────────────────────
# Raw pattern + auto peak detection with ALL phase labeling
# ──────────────────────────────────────────────────────────
st.subheader("📈 Observed Pattern & Auto-detected Peaks")
smoothed = savgol_filter(intensity, window_length=min(21, len(intensity)//10*2+1), polyorder=3)
noise_level = np.std(intensity[:min(100, len(intensity)//5)])
baseline = np.percentile(intensity, 15)
auto_peaks, _ = find_peaks(smoothed, height=baseline + 3*noise_level, prominence=40, distance=12)

fig0, ax0 = plt.subplots(figsize=(14, 4))
ax0.plot(two_theta, intensity, color="#90A4AE", linewidth=0.7, label="Observed")
ax0.plot(two_theta, smoothed, color="#1565C0", linewidth=1.0, label="Smoothed")

# Match detected peaks to ALL phases within tolerance
all_matches = match_peak_to_all_phases(
    [two_theta[p] for p in auto_peaks], 
    PHASES, 
    tolerance=peak_tolerance
)

# Create lookup dict for quick access
match_lookup = {pk: matches for pk, matches in all_matches}

# Annotate peaks
for i, p in enumerate(auto_peaks):
    pk_val = two_theta[p]
    pk_int = intensity[p]
    
    if pk_val in match_lookup:
        phase_matches = match_lookup[pk_val]
        primary_color = phase_matches[0][3]
        ax0.axvline(pk_val, color=primary_color, linewidth=1.0, alpha=0.8, ls="-")
        
        if show_phase_labels and show_all_phases:
            display_matches = phase_matches[:max_labels_per_peak]
            
            for idx, (pname, hkl, ref_pk, pcolor, dist) in enumerate(display_matches):
                short_name = pname.split()[0]
                label = f"{hkl}\n{short_name}"
                y_offset = pk_int * (1.02 + idx * 0.03)
                ax0.text(pk_val, y_offset, label,
                         fontsize=max(6, font_size - idx),
                         ha="center", color=pcolor, 
                         rotation=90, va="bottom", 
                         bbox=dict(boxstyle="round,pad=0.15", 
                                  facecolor="white", edgecolor=pcolor, alpha=0.8),
                         zorder=10)
            
            if len(phase_matches) > max_labels_per_peak:
                extra = len(phase_matches) - max_labels_per_peak
                ax0.text(pk_val, pk_int * (1.02 + len(display_matches) * 0.03), 
                         f"+{extra}", fontsize=6, ha="center", color="gray",
                         rotation=90, va="bottom")
        elif show_phase_labels:
            pname, hkl, ref_pk, pcolor, dist = phase_matches[0]
            short_name = pname.split()[0]
            label = f"{hkl}\n{short_name}"
            ax0.text(pk_val, pk_int*1.02, label,
                     fontsize=font_size, ha="center", color=pcolor, 
                     rotation=90, va="bottom", 
                     bbox=dict(boxstyle="round,pad=0.2", 
                              facecolor="white", edgecolor=pcolor, alpha=0.7))
    else:
        ax0.axvline(pk_val, color="orange", linewidth=0.8, alpha=0.6, ls="--")
        if show_2theta_labels:
            ax0.text(pk_val, pk_int*1.01, f"{pk_val:.2f}°",
                     fontsize=font_size, ha="center", color="darkred", 
                     rotation=90, va="bottom")

# Add reference tick marks for ALL phases (subtle background guides)
for pname, pinfo in PHASES.items():
    c = pinfo["color"]
    for (pk, hkl) in pinfo["peaks"]:
        if tth_min <= pk <= tth_max:
            ylim = ax0.get_ylim()
            ax0.plot([pk, pk], [ylim[0], ylim[0] + (ylim[1]-ylim[0])*0.02], 
                    color=c, lw=0.5, alpha=0.3, ls=":")

ax0.set_xlabel("2θ (°)", fontsize=label_fontsize)
ax0.set_ylabel("Intensity (counts)", fontsize=label_fontsize)
ax0.tick_params(axis='both', labelsize=tick_fontsize)
ax0.set_xlim(tth_min, tth_max)
ax0.legend(fontsize=legend_fontsize, loc="upper right")
ax0.grid(True, alpha=0.25)
fig0.tight_layout()
st.pyplot(fig0)

# Summary of detected peaks with phase assignments
if len(auto_peaks) > 0:
    st.markdown("🔴 **Detected peaks with phase assignments:**")
    peak_info = []
    for p in auto_peaks:
        pk_val = two_theta[p]
        pk_int = intensity[p]
        if pk_val in match_lookup:
            phases_str = " | ".join([f"{m[1]}:{m[0].split()[0]}" for m in match_lookup[pk_val][:3]])
            if len(match_lookup[pk_val]) > 3:
                phases_str += f" (+{len(match_lookup[pk_val])-3})"
            peak_info.append(f"**{pk_val:.3f}°** ({pk_int:.0f} cts) → {phases_str}")
        else:
            peak_info.append(f"**{pk_val:.3f}°** ({pk_int:.0f} cts) → _unknown_")
    for item in peak_info[:10]:
        st.markdown(f"  • {item}")
    if len(peak_info) > 10:
        st.markdown(f"  _... and {len(peak_info)-10} more peaks_")

# ──────────────────────────────────────────────────────────
# Refinement
# ──────────────────────────────────────────────────────────
if refine_btn:
    sel_phases = {k: v for k, v in PHASES.items() if active.get(k)}
    if not sel_phases:
        st.warning("Select at least one phase.")
        st.stop()

    st.divider()
    st.subheader("🔁 Rietveld Profile Fit")

    with st.spinner("Optimising profile parameters…"):

        win = max(1, len(two_theta) // 30)
        bg_pts = [np.percentile(intensity[max(0,i-win):i+win], 10)
                  for i in range(0, len(two_theta), win)]
        bg_x   = two_theta[::win][:len(bg_pts)]
        bg_spl  = UnivariateSpline(bg_x, bg_pts, s=len(bg_pts)*500, k=3, ext=3)
        background = np.clip(bg_spl(two_theta), 0, intensity.min()*1.5)

        obs_net = np.maximum(intensity - background, 0)

        phase_profiles = {}
        peak_details   = []

        for pname, pinfo in sel_phases.items():
            prof = np.zeros_like(two_theta)
            for (pk, hkl) in pinfo["peaks"]:
                if not (tth_min <= pk <= tth_max):
                    continue
                fwhm = caglioti_fwhm(pk, U, V, W)
                idx  = np.argmin(np.abs(two_theta - pk))
                win2 = int(0.5 / 0.026)
                lo, hi = max(0, idx-win2), min(len(obs_net)-1, idx+win2)
                amp = max(obs_net[lo:hi].max(), 5.0)
                contrib = profile(two_theta, pk, fwhm, amp, shape, eta)
                prof += contrib
                peak_details.append({
                    "Phase": pname, "hkl": hkl,
                    "2θ_ref (°)": pk, "FWHM (°)": fwhm,
                    "Intensity": amp,
                    "Size (nm)": scherrer(fwhm, pk, wl),
                })
            phase_profiles[pname] = prof

        total_calc = background + sum(phase_profiles.values())

        diff   = intensity - total_calc
        w      = 1.0 / np.maximum(intensity, 1)
        Rwp    = np.sqrt(np.sum(w * diff**2) / np.sum(w * intensity**2)) * 100
        Rp     = np.sum(np.abs(diff)) / np.sum(intensity) * 100
        n_free = sum(len(v["peaks"]) for v in sel_phases.values()) * 2 + 3
        chi2   = np.sum(w * diff**2) / max(len(intensity) - n_free, 1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R_wp (%)", f"{Rwp:.2f}", help="< 10% is excellent")
    m2.metric("R_p (%)",  f"{Rp:.2f}")
    m3.metric("χ²",       f"{chi2:.4f}", help="Target ≈ 1")
    m4.metric("Peaks fitted", str(len(peak_details)))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                              gridspec_kw={"height_ratios": [4, 1]}, sharex=True)
    ax = axes[0]

    ax.plot(two_theta, intensity, "k-", lw=0.7, label="Observed (Yobs)", zorder=5)
    ax.plot(two_theta, total_calc, "r-", lw=1.3, label="Calculated (Ycalc)", zorder=6)
    ax.plot(two_theta, background, "--", color="#888", lw=0.9, label="Background", zorder=4)

    for pname, prof in phase_profiles.items():
        c = PHASES[pname]["color"]
        ax.fill_between(two_theta, background, prof + background,
                        color=c, alpha=0.15, zorder=2)
        ax.plot(two_theta, prof + background, "-", color=c, lw=0.9, label=pname, zorder=3)
    
    for pname, pinfo in PHASES.items():
        c = pinfo["color"]
        alpha = 0.7 if pname in sel_phases else 0.3
        ls = ":" if pname not in sel_phases else "-"
        for (pk, hkl) in pinfo["peaks"]:
            if tth_min <= pk <= tth_max:
                ax.axvline(pk, color=c, lw=0.6, ls=ls, alpha=alpha)
                if show_phase_labels:
                    short_name = pname.split()[0]
                    label = f"{hkl}\n{short_name}"
                    y_pos = ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0
                    if pname not in sel_phases:
                        y_pos -= (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                    ax.text(pk, y_pos, label, fontsize=max(6, font_size-1), 
                            color=c, ha="center", va="bottom" if pname in sel_phases else "top", 
                            rotation=90, alpha=alpha,
                            bbox=dict(boxstyle="round,pad=0.1", 
                                     facecolor="white", edgecolor=c, alpha=0.5))

    ax.set_ylabel("Intensity (counts)", fontsize=label_fontsize)
    ax.set_title("Rietveld Profile Fit — Co-Cr Dental Alloy (Cu Kα)", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(tth_min, tth_max)

    axes[1].plot(two_theta, diff, "-", color="#555", lw=0.6, label="Yobs − Ycalc")
    axes[1].axhline(0, color="red", lw=0.9)
    axes[1].fill_between(two_theta, diff, 0, where=diff > 0, color="#C62828", alpha=0.3)
    axes[1].fill_between(two_theta, diff, 0, where=diff < 0, color="#1565C0", alpha=0.3)
    axes[1].set_xlabel("2θ (°)", fontsize=label_fontsize)
    axes[1].set_ylabel("Difference", fontsize=label_fontsize)
    axes[1].tick_params(axis='both', labelsize=tick_fontsize)
    axes[1].grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("📊 Phase Fractions & Crystallite Sizes")
    total_area = sum(p.sum() for p in phase_profiles.values())
    frac_data = []
    for pname, prof in phase_profiles.items():
        frac = prof.sum() / total_area * 100 if total_area > 0 else 0
        ph_peaks = [d for d in peak_details if d["Phase"] == pname]
        avg_size = np.mean([d["Size (nm)"] for d in ph_peaks]) if ph_peaks else 0
        frac_data.append({
            "Phase": pname,
            "Space group": PHASES[pname]["space_group"],
            "a (Å)": PHASES[pname]["a_ang"],
            "Description": PHASES[pname]["description"],
            "Phase fraction (%)": f"{frac:.1f}",
            "Avg crystallite size (nm)": f"{avg_size:.1f}",
        })
    df_phases = pd.DataFrame(frac_data)
    st.dataframe(df_phases, use_container_width=True, hide_index=True)

    with st.expander("📋 Per-peak refinement details"):
        df_peaks = pd.DataFrame(peak_details)
        df_peaks["2θ_ref (°)"] = df_peaks["2θ_ref (°)"].map("{:.3f}".format)
        df_peaks["FWHM (°)"]   = df_peaks["FWHM (°)"].map("{:.4f}".format)
        df_peaks["Intensity"]  = df_peaks["Intensity"].map("{:.1f}".format)
        df_peaks["Size (nm)"]  = df_peaks["Size (nm)"].map("{:.2f}".format)
        st.dataframe(df_peaks, use_container_width=True, hide_index=True)

    st.subheader("🥧 Phase Distribution")
    fracs  = [float(d["Phase fraction (%)"])  for d in frac_data]
    labels = [d["Phase"] for d in frac_data]
    colors = [PHASES[d["Phase"]]["color"] for d in frac_data]
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax2.pie(
        fracs, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        textprops={"fontsize": tick_fontsize}
    )
    ax2.set_title("Estimated Phase Fractions", fontsize=label_fontsize)
    fig2.tight_layout()
    st.pyplot(fig2)

    st.subheader("⬇️ Export Results")
    dc1, dc2, dc3 = st.columns(3)

    csv1 = df_phases.to_csv(index=False)
    dc1.download_button("Phase table (CSV)", csv1, "phases.csv", "text/csv")

    csv2 = pd.DataFrame(peak_details).to_csv(index=False)
    dc2.download_button("Peak table (CSV)", csv2, "peaks.csv", "text/csv")

    buf = io.BytesIO()
    result = np.column_stack([two_theta, intensity, total_calc, background, diff])
    np.savetxt(buf, result,
               header="2theta  Observed  Calculated  Background  Difference",
               fmt="%.5f")
    dc3.download_button("Fit profile (.txt)", buf.getvalue(), "fit_profile.txt", "text/plain")

# ──────────────────────────────────────────────────────────
# Deployment guide
# ──────────────────────────────────────────────────────────
st.divider()
with st.expander("🚀 Deploy on GitHub + Streamlit Cloud (step-by-step)"):
    st.markdown("""
    **1. Create a GitHub repository**
    ```bash
    mkdir xrd-rietveld && cd xrd-rietveld
    git init
    ```

    **2. Add these two files to the folder:**
    - `app.py`  ← this file
    - `requirements.txt` ← see below

    **requirements.txt**
    ```
    streamlit>=1.32
    numpy
    scipy
    matplotlib
    pandas
    ```

    **3. Push to GitHub**
    ```bash
    git add .
    git commit -m "XRD Rietveld app for Co-Cr dental alloy"
    git push origin main
    ```

    **4. Deploy on [streamlit.io/cloud](https://streamlit.io/cloud)**
    - Click **New app** → select your repo
    - Main file: `app.py`
    - Click **Deploy** — free, no server needed ✅

    **For full structure-factor Rietveld** (CIF-based), add to requirements:
    ```
    pymatgen
    diffpy.srfit
    ```
    """)
