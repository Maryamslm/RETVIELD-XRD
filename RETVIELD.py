#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Co-Cr XRD Analysis - Minimal Working Version"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Try to import powerxrd
try:
    import powerxrd as xrd
    from powerxrd.model import CubicModel
    import powerxrd.refine as rr
    POWERXRD_OK = True
except ImportError:
    POWERXRD_OK = False

st.set_page_config(page_title="Co-Cr XRD Analysis", layout="wide")
st.title("🧪 Co-Cr Dental Alloy XRD Analysis")

# Show dependency status
if not POWERXRD_OK:
    st.error("❌ `powerxrd` not installed. Add to requirements.txt and redeploy.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload .ASC file", type=["asc", "txt", "csv"])

if uploaded_file:
    try:
        # Load data
        df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, 
                        names=['two_theta', 'intensity'], engine='python')
        df = df.dropna().reset_index(drop=True)
        
        x = df['two_theta'].values
        y = df['intensity'].values
        
        # Plot raw data
        st.subheader("Raw Pattern")
        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))
        fig.update_layout(xaxis_title='2θ (°)', yaxis_title='Intensity')
        st.plotly_chart(fig, use_container_width=True)
        
        # Process & refine if powerxrd available
        if POWERXRD_OK and st.button("Run Refinement"):
            chart = xrd.Chart(x, y)
            chart.backsub()
            chart.mav(window=7)
            
            model = CubicModel()
            model.params = {"a": 3.58, "scale": 1000, "U": 0.01, "W": 0.01,
                           "bkg_intercept": 0, "bkg_slope": 0}
            
            rr.refine(model, chart.x, chart.y, ["scale"])
            rr.refine(model, chart.x, chart.y, ["bkg_intercept", "bkg_slope"])
            rr.refine(model, chart.x, chart.y, ["a"])
            
            y_calc = model.pattern(chart.x)
            
            # Plot fit
            fig_fit = go.Figure()
            fig_fit.add_trace(go.Scatter(x=chart.x, y=chart.y, name='Observed'))
            fig_fit.add_trace(go.Scatter(x=chart.x, y=y_calc, name='Calculated'))
            fig_fit.update_layout(xaxis_title='2θ (°)', yaxis_title='Intensity')
            st.plotly_chart(fig_fit, use_container_width=True)
            
            st.success(f"✅ Refined a = {model.params['a']:.4f} Å")
            
    except Exception as e:
        st.error(f"Error: {e}")
        st.code("Check file format: two space-separated columns (2θ Intensity)")
else:
    st.info("👆 Upload a file to begin")
