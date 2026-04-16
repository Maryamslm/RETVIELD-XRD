# Add this to your existing tab_fit code:
with st.expander("🎨 Publication-Quality Plot", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        enable_smooth = st.checkbox("Apply Savitzky-Golay smoothing", value=True)
        smooth_win = st.slider("Smoothing window", 3, 21, 7, 2) if enable_smooth else 5
    with col2:
        fig_width = st.slider("Figure width (inches)", 6, 12, 8)
        fig_height = st.slider("Figure height (inches)", 4, 10, 6)
    with col3:
        export_format = st.selectbox("Export format", ["PNG", "PDF", "SVG"])
    
    if st.button("🖨️ Generate Publication Plot"):
        from rietveld_plot import streamlit_rietveld_plot
        
        phase_data_for_plot = {}
        for i, ph_obj in enumerate(refiner.phases):
            phase_data_for_plot[f"phase_{i}"] = {
                'positions': [p["tt"] + z_shift for p in generate_reflections(
                    _make_refined_phase(ph_obj, pp_vec[i][1], pp_vec[i][2]), 
                    wl=wavelength, tt_min=float(tt.min()), tt_max=float(tt.max()))],
                'label': ph_obj.name,
                'color': ph_obj.color
            }
        
        refinement_info = {
            'Rp': rp * 100,
            'Rwp': rwp * 100, 
            'chi2': chi2,
            'GOF': gof
        }
        
        streamlit_rietveld_plot(
            tt, Iobs, I_calc=r["Icalc"], I_bg=r["Ibg"], I_diff=r["diff"],
            phase_data=phase_data_for_plot,
            refinement_results=refinement_info,
            sample_name=st.session_state.selected_sample or "Co-Cr Alloy"
        )
