from functions import *

# Set page config for a wider layout
st.set_page_config(layout="wide")
ss = st.session_state
ss.setdefault("mode", "Selection")             # "Selection" | "Generation"
ss.setdefault("current_source", None)          # "empirical" | "generated"
ss.setdefault("empirical", {})                 # dict with 'veg', 'micro'
ss.setdefault("generated", {})                 # dict with 'veg', 'micro'
ss.setdefault("landscape_ready", False)

interpolation_data = {
    'grassland': {'alpha': np.float64(0.001318), 'alpha_std': np.float64(0.032865), 'beta': np.float64(-0.003057)},
    'shrubland': {'alpha': np.float64(0.00842), 'alpha_std': np.float64(0.011797), 'beta': np.float64(-0.009644)}
}

# Create two columns: 1/4 (sidebar) and 3/4 (main area)
col1, col2 = st.columns([1, 3])

with col1:
    ss.mode = st.segmented_control("Select a mode", ["Selection", "Generation"], default=ss.mode)
    if ss.mode == "Selection":
        with st.form("select_form"):
            kind = st.selectbox("Select a real landscape plot", ["shrubland", "grassland"])
            submit = st.form_submit_button("Load landscape")
        if submit:
            veg, plane, micro = load_empirical(kind)
            d4_direction = d4_steepest_descent(plane+micro)
            sc = compute_SC(d4_direction)
            ss.empirical = {"veg": veg, "micro": micro, "sc": sc, "kind": kind}
            ss.current_source = "empirical"
            ss.landscape_ready = True
    else:
        with st.form("gen_form"):
            col_w, col_h = st.columns(2)
            width = col_w.number_input("Width (cells)", min_value=10, max_value=200, value=20, step=1)
            height = col_h.number_input("Height (cells)", min_value=10, max_value=200, value=60, step=1)
            vegetation_cover = st.number_input("Vegetation cover", 0.0, 1.0, 0.3, 0.01, format="%.2f")
            clustering_prob = st.number_input("Clustering probability", 0.0, 1.0, 0.6, 0.1, format="%.1f")
            kind = st.segmented_control("Vegetation type", ["shrubland", "grassland"], default="shrubland")
            submit = st.form_submit_button("Generate landscape")
        if submit:
            if not kind:
                st.error("Please select a vegetation type to generate a landscape.")
            else:
                ss["gen_width"] = int(width)
                ss["gen_height"] = int(height)
                v_src, plane, _ = load_empirical(kind)
                v = generate_vegetation_matrix(height, width, vegetation_cover, clustering_prob, v_src)
                micro = generate_microtopography(v, interpolation_data[kind])
                plane_adapted = adapt_plane(plane, height, width)
                d4_direction = d4_steepest_descent(plane_adapted+micro)
                sc = compute_SC(d4_direction)
                ss.generated = {"veg": v, "micro": micro, "sc": sc, "kind": kind}
                ss.current_source = "generated"
                ss.landscape_ready = True
with col2:
    map_kind = st.segmented_control("Show map", ["Vegetation", "Microtopography", "Structural Connectivity"], default="Vegetation")
    src = ss.current_source
    if ss.landscape_ready and src in ("empirical", "generated"):
        data = ss[src]["veg"] if map_kind == "Vegetation" else (ss[src]["micro"] if map_kind == "Microtopography" else ss[src]["sc"])
        # TODO: add structural connectivity if available; fallback to vegetation
        cmap = "Greens" if map_kind == "Vegetation" else ("rainbow" if map_kind == "Microtopography" else "terrain_r")
        fig, ax = plt.subplots(figsize=get_figsize(data.shape))
        ax.imshow(data, cmap=cmap, aspect="equal")
        ax.axis("off")
        st.pyplot(fig, use_container_width=False)
    else:
        st.write("Choose a mode and submit to display a map.")

