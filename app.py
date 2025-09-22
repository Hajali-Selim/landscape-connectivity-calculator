from functions import *

# Set page config for a wider layout
st.set_page_config(layout="wide")
if "landscape_initialised" not in st.session_state:
    st.session_state.landscape_initialised = False

if "landscape_matrix" not in st.session_state:
    st.session_state.landscape_matrix = None

# Create two columns: 1/4 (sidebar) and 3/4 (main area)
col1, col2 = st.columns([1, 3])
with col1:
    
    tab_selection, tab_generation = st.tabs(["Landscape Selection", "Landscape Generation"])
    # Two equal rows in left column
    with tab_selection:
        empirical_landscape = st.selectbox("Select a saved plot", ["Grassland", "Shrubland"])
        if empirical_landscape == "Grassland":
            landscape_vegetation = np.loadtxt('field-data/'+empirical_landscape+'/vegetation.asc', skiprows=6)[1:-1,1:-1]/100
            landscape_elevation = np.loadtxt('field-data/'+empirical_landscape+'/topography.asc', skiprows=6)[1:-1,1:-1]
            landscape_microtopography = landscape_elevation - landscape_elevation.mean(axis=1, keepdims=True)
            st.session_state.landscape_initialised = True
        st.markdown("Upload my own landscape")

    with tab_generation:
        generate_button = st.button("Generate New Landscape")
        vegetation_cover = st.number_input("Select vegetation cover between 0 and 1:", min_value=0.0, max_value=1.0, step=0.01, format="%.2f", value=0.3, key="cover")
        clustering_prob = st.number_input("Select clustering probability between 0 and 1:", min_value=0.0, max_value=1.0, step=0.01, format="%.2f", value=0.5, key="clust")
        vegetation_type = st.segmented_control("Select vegetation type:", ["Grassland", "Shrubland"], default="Grassland")
        if generate_button:
            landscape_vegetation = st.session_state.landscape_matrix = np.random.rand(60, 20)
            #landscape_vegetation = generate_vegetation_matrix(60, 20, vegetation_cover, clustering_prob, landscape_vegetation)
            landscape_elevation = st.session_state.landscape_matrix = np.random.rand(60, 20)
            landscape_microtopography = landscape_elevation - landscape_elevation.mean(axis=1, keepdims=True)
            st.session_state.landscape_initialised = True



with col2:
    selected_map = st.segmented_control(
        "Show map relative to:", ["Vegetation", "Microtopography", "Structural Connectivity"], default="Vegetation")
    #st.subheader(f"Map: {selected_map}")
    cmap = "Greens" if selected_map == "Vegetation" else ("rainbow" if selected_map == "Microtopography" else "terrain_r")
    if st.session_state.landscape_matrix is not None:
        figsize = get_figsize(st.session_state.landscape_matrix.shape)
        st.markdown(figsize)
        if st.session_state.landscape_initialised:
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(st.session_state.landscape_matrix, cmap=cmap, aspect='equal')
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close()
        else:
            st.write("Click on 'Generate Landscape' to display a map")

