"""
Coverage Analysis Page

This page provides comprehensive radar coverage analysis:
1. Loads terrain data from NPZ files
2. Visualizes terrain in 2D and 3D
3. Computes coverage on Flight Levels (FL)
4. Outputs coverage maps with various backgrounds
5. Exports results to KMZ format
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import shared utilities from main app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from radar_coverage_app import (
    # Theme and styling
    apply_theme,
    BG_APP, BG_PANEL, BG_CARD, BORDER_COLOR,
    TEXT_PRIMARY, TEXT_SECONDARY,
    ACCENT_PRIMARY, ACCENT_HOVER,
    COVERAGE_VISIBLE, BLOCKED_AREA,
    METRIC_VALUE,
    # Constants
    DEFAULT_REF_LAT, DEFAULT_REF_LON, EARTH_RADIUS_M,
    AIRPORT_LAT, AIRPORT_LON, AIRPORT_NAME,
    FLIGHT_LEVELS,
    # Terrain functions
    load_terrain_npz, load_terrain_with_curvature,
    convert_to_enu, normalize_xy_grid, normalize_all,
    # Coverage computation
    fl_to_m, latlon_to_xy_m,
    compute_fl_coverage_curvature, compute_all_fl_curvature,
    # Visualization
    plot_terrain_2d, plot_terrain_3d,
    plot_coverage, plot_coverage_large, plot_all_coverage_grid,
    colored_relief, hillshade,
    # Export
    create_png_zip, create_coverage_csv, export_kmz,
    # Module availability flags
    HAS_NUMBA, HAS_CONTEXTILY, HAS_CURVATURE_MODULES,
)


def render_page_title():
    """Render centered page title with modern styling."""
    st.markdown(f"""
        <div style="margin-bottom: 2.5rem;">
            <div style="text-align: center; font-size: 3rem; font-weight: 800; color: {TEXT_PRIMARY}; margin-bottom: 0.1rem; letter-spacing: -1px;">
                Coverage Analysis
            </div>
            <div style="text-align: center; font-size: 1rem; color: {TEXT_SECONDARY}; margin-bottom: 2rem; font-weight: 500; letter-spacing: 2px; text-transform: uppercase; opacity: 0.8;">
                Terrain Modeling & Line-of-Sight Computing
            </div>
        </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Coverage Analysis - Radar Coverage",
        page_icon="📡",
        layout="wide"
    )
    apply_theme()
    render_page_title()
    
    if 'coverage_computed' not in st.session_state:
        st.session_state.coverage_computed = False
    if 'expanded_fl' not in st.session_state:
        st.session_state.expanded_fl = None
    
    # Sidebar - Configuration Inputs Only
    with st.sidebar:
        st.header("Configuration")
        
        # Terrain Data Expander
        with st.expander("Terrain Data", expanded=True):
            uploaded = st.file_uploader("Upload NPZ terrain file", type=['npz'])
            use_sample = st.checkbox("Use sample terrain", value=not uploaded)
        
        terrain_data = None
        if uploaded:
            try:
                # Load raw terrain for visualization
                lats, lons, Z = load_terrain_npz(uploaded)
                # For uploaded files, apply curvature correction manually
                # (since load_terrain_with_curvature expects a file path)
                ref_lat_corr = 43.6584  # Nice Airport (same as geo_utils_earth_curvature)
                ref_lon_corr = 7.2159
                lat_ref_rad = np.radians(ref_lat_corr)
                meters_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
                meters_per_deg_lon = (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(lat_ref_rad)
                Y_m_corr = (lats - ref_lat_corr) * meters_per_deg_lat
                X_m_corr = (lons - ref_lon_corr) * meters_per_deg_lon
                X_grid, Y_grid = np.meshgrid(X_m_corr, Y_m_corr)
                dist_sq = X_grid**2 + Y_grid**2
                curvature_drop = dist_sq / (2.0 * EARTH_RADIUS_M)
                Z_corrected = Z - curvature_drop
                X_m_corr, Y_m_corr, Z_corrected, lats_norm, lons_norm = normalize_all(
                    X_m_corr, Y_m_corr, Z_corrected, lats, lons
                )
                terrain_data = {
                    'lats': lats, 'lons': lons, 'Z': Z,
                    'X_m': X_m_corr, 'Y_m': Y_m_corr, 'Z_corrected': Z_corrected,
                    'lats_norm': lats_norm, 'lons_norm': lons_norm
                }
            except Exception as e:
                st.error(f"Error loading terrain: {e}")
        elif use_sample:
            sample_path = Path(__file__).parent.parent / "terrain_mat.npz"
            if sample_path.exists():
                # Load with curvature correction (same as main_coverage.py)
                X_m_corr, Y_m_corr, Z_corrected, lats, lons = load_terrain_with_curvature(str(sample_path))
                X_m_corr, Y_m_corr, Z_corrected, lats_norm, lons_norm = normalize_all(
                    X_m_corr, Y_m_corr, Z_corrected, lats, lons
                )
                # Also load raw Z for visualization
                lats_raw, lons_raw, Z_raw = load_terrain_npz(str(sample_path))
                terrain_data = {
                    'lats': lats_raw, 'lons': lons_raw, 'Z': Z_raw,
                    'X_m': X_m_corr, 'Y_m': Y_m_corr, 'Z_corrected': Z_corrected,
                    'lats_norm': lats_norm, 'lons_norm': lons_norm
                }
        
        if terrain_data:
            lats, lons, Z = terrain_data['lats'], terrain_data['lons'], terrain_data['Z']
            
            # Radar Position Expander
            with st.expander("Radar Position", expanded=True):
                radar_lat = st.number_input("Latitude", value=DEFAULT_REF_LAT, format="%.6f")
                radar_lon = st.number_input("Longitude", value=DEFAULT_REF_LON, format="%.6f")
                radar_h = st.number_input("Height (m AGL)", value=20.0, min_value=0.0)
            
            # Flight Levels Expander
            with st.expander("Flight Levels", expanded=True):
                # Initialize session state for FL selection if not present
                if 'selected_fls' not in st.session_state:
                    st.session_state.selected_fls = [10, 50, 100, 200]
                
                # FL multiselect (simplified, no preset buttons)
                selected_fls = st.multiselect(
                    "Select FLs", 
                    FLIGHT_LEVELS, 
                    default=st.session_state.selected_fls,
                    key="fl_multiselect"
                )
                # Update session state when user manually changes selection
                st.session_state.selected_fls = selected_fls
            
            # Advanced Settings Expander
            with st.expander("Advanced", expanded=False):
                n_samples = st.number_input("LOS Samples", value=400, min_value=50)
                margin = st.number_input("Margin (m)", value=0.0)
            
            # Compute button at the bottom
            sidebar_compute = st.button("Compute Coverage", type="primary", use_container_width=True)
        else:
            # No terrain data loaded - set defaults for sidebar_compute check
            sidebar_compute = False
            selected_fls = []
            n_samples = 400
            margin = 0.0
    
    # =========================================================================
    # MAIN CONTENT (outside sidebar) - LINEAR SECTION LAYOUT
    # =========================================================================
    
    if terrain_data:
        # Raw terrain for visualization
        lats, lons, Z = terrain_data['lats'], terrain_data['lons'], terrain_data['Z']
        
        # Curvature-corrected terrain for coverage computation (same as main_coverage.py)
        X_m = terrain_data['X_m']
        Y_m = terrain_data['Y_m']
        Z_corrected = terrain_data['Z_corrected']
        lats_norm = terrain_data['lats_norm']
        lons_norm = terrain_data['lons_norm']
        
        # Initialize selected_overview_fl if not set
        if 'selected_overview_fl' not in st.session_state:
            st.session_state.selected_overview_fl = None
        
        # =====================================================================
        # SECTION 1: TERRAIN VISUALIZATION (Always visible when terrain loaded)
        # =====================================================================
        st.header("Terrain Visualization")
        
        # Terrain metrics container
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min Elevation", f"{Z.min():.1f} m")
            with col2:
                st.metric("Max Elevation", f"{Z.max():.1f} m")
            with col3:
                st.metric("Mean Elevation", f"{Z.mean():.1f} m")
            with col4:
                st.metric("Grid Points", f"{Z.size:,}")
        
        st.markdown("<br>", unsafe_allow_html=True)

        # 2D and 3D terrain visualization
        col_2d, col_3d = st.columns(2)
        
        with col_2d:
            st.subheader("2D Elevation Map")
            fig_2d = plot_terrain_2d(lats, lons, Z, radar_lat, radar_lon, figsize=(7, 6))
            st.pyplot(fig_2d, use_container_width=True)
            plt.close(fig_2d)
        
        with col_3d:
            st.subheader("3D Terrain Surface")
            # Initialize angle values in session state if not set
            if 'terrain_elev_val' not in st.session_state:
                st.session_state.terrain_elev_val = 30
            if 'terrain_azim_val' not in st.session_state:
                st.session_state.terrain_azim_val = -60
            fig_3d = plot_terrain_3d(lats, lons, Z, radar_lat, radar_lon, 
                                     st.session_state.terrain_elev_val, 
                                     st.session_state.terrain_azim_val, figsize=(7, 6))
            st.pyplot(fig_3d, use_container_width=True)
            plt.close(fig_3d)
            # Angle controls below the 3D surface
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.terrain_elev_val = st.slider("Elevation", 0, 90, st.session_state.terrain_elev_val, key="terrain_elev")
            with c2:
                st.session_state.terrain_azim_val = st.slider("Azimuth", -180, 180, st.session_state.terrain_azim_val, key="terrain_azim")
        
        # Compute Coverage button at end of terrain section
        st.markdown("<br>", unsafe_allow_html=True)
        compute_col1, compute_col2, compute_col3 = st.columns([1, 2, 1])
        with compute_col2:
            main_compute = st.button(
                "COMPUTE ALL COVERAGE", 
                type="primary", 
                use_container_width=True,
                key="main_compute_btn"
            )
        
        # Handle coverage computation (triggered from main button or sidebar)
        # Uses Earth curvature-corrected algorithms (same as main_coverage.py)
        if (main_compute or sidebar_compute) and selected_fls:
            prog = st.progress(0)
            stat = st.empty()
            def upd(p):
                prog.progress(p)
                stat.text(f"Computing with Earth curvature correction... {p*100:.0f}%")
            # Use curvature-corrected terrain (Z_corrected) for computation
            maps = compute_all_fl_curvature(
                radar_lat, radar_lon, radar_h, 
                sorted(selected_fls), 
                X_m, Y_m, Z_corrected,  # Using curvature-corrected terrain
                n_samples, margin, upd
            )
            st.session_state.coverage_maps = maps
            st.session_state.coverage_computed = True
            # Store normalized lats/lons that match Z_corrected orientation
            st.session_state.coverage_lats = lats_norm
            st.session_state.coverage_lons = lons_norm
            st.session_state.coverage_terrain = Z_corrected  # Store curvature-corrected terrain
            st.session_state.radar_lat = radar_lat
            st.session_state.radar_lon = radar_lon
            st.session_state.radar_h = radar_h
            # Store grid data for computing additional FLs in export
            st.session_state.X_m = X_m
            st.session_state.Y_m = Y_m
            st.session_state.n_samples = n_samples
            st.session_state.margin = margin
            st.session_state.expanded_fl = None
            prog.progress(1.0)
            stat.text("Done!")
        
        # =====================================================================
        # SECTION 2: COVERAGE ANALYSIS (After computation)
        # =====================================================================
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.header("Coverage Analysis")

        if not st.session_state.coverage_computed or 'coverage_maps' not in st.session_state:
            st.info("Configure settings in the sidebar and click **Compute Coverage** to see results.")
        else:
            maps = st.session_state.coverage_maps
            sorted_fls = sorted(maps.keys())
            
            # Coverage maps display options - shown FIRST
            st.subheader("Coverage Maps")
            map_opt_col1, map_opt_col2, map_opt_col3, map_opt_col4 = st.columns(4)
            with map_opt_col1:
                map_style = st.selectbox(
                    "Background",
                    ["Elevation", "Satellite (Esri)", "Street (Carto)"],
                    index=0,  # default to elevation
                    key="coverage_map_style"
                )
            with map_opt_col2:
                map_show_blocked = st.toggle("Show blocked areas", value=False, key="coverage_map_blocked")
            with map_opt_col3:
                map_green_alpha = st.slider("Visible opacity", 0.1, 1.0, 0.6, 0.05, key="coverage_green_alpha")
            with map_opt_col4:
                map_red_alpha = st.slider("Blocked opacity", 0.1, 1.0, 0.5, 0.05, key="coverage_red_alpha")

            # Map UI style -> plotting params
            if map_style == "Elevation":
                plot_bg = "terrain"
                basemap_provider = None
                basemap_labels = False
            elif map_style == "Satellite (Esri)":
                plot_bg = "basemap"
                basemap_provider = "esri"
                basemap_labels = False
            else:  # "Street (Carto)"
                plot_bg = "basemap"
                basemap_provider = "carto"
                basemap_labels = False
            
            # All coverage maps in a grid (max 2 per row)
            n_fls = len(sorted_fls)
            n_cols = min(2, n_fls)  # Max 2 columns
            rows = (n_fls + n_cols - 1) // n_cols
            for r in range(rows):
                cols = st.columns(n_cols)
                for c in range(n_cols):
                    idx = r * n_cols + c
                    if idx < n_fls:
                        fl = sorted_fls[idx]
                        with cols[c]:
                            fig = plot_coverage(
                                cov=maps[fl],
                                lats=st.session_state.coverage_lats,
                                lons=st.session_state.coverage_lons,
                                fl=fl,
                                terrain=st.session_state.coverage_terrain,
                                radar_lat=st.session_state.radar_lat,
                                radar_lon=st.session_state.radar_lon,
                                bg=plot_bg,
                                basemap_provider=basemap_provider,
                                basemap_labels=basemap_labels,
                                show_blocked=map_show_blocked,
                                green_alpha=map_green_alpha,
                                red_alpha=map_red_alpha,
                                figsize=(5, 4)
                            )
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
            
            st.markdown("---")
            
            # Coverage statistics table (all FLs) - shown AFTER maps
            st.subheader("Coverage Statistics")
            data = []
            for fl in sorted(maps.keys()):
                cov_fl = maps[fl]
                vis = np.sum(cov_fl)
                tot = cov_fl.size
                data.append({
                    'Flight Level': f'FL{int(fl)}',
                    'Altitude (m)': f'{fl_to_m(fl):.0f}',
                    'Altitude (ft)': f'{int(fl)*100}',
                    'Visible Points': f'{vis:,}',
                    'Total Points': f'{tot:,}',
                    'Coverage (%)': f'{100*vis/tot:.2f}%'
                })
            st.dataframe(data, use_container_width=True, hide_index=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # =====================================================================
        # SECTION 3: EXPORT (After computation)
        # =====================================================================
        st.header("Export Results")
        
        if not st.session_state.coverage_computed or 'coverage_maps' not in st.session_state:
            st.info("Compute coverage first to enable exports.")
        else:
            maps = st.session_state.coverage_maps
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Three large, clear export cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("KMZ Export")
                st.caption("For Google Earth / GIS software")
                st.markdown("""
                - Coverage polygons by FL
                - Radar & airport markers
                - Color-coded visibility
                """)
                kmz = export_kmz(
                    maps, 
                    st.session_state.coverage_lats, 
                    st.session_state.coverage_lons,
                    st.session_state.radar_lat, 
                    st.session_state.radar_lon, 
                    show_blocked=True
                )
                st.download_button(
                    "Download KMZ",
                    kmz,
                    "radar_coverage.kmz",
                    "application/vnd.google-earth.kmz",
                    use_container_width=True,
                    key="kmz_download"
                )
            
            with col2:
                st.subheader("PNG Images")
                st.caption("High-resolution coverage maps")
                st.markdown("""
                - One image per FL
                - Choose background style
                - Optional blocked layer
                """)

                png_style = st.selectbox(
                    "PNG background",
                    ["Elevation", "Satellite (Esri)", "Street (Carto)"],
                    index=0,
                    key="png_bg_style",
                )
                png_show_blocked = st.toggle("Include blocked areas", value=False, key="png_blocked")
                
                # Expanders for advanced PNG options to keep the card clean
                with st.expander("Advanced PNG Settings"):
                    png_green_alpha = st.slider("Visible opacity", 0.1, 1.0, 0.6, 0.05, key="png_green_alpha")
                    png_red_alpha = st.slider("Blocked opacity", 0.1, 1.0, 0.5, 0.05, key="png_red_alpha")
                    png_dpi = st.selectbox("PNG DPI", [100, 150, 200, 300], index=1, key="png_dpi")
                    png_fls = st.multiselect(
                        "Flight levels to export",
                        sorted(maps.keys()),
                        default=sorted(maps.keys()),
                        format_func=lambda x: f"FL{int(x)}",
                        key="png_fls",
                    )

                if png_style == "Elevation":
                    png_plot_bg = "terrain"
                    png_basemap_provider = None
                    png_basemap_labels = False
                elif png_style == "Satellite (Esri)":
                    png_plot_bg = "basemap"
                    png_basemap_provider = "esri"
                    png_basemap_labels = False
                else:
                    png_plot_bg = "basemap"
                    png_basemap_provider = "carto"
                    png_basemap_labels = False

                png_zip = create_png_zip(
                    maps, 
                    st.session_state.coverage_lats, 
                    st.session_state.coverage_lons,
                    st.session_state.coverage_terrain, 
                    st.session_state.radar_lat,
                    st.session_state.radar_lon,
                    st.session_state.radar_h,
                    bg=png_plot_bg,
                    show_blocked=png_show_blocked,
                    green_alpha=png_green_alpha,
                    red_alpha=png_red_alpha,
                    basemap_provider=png_basemap_provider,
                    basemap_labels=png_basemap_labels,
                    fls_to_export=png_fls,
                    dpi=png_dpi,
                    X_m=st.session_state.X_m,
                    Y_m=st.session_state.Y_m,
                    n_samples=st.session_state.n_samples,
                    margin=st.session_state.margin,
                )
                st.download_button(
                    "Download PNG (ZIP)",
                    png_zip,
                    "coverage_maps.zip",
                    "application/zip",
                    use_container_width=True,
                    key="png_download"
                )
            
            with col3:
                st.subheader("CSV Data")
                st.caption("Coverage statistics spreadsheet")
                st.markdown("""
                - Statistics per FL
                - Altitude conversions
                - Import to Excel/analysis
                """)
                csv_content = create_coverage_csv(
                    maps, 
                    st.session_state.radar_lat,
                    st.session_state.radar_lon, 
                    st.session_state.radar_h
                )
                st.download_button(
                    "Download CSV",
                    csv_content,
                    "coverage_statistics.csv",
                    "text/csv",
                    use_container_width=True,
                    key="csv_download"
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Export summary in a centered box
            st.markdown(f"""
                <div style='background-color: {BG_CARD}; padding: 20px; border-radius: 12px; border: 1px solid {BORDER_COLOR}; text-align: center;'>
                    <h4 style='margin-top: 0;'>Export Summary</h4>
                    <p style='color: {TEXT_SECONDARY}; margin-bottom: 5px;'>Coverage computed for {len(maps)} flight levels: {', '.join([f'FL{int(fl)}' for fl in sorted(maps.keys())])}</p>
                    <p style='color: {TEXT_SECONDARY}; margin-bottom: 0;'>Radar position: ({st.session_state.radar_lat:.6f}°N, {st.session_state.radar_lon:.6f}°E)</p>
                </div>
            """, unsafe_allow_html=True)
    
    else:
        st.info("Upload terrain data or enable sample terrain in the sidebar to begin.")


if __name__ == "__main__":
    main()
