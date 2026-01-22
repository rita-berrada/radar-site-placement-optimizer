"""
Site Selection Page

This page provides optimal radar site selection analysis using geographic constraints:
1. Predefined constraints (land, 50km radius, French territory, coastline buffer, slope)
2. User-defined constraints via GeoJSON uploads (roads, buildings, residential, protected areas)
3. Interactive constraint visualization
4. Candidate site ranking and export
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from io import BytesIO
import zipfile
import json
import tempfile
import os

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
    # Terrain functions
    load_terrain_npz,
    # Contextily availability
    HAS_CONTEXTILY,
)

# Import mask modules
try:
    from site_location_masks import (
        mask_land, mask_50km, mask_french_territory, 
        mask_coastline_buffer, combine_masks
    )
    from mask_slope import mask_slope
    from roads_masks import mask_roads_from_geojson, load_roads_and_convert_to_enu, mask_roads_proximity_fast
    from buildings_masks import mask_buildings_from_geojson, load_buildings_and_convert_to_enu, mask_buildings_exclusion_fast
    from mask_residential import mask_residential_from_geojson
    from protected_areas_mask import mask_protected_areas_from_geojson
    from electrical_stations_masks import load_stations_and_convert_to_enu, mask_electrical_proximity_fast
    from geo_utils_earth_curvature import load_and_convert_to_enu, REF_LAT, REF_LON, EARTH_RADIUS_M as GEO_EARTH_RADIUS_M
    HAS_MASK_MODULES = True
except ImportError as e:
    HAS_MASK_MODULES = False
    MASK_IMPORT_ERROR = str(e)

# Import export module
try:
    from export_site_location_masks_kml import export_masks_to_kmz
    HAS_KMZ_EXPORT = True
except ImportError:
    HAS_KMZ_EXPORT = False

# Import contextily for basemaps
if HAS_CONTEXTILY:
    import contextily as cx


# ============================================================================
# STYLING CONSTANTS
# ============================================================================
ADMISSIBLE_COLOR = "#4CAF50"  # Green for valid areas
EXCLUDED_COLOR = "#9E9E9E"    # Grey for excluded areas


# ============================================================================
# PAGE RENDERING
# ============================================================================

def render_page_title():
    """Render centered page title with modern styling."""
    st.markdown(f"""
        <div style="margin-bottom: 2.5rem;">
            <div style="text-align: center; font-size: 3rem; font-weight: 800; color: {TEXT_PRIMARY}; margin-bottom: 0.1rem; letter-spacing: -1px;">
                Site Selection
            </div>
            <div style="text-align: center; font-size: 1rem; color: {TEXT_SECONDARY}; margin-bottom: 2rem; font-weight: 500; letter-spacing: 2px; text-transform: uppercase; opacity: 0.8;">
                Constraint-Based Radar Site Selection
            </div>
        </div>
    """, unsafe_allow_html=True)


# ============================================================================
# TERRAIN LOADING & COORDINATE CONVERSION
# ============================================================================

def load_terrain_data(terrain_source, uploaded_file=None):
    """
    Load terrain data and convert to ENU metric coordinates.
    
    Returns:
        dict with keys: X_m, Y_m, X_grid, Y_grid, Z_raw, Z_corrected, lats, lons
    """
    if terrain_source == "uploaded" and uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            # Load with curvature correction
            X_m, Y_m, Z_corrected, lats, lons = load_and_convert_to_enu(tmp_path)
            # Load raw Z for land/sea detection
            raw_data = np.load(tmp_path)
            Z_raw = raw_data['ter']
        finally:
            os.unlink(tmp_path)
    else:
        # Use sample terrain
        sample_path = Path(__file__).parent.parent / "terrain_mat.npz"
        if not sample_path.exists():
            sample_path = Path(__file__).parent.parent / "terrain_req01_50km.npz"
        
        if not sample_path.exists():
            return None
        
        X_m, Y_m, Z_corrected, lats, lons = load_and_convert_to_enu(str(sample_path))
        raw_data = np.load(str(sample_path))
        Z_raw = raw_data['ter']
    
    # Create 2D meshgrids for mask computations
    X_grid, Y_grid = np.meshgrid(X_m, Y_m)
    
    return {
        'X_m': X_m,
        'Y_m': Y_m,
        'X_grid': X_grid,
        'Y_grid': Y_grid,
        'Z_raw': Z_raw,
        'Z_corrected': Z_corrected,
        'lats': lats,
        'lons': lons
    }


# ============================================================================
# MASK COMPUTATION FUNCTIONS
# ============================================================================

def compute_predefined_masks(terrain_data, config):
    """
    Compute all enabled predefined masks.
    
    Args:
        terrain_data: dict from load_terrain_data()
        config: dict with constraint configuration
    
    Returns:
        dict mapping constraint names to boolean mask arrays
    """
    masks = {}
    X_grid = terrain_data['X_grid']
    Y_grid = terrain_data['Y_grid']
    Z_raw = terrain_data['Z_raw']
    Z_corrected = terrain_data['Z_corrected']
    X_m = terrain_data['X_m']
    Y_m = terrain_data['Y_m']
    
    # Land mask (always computed as base)
    if config.get('land_enabled', True):
        masks['Land Only'] = mask_land(Z_raw)
    
    # 50km radius
    if config.get('radius_enabled', True):
        radius_km = config.get('radius_km', 50.0)
        masks[f'{radius_km:.0f}km Radius'] = mask_50km(X_grid, Y_grid, radius_km=radius_km)
    
    # French territory
    if config.get('french_enabled', False):
        masks['French Territory'] = mask_french_territory(X_grid, Y_grid)
    
    # Coastline buffer
    if config.get('coastline_enabled', False):
        buffer_m = config.get('coastline_buffer_m', 100.0)
        masks[f'Coastline Buffer ({buffer_m:.0f}m)'] = mask_coastline_buffer(X_grid, Y_grid, Z_raw, buffer_m=buffer_m)
    
    # Slope constraint
    if config.get('slope_enabled', False):
        max_slope = config.get('max_slope_percent', 15.0)
        masks[f'Slope ≤{max_slope:.0f}%'] = mask_slope(X_m, Y_m, Z_corrected, max_slope_percent=max_slope)
    
    return masks


def compute_user_masks(terrain_data, config):
    """
    Compute user-defined masks from GeoJSON uploads.
    
    Args:
        terrain_data: dict from load_terrain_data()
        config: dict with user constraint configuration
    
    Returns:
        dict mapping constraint names to boolean mask arrays
    """
    masks = {}
    X_grid = terrain_data['X_grid']
    Y_grid = terrain_data['Y_grid']
    
    # Roads proximity (inclusion mask - points within distance are valid)
    if config.get('roads_enabled', False) and config.get('roads_data') is not None:
        max_dist = config.get('roads_max_distance_m', 500.0)
        roads_metric = config['roads_data']
        masks[f'Roads Proximity (<{max_dist:.0f}m)'] = mask_roads_proximity_fast(
            X_grid, Y_grid, roads_metric, max_distance_m=max_dist
        )
    
    # Buildings exclusion (exclusion mask - points within distance are invalid)
    if config.get('buildings_enabled', False) and config.get('buildings_data') is not None:
        exclusion_radius = config.get('buildings_exclusion_m', 1000.0)
        buffer_m = config.get('buildings_buffer_m', 100.0)
        effective_radius = max(0.0, exclusion_radius - buffer_m)
        buildings_metric = config['buildings_data']
        masks[f'Buildings Exclusion (>{exclusion_radius:.0f}m)'] = mask_buildings_exclusion_fast(
            X_grid, Y_grid, buildings_metric, radius_m=effective_radius
        )
    
    # Residential areas exclusion
    if config.get('residential_enabled', False) and config.get('residential_file') is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson', mode='w') as tmp:
            json.dump(config['residential_file'], tmp)
            tmp_path = tmp.name
        try:
            masks['Residential Exclusion'] = mask_residential_from_geojson(X_grid, Y_grid, tmp_path)
        finally:
            os.unlink(tmp_path)
    
    # Protected areas exclusion
    if config.get('protected_enabled', False) and config.get('protected_file') is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson', mode='w') as tmp:
            json.dump(config['protected_file'], tmp)
            tmp_path = tmp.name
        try:
            masks['Protected Areas Exclusion'] = mask_protected_areas_from_geojson(X_grid, Y_grid, tmp_path)
        finally:
            os.unlink(tmp_path)
    
    # Electric stations proximity (inclusion mask - points within distance are valid)
    if config.get('elec_stations_enabled', False) and config.get('elec_stations_data') is not None:
        distance_m = config.get('elec_stations_distance_m', 500.0)
        stations_metric = config['elec_stations_data']
        masks[f'Electric Stations Proximity (<{distance_m:.0f}m)'] = mask_electrical_proximity_fast(
            X_grid, Y_grid, stations_metric, radius_m=distance_m
        )
    
    return masks


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_single_mask(mask, lats, lons, Z_raw, title, radar_lat=None, radar_lon=None,
                     bg_style="terrain", excluded_alpha=0.6, figsize=(6, 5)):
    """
    Plot a single constraint mask overlaid on terrain/basemap.
    
    Args:
        mask: boolean array (True = admissible, False = excluded)
        lats, lons: coordinate arrays
        Z_raw: terrain elevation
        title: plot title
        radar_lat, radar_lon: optional radar position marker
        bg_style: "terrain" or "satellite" or "streets"
        excluded_alpha: opacity for excluded areas
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG_PANEL)
    ax.set_facecolor(BG_PANEL)
    
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Background layer
    if bg_style == "terrain":
        # Terrain contour background
        Z_plot = np.ma.masked_where(Z_raw < -100, Z_raw)
        ax.contourf(lon_grid, lat_grid, Z_plot, levels=20, cmap='terrain', alpha=1.0)
    elif bg_style in ["satellite", "streets"] and HAS_CONTEXTILY:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        try:
            if bg_style == "satellite":
                cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.Esri.WorldImagery, attribution=False)
            else:
                cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Voyager, attribution=False)
        except Exception:
            Z_plot = np.ma.masked_where(Z_raw < -100, Z_raw)
            ax.contourf(lon_grid, lat_grid, Z_plot, levels=20, cmap='terrain', alpha=1.0)
    else:
        # Fallback to terrain
        Z_plot = np.ma.masked_where(Z_raw < -100, Z_raw)
        ax.contourf(lon_grid, lat_grid, Z_plot, levels=20, cmap='terrain', alpha=1.0)
    
    # Mask overlay - excluded areas in grey
    excluded_mask = ~mask
    overlay = np.zeros((*mask.shape, 4))
    overlay[excluded_mask] = (*[int(EXCLUDED_COLOR[i:i+2], 16)/255 for i in (1, 3, 5)], excluded_alpha)
    overlay[~excluded_mask] = (0, 0, 0, 0)  # Transparent for admissible
    
    ax.imshow(overlay, aspect='auto', origin='lower', extent=extent, interpolation='nearest', zorder=2)
    
    # Markers
    if radar_lat is not None and radar_lon is not None:
        ax.plot(radar_lon, radar_lat, '*', color=ACCENT_PRIMARY, markersize=12,
                markeredgecolor='white', markeredgewidth=1, zorder=10, label='Radar')
    
    ax.plot(AIRPORT_LON, AIRPORT_LAT, '^', color=TEXT_PRIMARY, markersize=10,
            markeredgecolor='black', markeredgewidth=0.8, zorder=10, label='Airport')
    
    # Statistics
    admissible_pct = np.sum(mask) / mask.size * 100
    ax.text(0.02, 0.98, f'{admissible_pct:.1f}% valid', transform=ax.transAxes,
            fontsize=9, va='top', color=TEXT_PRIMARY, fontweight='600',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=BG_PANEL, alpha=0.9, edgecolor=BORDER_COLOR))
    
    # Styling
    ax.set_xlabel('Longitude (°)', fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel('Latitude (°)', fontsize=9, color=TEXT_SECONDARY)
    ax.set_title(title, fontsize=11, fontweight='600', color=ACCENT_PRIMARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    ax.legend(loc='upper right', fontsize=8, facecolor=BG_PANEL, edgecolor=BORDER_COLOR, labelcolor=TEXT_PRIMARY)
    
    plt.tight_layout()
    return fig


def plot_combined_mask(mask, lats, lons, Z_raw, radar_lat=None, radar_lon=None,
                       bg_style="terrain", admissible_alpha=0.5, excluded_alpha=0.6, figsize=(10, 8)):
    """
    Plot the combined (final) constraint mask with larger size.
    Only excluded areas are shown in grey, valid areas are transparent.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG_PANEL)
    ax.set_facecolor(BG_PANEL)
    
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Background
    if bg_style == "terrain":
        Z_plot = np.ma.masked_where(Z_raw < -100, Z_raw)
        terrain_im = ax.contourf(lon_grid, lat_grid, Z_plot, levels=30, cmap='terrain', alpha=1.0)
        cbar = plt.colorbar(terrain_im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('Elevation (m)', fontsize=10, color=TEXT_SECONDARY)
        cbar.ax.tick_params(colors=TEXT_SECONDARY)
    elif bg_style in ["satellite", "streets"] and HAS_CONTEXTILY:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        try:
            if bg_style == "satellite":
                cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.Esri.WorldImagery, attribution=False)
            else:
                cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Voyager, attribution=False)
        except Exception:
            Z_plot = np.ma.masked_where(Z_raw < -100, Z_raw)
            ax.contourf(lon_grid, lat_grid, Z_plot, levels=30, cmap='terrain', alpha=1.0)
    else:
        Z_plot = np.ma.masked_where(Z_raw < -100, Z_raw)
        ax.contourf(lon_grid, lat_grid, Z_plot, levels=30, cmap='terrain', alpha=1.0)
    
    # Mask overlay - only excluded areas in grey, valid areas transparent
    excluded_mask = ~mask
    overlay = np.zeros((*mask.shape, 4))
    overlay[excluded_mask] = (*[int(EXCLUDED_COLOR[i:i+2], 16)/255 for i in (1, 3, 5)], excluded_alpha)
    overlay[~excluded_mask] = (0, 0, 0, 0)  # Transparent for valid
    
    ax.imshow(overlay, aspect='auto', origin='lower', extent=extent, interpolation='nearest', zorder=2)
    
    # Markers
    if radar_lat is not None and radar_lon is not None:
        ax.plot(radar_lon, radar_lat, '*', color=ACCENT_PRIMARY, markersize=16,
                markeredgecolor='white', markeredgewidth=1.5, zorder=10, label='Radar')
    
    ax.plot(AIRPORT_LON, AIRPORT_LAT, '^', color=TEXT_PRIMARY, markersize=12,
            markeredgecolor='black', markeredgewidth=1, zorder=10, label='Airport')
    
    # Statistics
    admissible_count = np.sum(mask)
    total_count = mask.size
    admissible_pct = admissible_count / total_count * 100
    
    stats_text = f'Valid Sites: {admissible_count:,} ({admissible_pct:.2f}%)'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, va='top', color=TEXT_PRIMARY, fontweight='600',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=BG_PANEL, alpha=0.95, edgecolor=BORDER_COLOR))
    
    # Styling
    ax.set_xlabel('Longitude (°)', fontsize=11, color=TEXT_SECONDARY)
    ax.set_ylabel('Latitude (°)', fontsize=11, color=TEXT_SECONDARY)
    ax.set_title('Combined Constraints - Valid Radar Sites', fontsize=14, fontweight='700', color=ACCENT_PRIMARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    
    # Legend for markers and excluded areas
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=EXCLUDED_COLOR, alpha=excluded_alpha, edgecolor=BORDER_COLOR, label='Excluded'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
              facecolor=BG_PANEL, edgecolor=BORDER_COLOR, labelcolor=TEXT_PRIMARY)
    
    plt.tight_layout()
    return fig


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_results_npz(masks_dict, combined_mask, terrain_data):
    """Export results to NPZ file."""
    lats = terrain_data['lats']
    lons = terrain_data['lons']
    Z_raw = terrain_data['Z_raw']
    
    # Get indices of valid points
    i_rows, j_cols = np.where(combined_mask)
    
    buf = BytesIO()
    np.savez(
        buf,
        lat=lats[i_rows],
        lon=lons[j_cols],
        z=Z_raw[i_rows, j_cols],
        combined_mask=combined_mask,
        **{f'mask_{k.replace(" ", "_")}': v for k, v in masks_dict.items()}
    )
    return buf.getvalue()


def export_results_csv(masks_dict, combined_mask, terrain_data):
    """Export constraint statistics to CSV."""
    lines = []
    lines.append("# Radar Site Selection - Constraint Analysis")
    lines.append(f"# Reference: Nice Airport ({AIRPORT_LAT}°N, {AIRPORT_LON}°E)")
    lines.append("")
    lines.append("Constraint,Type,Points Excluded,% Territory Excluded,Admissible Points,% Remaining")
    
    total_points = combined_mask.size
    cumulative_mask = np.ones(combined_mask.shape, dtype=bool)
    
    for name, mask in masks_dict.items():
        # Determine constraint type
        if 'Proximity' in name or 'km Radius' in name:
            ctype = 'Inclusion'
        else:
            ctype = 'Exclusion'
        
        # Apply this constraint
        new_cumulative = np.logical_and(cumulative_mask, mask)
        
        # Calculate stats
        points_excluded = np.sum(cumulative_mask) - np.sum(new_cumulative)
        pct_excluded = points_excluded / total_points * 100
        admissible = np.sum(new_cumulative)
        pct_remaining = admissible / total_points * 100
        
        lines.append(f'"{name}",{ctype},{points_excluded},{pct_excluded:.4f},{admissible},{pct_remaining:.4f}')
        
        cumulative_mask = new_cumulative
    
    # Final row
    final_admissible = np.sum(combined_mask)
    lines.append("")
    lines.append(f'"FINAL RESULT",Combined,{total_points - final_admissible},{(total_points - final_admissible)/total_points*100:.4f},{final_admissible},{final_admissible/total_points*100:.4f}')
    
    return "\n".join(lines)


def export_candidates_csv(combined_mask, terrain_data):
    """Export candidate coordinates to CSV."""
    lats = terrain_data['lats']
    lons = terrain_data['lons']
    Z_raw = terrain_data['Z_raw']
    
    lines = []
    lines.append("# Valid Radar Site Candidates")
    lines.append("latitude,longitude,elevation_m")
    
    i_rows, j_cols = np.where(combined_mask)
    
    # Downsample if too many points
    max_points = 10000
    if len(i_rows) > max_points:
        step = len(i_rows) // max_points
        i_rows = i_rows[::step]
        j_cols = j_cols[::step]
    
    for i, j in zip(i_rows, j_cols):
        lines.append(f"{lats[i]:.6f},{lons[j]:.6f},{Z_raw[i, j]:.1f}")
    
    return "\n".join(lines)


# ============================================================================
# GEOJSON VALIDATION HELPERS
# ============================================================================

class GeoJSONValidationResult:
    """Holds the result of GeoJSON validation."""
    def __init__(self, valid: bool, data=None, message: str = "", 
                 feature_count: int = 0, geometry_types: dict = None):
        self.valid = valid
        self.data = data
        self.message = message
        self.feature_count = feature_count
        self.geometry_types = geometry_types or {}


def validate_geojson_structure(geojson_data) -> GeoJSONValidationResult:
    """
    Validate basic GeoJSON structure.
    
    Returns:
        GeoJSONValidationResult with validation status and details.
    """
    if not isinstance(geojson_data, dict):
        return GeoJSONValidationResult(False, message="Invalid JSON: not a dictionary")
    
    # Check for FeatureCollection
    if geojson_data.get('type') == 'FeatureCollection':
        features = geojson_data.get('features', [])
        if not isinstance(features, list):
            return GeoJSONValidationResult(False, message="Invalid FeatureCollection: 'features' is not a list")
        
        # Count geometry types
        geometry_types = {}
        for feat in features:
            if isinstance(feat, dict) and 'geometry' in feat:
                geom = feat.get('geometry', {})
                if geom and isinstance(geom, dict):
                    gtype = geom.get('type', 'Unknown')
                    geometry_types[gtype] = geometry_types.get(gtype, 0) + 1
        
        return GeoJSONValidationResult(
            True, 
            data=geojson_data, 
            message="Valid FeatureCollection",
            feature_count=len(features),
            geometry_types=geometry_types
        )
    
    # Check for single Feature
    elif geojson_data.get('type') == 'Feature':
        geom = geojson_data.get('geometry', {})
        gtype = geom.get('type', 'Unknown') if geom else 'None'
        return GeoJSONValidationResult(
            True,
            data={'type': 'FeatureCollection', 'features': [geojson_data]},
            message="Valid Feature (converted to FeatureCollection)",
            feature_count=1,
            geometry_types={gtype: 1}
        )
    
    # Check for direct Geometry
    elif geojson_data.get('type') in ['Point', 'LineString', 'Polygon', 'MultiPoint', 
                                       'MultiLineString', 'MultiPolygon', 'GeometryCollection']:
        gtype = geojson_data.get('type')
        wrapped = {
            'type': 'FeatureCollection',
            'features': [{'type': 'Feature', 'geometry': geojson_data, 'properties': {}}]
        }
        return GeoJSONValidationResult(
            True,
            data=wrapped,
            message=f"Valid {gtype} geometry (wrapped in FeatureCollection)",
            feature_count=1,
            geometry_types={gtype: 1}
        )
    
    else:
        return GeoJSONValidationResult(
            False, 
            message=f"Invalid GeoJSON type: '{geojson_data.get('type', 'missing')}'"
        )


def validate_roads_geojson(geojson_data) -> GeoJSONValidationResult:
    """
    Validate GeoJSON for roads constraint. 
    Expects LineString geometries.
    """
    base_result = validate_geojson_structure(geojson_data)
    if not base_result.valid:
        return base_result
    
    # Check for LineString geometries
    linestring_count = base_result.geometry_types.get('LineString', 0)
    multilinestring_count = base_result.geometry_types.get('MultiLineString', 0)
    total_roads = linestring_count + multilinestring_count
    
    if total_roads == 0:
        types_found = ', '.join(base_result.geometry_types.keys()) or 'none'
        return GeoJSONValidationResult(
            False,
            message=f"No road geometries found. Expected LineString, found: {types_found}"
        )
    
    return GeoJSONValidationResult(
        True,
        data=base_result.data,
        message=f"Valid roads GeoJSON",
        feature_count=base_result.feature_count,
        geometry_types=base_result.geometry_types
    )


def validate_buildings_geojson(geojson_data) -> GeoJSONValidationResult:
    """
    Validate GeoJSON for buildings constraint.
    Expects Point or Polygon geometries (centroids computed from polygons).
    """
    base_result = validate_geojson_structure(geojson_data)
    if not base_result.valid:
        return base_result
    
    # Check for Point or Polygon geometries
    point_count = base_result.geometry_types.get('Point', 0)
    polygon_count = base_result.geometry_types.get('Polygon', 0)
    multipolygon_count = base_result.geometry_types.get('MultiPolygon', 0)
    total_buildings = point_count + polygon_count + multipolygon_count
    
    if total_buildings == 0:
        types_found = ', '.join(base_result.geometry_types.keys()) or 'none'
        return GeoJSONValidationResult(
            False,
            message=f"No building geometries found. Expected Point or Polygon, found: {types_found}"
        )
    
    return GeoJSONValidationResult(
        True,
        data=base_result.data,
        message=f"Valid buildings GeoJSON",
        feature_count=base_result.feature_count,
        geometry_types=base_result.geometry_types
    )


def validate_polygon_geojson(geojson_data, constraint_name="area") -> GeoJSONValidationResult:
    """
    Validate GeoJSON for polygon-based constraints (residential, protected areas).
    Expects Polygon or MultiPolygon geometries.
    """
    base_result = validate_geojson_structure(geojson_data)
    if not base_result.valid:
        return base_result
    
    # Check for Polygon geometries
    polygon_count = base_result.geometry_types.get('Polygon', 0)
    multipolygon_count = base_result.geometry_types.get('MultiPolygon', 0)
    total_polygons = polygon_count + multipolygon_count
    
    if total_polygons == 0:
        types_found = ', '.join(base_result.geometry_types.keys()) or 'none'
        return GeoJSONValidationResult(
            False,
            message=f"No {constraint_name} polygons found. Expected Polygon, found: {types_found}"
        )
    
    return GeoJSONValidationResult(
        True,
        data=base_result.data,
        message=f"Valid {constraint_name} GeoJSON",
        feature_count=base_result.feature_count,
        geometry_types=base_result.geometry_types
    )


# ============================================================================
# GEOJSON PARSING HELPERS
# ============================================================================

def parse_roads_geojson(geojson_data, major_only=True):
    """Parse roads GeoJSON and convert to metric coordinates."""
    lat_ref_rad = np.radians(REF_LAT)
    meters_per_deg_lat = (np.pi / 180.0) * GEO_EARTH_RADIUS_M
    meters_per_deg_lon = (np.pi / 180.0) * GEO_EARTH_RADIUS_M * np.cos(lat_ref_rad)
    
    roads_enu = []
    major_road_types = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary'}
    
    def process_coords(coords_list):
        road_metric = []
        for pt in coords_list:
            lon, lat = pt[0], pt[1]
            y_m = (lat - REF_LAT) * meters_per_deg_lat
            x_m = (lon - REF_LON) * meters_per_deg_lon
            road_metric.append((x_m, y_m))
        return road_metric
    
    if 'features' in geojson_data:
        for feature in geojson_data['features']:
            geom = feature.get('geometry', {})
            if not geom:
                continue
            
            if geom.get('type') == 'LineString':
                if major_only:
                    props = feature.get('properties', {})
                    road_type = props.get('highway', '').lower()
                    if road_type not in major_road_types:
                        continue
                raw_coords = geom.get('coordinates', [])
                if raw_coords:
                    roads_enu.append(process_coords(raw_coords))
            
            elif geom.get('type') == 'MultiLineString':
                if major_only:
                    props = feature.get('properties', {})
                    road_type = props.get('highway', '').lower()
                    if road_type not in major_road_types:
                        continue
                for line_coords in geom.get('coordinates', []):
                    if line_coords:
                        roads_enu.append(process_coords(line_coords))
    
    return roads_enu


def parse_buildings_geojson(geojson_data):
    """Parse buildings GeoJSON and convert to metric coordinates (supports Point and Polygon)."""
    lat_ref_rad = np.radians(REF_LAT)
    meters_per_deg_lat = (np.pi / 180.0) * GEO_EARTH_RADIUS_M
    meters_per_deg_lon = (np.pi / 180.0) * GEO_EARTH_RADIUS_M * np.cos(lat_ref_rad)
    
    buildings_enu = []
    
    def convert_point(lon, lat):
        y_m = (lat - REF_LAT) * meters_per_deg_lat
        x_m = (lon - REF_LON) * meters_per_deg_lon
        return {'x_m': x_m, 'y_m': y_m}
    
    def compute_centroid(coords):
        """Compute centroid of a polygon ring."""
        if not coords:
            return None
        lons = [pt[0] for pt in coords]
        lats = [pt[1] for pt in coords]
        return np.mean(lons), np.mean(lats)
    
    features = geojson_data.get('features', [])
    for feat in features:
        geom = feat.get('geometry', {})
        if not geom:
            continue
        
        gtype = geom.get('type')
        
        if gtype == 'Point':
            coords = geom.get('coordinates', None)
            if coords and len(coords) >= 2:
                lon, lat = float(coords[0]), float(coords[1])
                buildings_enu.append(convert_point(lon, lat))
        
        elif gtype == 'Polygon':
            # Use centroid of exterior ring
            rings = geom.get('coordinates', [])
            if rings and len(rings) > 0:
                centroid = compute_centroid(rings[0])
                if centroid:
                    buildings_enu.append(convert_point(centroid[0], centroid[1]))
        
        elif gtype == 'MultiPolygon':
            # Use centroid of each polygon
            for polygon_rings in geom.get('coordinates', []):
                if polygon_rings and len(polygon_rings) > 0:
                    centroid = compute_centroid(polygon_rings[0])
                    if centroid:
                        buildings_enu.append(convert_point(centroid[0], centroid[1]))
    
    return buildings_enu


def render_geojson_validation_feedback(validation_result: GeoJSONValidationResult, constraint_name: str):
    """Render validation feedback in Streamlit UI."""
    if validation_result.valid:
        # Success message with details
        geom_summary = ", ".join([f"{count} {gtype}" for gtype, count in validation_result.geometry_types.items()])
        st.success(f"Loaded {validation_result.feature_count} features ({geom_summary})")
    else:
        # Error message
        st.error(f"Invalid {constraint_name} file: {validation_result.message}")


def render_user_constraint_card(title: str, description: str, key_prefix: str,
                                 param_label: str, param_min: int, param_max: int, 
                                 param_default: int, param_step: int,
                                 expected_geometry: str, validator_func,
                                 additional_options: dict = None):
    """
    Render a user-defined constraint card with consistent UI pattern.
    
    Args:
        title: Constraint title
        description: Short description
        key_prefix: Unique key prefix for Streamlit widgets
        param_label: Label for the distance/rule parameter
        param_min, param_max, param_default, param_step: Parameter input settings
        expected_geometry: Description of expected geometry type
        validator_func: Function to validate the uploaded GeoJSON
        additional_options: Dict of additional checkbox options {label: default_value}
    
    Returns:
        tuple: (enabled: bool, parameter_value: number, validated_data: dict or None, options: dict)
    """
    st.markdown(f"**{title}**")
    st.caption(description)
    
    enabled = st.checkbox(f"Enable {title.lower()}", value=False, key=f'{key_prefix}_enabled')
    
    validated_data = None
    param_value = param_default
    options = {}
    
    if enabled:
        # Parameter input
        param_value = st.number_input(
            param_label, 
            min_value=param_min, 
            max_value=param_max, 
            value=param_default, 
            step=param_step,
            key=f'{key_prefix}_param'
        )
        
        # Additional options
        if additional_options:
            for opt_label, opt_default in additional_options.items():
                opt_key = opt_label.lower().replace(' ', '_')
                options[opt_key] = st.checkbox(opt_label, value=opt_default, key=f'{key_prefix}_{opt_key}')
        
        # File upload
        st.caption(f"Expected format: GeoJSON with {expected_geometry}")
        uploaded_file = st.file_uploader(
            f"Upload {title} GeoJSON",
            type=['geojson', 'json'],
            key=f'{key_prefix}_file',
            help=f"Upload a GeoJSON file containing {expected_geometry} geometries"
        )
        
        if uploaded_file:
            try:
                geojson_data = json.load(uploaded_file)
                validation_result = validator_func(geojson_data)
                render_geojson_validation_feedback(validation_result, title.lower())
                
                if validation_result.valid:
                    validated_data = validation_result.data
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON file: {str(e)[:100]}")
            except Exception as e:
                st.error(f"Error reading file: {str(e)[:100]}")
        else:
            st.info(f"Upload a GeoJSON file to enable this constraint")
    
    return enabled, param_value, validated_data, options


# ============================================================================
# MAIN PAGE
# ============================================================================

def main():
    st.set_page_config(
        page_title="Site Selection - Radar Coverage",
        page_icon="📍",
        layout="wide"
    )
    apply_theme()
    render_page_title()
    
    # Check module availability
    if not HAS_MASK_MODULES:
        st.error(f"Required mask modules not available: {MASK_IMPORT_ERROR}")
        st.info("Please ensure all mask modules are properly installed.")
        return
    
    # Initialize session state
    if 'site_masks_computed' not in st.session_state:
        st.session_state.site_masks_computed = False
    if 'site_terrain_data' not in st.session_state:
        st.session_state.site_terrain_data = None
    if 'site_masks_dict' not in st.session_state:
        st.session_state.site_masks_dict = {}
    if 'site_combined_mask' not in st.session_state:
        st.session_state.site_combined_mask = None
    
    # =========================================================================
    # SIDEBAR - CONSTRAINT CONFIGURATION
    # =========================================================================
    with st.sidebar:
        st.header("Configuration")
        
        # Terrain Data
        with st.expander("Terrain Data", expanded=True):
            uploaded = st.file_uploader("Upload NPZ terrain file", type=['npz'], key='site_terrain_upload')
            use_sample = st.checkbox("Use sample terrain", value=not uploaded, key='site_use_sample')
        
        # Load terrain if available
        terrain_data = None
        if uploaded:
            try:
                terrain_data = load_terrain_data("uploaded", uploaded)
                st.session_state.site_terrain_data = terrain_data
            except Exception as e:
                st.error(f"Error loading terrain: {e}")
        elif use_sample:
            try:
                terrain_data = load_terrain_data("sample")
                st.session_state.site_terrain_data = terrain_data
            except Exception as e:
                st.error(f"Error loading sample terrain: {e}")
        
        if terrain_data is None and st.session_state.site_terrain_data is not None:
            terrain_data = st.session_state.site_terrain_data
        
        if terrain_data:
            st.success(f"Terrain loaded: {terrain_data['Z_raw'].shape[0]}×{terrain_data['Z_raw'].shape[1]} grid")
            
            # =================================================================
            # PREDEFINED CONSTRAINTS
            # =================================================================
            with st.expander("Predefined Constraints", expanded=True):
                st.markdown("**Geographic Constraints**")
                
                land_enabled = st.checkbox("Land only (exclude sea)", value=True, key='land_enabled')
                
                radius_enabled = st.checkbox("Radius from reference", value=True, key='radius_enabled')
                radius_km = 50.0
                if radius_enabled:
                    radius_km = st.slider("Radius (km)", 10.0, 100.0, 50.0, 5.0, key='radius_km')
                
                french_enabled = st.checkbox("French territory only", value=False, key='french_enabled')
                
                st.markdown("**Coastal Constraints**")
                coastline_enabled = st.checkbox("Coastline buffer", value=False, key='coastline_enabled')
                coastline_buffer_m = 100.0
                if coastline_enabled:
                    coastline_buffer_m = st.number_input("Buffer distance (m)", 50, 1000, 100, 50, key='coastline_buffer')
                
                st.markdown("**Terrain Constraints**")
                slope_enabled = st.checkbox("Maximum slope", value=False, key='slope_enabled')
                max_slope_percent = 15.0
                if slope_enabled:
                    max_slope_percent = st.slider("Max slope (%)", 5.0, 30.0, 15.0, 1.0, key='max_slope')
            
            # =================================================================
            # USER-DEFINED CONSTRAINTS
            # =================================================================
            with st.expander("User-Defined Constraints (GeoJSON)", expanded=False):
                st.markdown("""
                Upload GeoJSON files to define custom geographic constraints.
                Each constraint type expects specific geometry types.
                """)
                
                # Constraint type selector
                st.markdown("---")
                constraint_type = st.selectbox(
                    "Select constraint type to configure",
                    ["Roads Proximity", "Buildings Exclusion", "Residential Exclusion", "Protected Areas", "Electric Stations Proximity"],
                    key='constraint_type_selector',
                    help="Choose a constraint type to configure. Each can be enabled independently."
                )
                
                # Initialize constraint data variables
                roads_enabled = st.session_state.get('roads_constraint_enabled', False)
                roads_data = st.session_state.get('roads_constraint_data', None)
                roads_max_distance = st.session_state.get('roads_constraint_distance', 500)
                roads_major_only = st.session_state.get('roads_constraint_major', True)
                
                buildings_enabled = st.session_state.get('buildings_constraint_enabled', False)
                buildings_data = st.session_state.get('buildings_constraint_data', None)
                buildings_exclusion = st.session_state.get('buildings_constraint_exclusion', 1000)
                buildings_buffer = st.session_state.get('buildings_constraint_buffer', 100)
                
                residential_enabled = st.session_state.get('residential_constraint_enabled', False)
                residential_file_data = st.session_state.get('residential_constraint_data', None)
                
                protected_enabled = st.session_state.get('protected_constraint_enabled', False)
                protected_file_data = st.session_state.get('protected_constraint_data', None)
                
                elec_stations_enabled = st.session_state.get('elec_stations_constraint_enabled', False)
                elec_stations_data = st.session_state.get('elec_stations_constraint_data', None)
                elec_stations_distance = st.session_state.get('elec_stations_constraint_distance', 500)
                
                st.markdown("---")
                
                # ===============================================================
                # ROADS PROXIMITY CONSTRAINT
                # ===============================================================
                if constraint_type == "Roads Proximity":
                    st.markdown("### Roads Proximity")
                    st.caption("Include only locations within a specified distance of roads (OpenStreetMap format)")
                    
                    roads_enabled = st.checkbox(
                        "Enable roads proximity constraint", 
                        value=roads_enabled, 
                        key='roads_enabled_ui',
                        help="When enabled, only locations near roads are considered valid"
                    )
                    st.session_state['roads_constraint_enabled'] = roads_enabled
                    
                    if roads_enabled:
                        col1, col2 = st.columns(2)
                        with col1:
                            roads_max_distance = st.number_input(
                                "Maximum distance (m)", 
                                min_value=50, 
                                max_value=5000, 
                                value=int(roads_max_distance), 
                                step=50,
                                key='roads_dist_ui',
                                help="Sites must be within this distance from a road"
                            )
                            st.session_state['roads_constraint_distance'] = roads_max_distance
                        
                        with col2:
                            roads_major_only = st.checkbox(
                                "Major roads only", 
                                value=roads_major_only, 
                                key='roads_major_ui',
                                help="Filter to motorway, trunk, primary, secondary, tertiary"
                            )
                            st.session_state['roads_constraint_major'] = roads_major_only
                        
                        # Geometry info
                        st.info("**Expected geometry:** LineString or MultiLineString (roads from OpenStreetMap)")
                        
                        # File upload
                        roads_file = st.file_uploader(
                            "Upload roads GeoJSON", 
                            type=['geojson', 'json'], 
                            key='roads_file_ui',
                            help="GeoJSON file with road network data"
                        )
                        
                        if roads_file:
                            try:
                                roads_geojson = json.load(roads_file)
                                validation = validate_roads_geojson(roads_geojson)
                                render_geojson_validation_feedback(validation, "roads")
                                
                                if validation.valid:
                                    roads_data = parse_roads_geojson(validation.data, major_only=roads_major_only)
                                    st.session_state['roads_constraint_data'] = roads_data
                                    
                                    if roads_data:
                                        st.success(f"Parsed {len(roads_data)} road segments for mask computation")
                                    else:
                                        if roads_major_only:
                                            st.warning("No major roads found. Try disabling 'Major roads only' filter.")
                                        else:
                                            st.warning("No road segments parsed from the file.")
                                else:
                                    st.session_state['roads_constraint_data'] = None
                            except json.JSONDecodeError as e:
                                st.error(f"Invalid JSON format: {str(e)[:100]}")
                            except Exception as e:
                                st.error(f"Error processing file: {str(e)[:100]}")
                        else:
                            st.warning("Please upload a roads GeoJSON file to apply this constraint")
                            if st.session_state.get('roads_constraint_data'):
                                st.info("Using previously loaded roads data")
                
                # ===============================================================
                # BUILDINGS EXCLUSION CONSTRAINT
                # ===============================================================
                elif constraint_type == "Buildings Exclusion":
                    st.markdown("### Buildings Exclusion")
                    st.caption("Exclude locations within a specified distance of buildings/structures")
                    
                    buildings_enabled = st.checkbox(
                        "Enable buildings exclusion constraint", 
                        value=buildings_enabled, 
                        key='buildings_enabled_ui',
                        help="When enabled, locations near buildings are excluded"
                    )
                    st.session_state['buildings_constraint_enabled'] = buildings_enabled
                    
                    if buildings_enabled:
                        col1, col2 = st.columns(2)
                        with col1:
                            buildings_exclusion = st.number_input(
                                "Exclusion radius (m)", 
                                min_value=100, 
                                max_value=5000, 
                                value=int(buildings_exclusion), 
                                step=100,
                                key='buildings_radius_ui',
                                help="Minimum distance required from buildings"
                            )
                            st.session_state['buildings_constraint_exclusion'] = buildings_exclusion
                        
                        with col2:
                            buildings_buffer = st.number_input(
                                "Tolerance buffer (m)", 
                                min_value=0, 
                                max_value=500, 
                                value=int(buildings_buffer), 
                                step=25,
                                key='buildings_buffer_ui',
                                help="Buffer subtracted from radius (makes constraint more permissive)"
                            )
                            st.session_state['buildings_constraint_buffer'] = buildings_buffer
                        
                        effective_radius = max(0, buildings_exclusion - buildings_buffer)
                        st.caption(f"Effective exclusion radius: {effective_radius}m")
                        
                        # Geometry info
                        st.info("**Expected geometry:** Point (building locations) or Polygon (building footprints - centroids computed)")
                        
                        # File upload
                        buildings_file = st.file_uploader(
                            "Upload buildings GeoJSON", 
                            type=['geojson', 'json'], 
                            key='buildings_file_ui',
                            help="GeoJSON file with building locations or footprints"
                        )
                        
                        if buildings_file:
                            try:
                                buildings_geojson = json.load(buildings_file)
                                validation = validate_buildings_geojson(buildings_geojson)
                                render_geojson_validation_feedback(validation, "buildings")
                                
                                if validation.valid:
                                    buildings_data = parse_buildings_geojson(validation.data)
                                    st.session_state['buildings_constraint_data'] = buildings_data
                                    
                                    if buildings_data:
                                        st.success(f"Parsed {len(buildings_data)} building locations for mask computation")
                                    else:
                                        st.warning("No building points extracted from the file.")
                                else:
                                    st.session_state['buildings_constraint_data'] = None
                            except json.JSONDecodeError as e:
                                st.error(f"Invalid JSON format: {str(e)[:100]}")
                            except Exception as e:
                                st.error(f"Error processing file: {str(e)[:100]}")
                        else:
                            st.warning("Please upload a buildings GeoJSON file to apply this constraint")
                            if st.session_state.get('buildings_constraint_data'):
                                st.info("Using previously loaded buildings data")
                
                # ===============================================================
                # RESIDENTIAL EXCLUSION CONSTRAINT
                # ===============================================================
                elif constraint_type == "Residential Exclusion":
                    st.markdown("### Residential Areas Exclusion")
                    st.caption("Exclude locations within residential zone polygons")
                    
                    residential_enabled = st.checkbox(
                        "Enable residential exclusion constraint", 
                        value=residential_enabled, 
                        key='residential_enabled_ui',
                        help="When enabled, locations inside residential polygons are excluded"
                    )
                    st.session_state['residential_constraint_enabled'] = residential_enabled
                    
                    if residential_enabled:
                        # Geometry info
                        st.info("**Expected geometry:** Polygon or MultiPolygon (residential zone boundaries)")
                        
                        # File upload
                        residential_file = st.file_uploader(
                            "Upload residential zones GeoJSON", 
                            type=['geojson', 'json'], 
                            key='residential_file_ui',
                            help="GeoJSON file with residential area polygons"
                        )
                        
                        if residential_file:
                            try:
                                residential_geojson = json.load(residential_file)
                                validation = validate_polygon_geojson(residential_geojson, "residential area")
                                render_geojson_validation_feedback(validation, "residential zones")
                                
                                if validation.valid:
                                    residential_file_data = validation.data
                                    st.session_state['residential_constraint_data'] = residential_file_data
                                    
                                    polygon_count = (validation.geometry_types.get('Polygon', 0) + 
                                                    validation.geometry_types.get('MultiPolygon', 0))
                                    st.success(f"Ready to process {polygon_count} residential polygons")
                                else:
                                    st.session_state['residential_constraint_data'] = None
                            except json.JSONDecodeError as e:
                                st.error(f"Invalid JSON format: {str(e)[:100]}")
                            except Exception as e:
                                st.error(f"Error processing file: {str(e)[:100]}")
                        else:
                            st.warning("Please upload a residential zones GeoJSON file to apply this constraint")
                            if st.session_state.get('residential_constraint_data'):
                                st.info("Using previously loaded residential data")
                
                # ===============================================================
                # PROTECTED AREAS EXCLUSION CONSTRAINT
                # ===============================================================
                elif constraint_type == "Protected Areas":
                    st.markdown("### Protected Areas Exclusion")
                    st.caption("Exclude locations within protected area polygons (parks, reserves, etc.)")
                    
                    protected_enabled = st.checkbox(
                        "Enable protected areas exclusion constraint", 
                        value=protected_enabled, 
                        key='protected_enabled_ui',
                        help="When enabled, locations inside protected areas are excluded"
                    )
                    st.session_state['protected_constraint_enabled'] = protected_enabled
                    
                    if protected_enabled:
                        # Geometry info
                        st.info("**Expected geometry:** Polygon or MultiPolygon (national parks, nature reserves, forests)")
                        
                        # File upload
                        protected_file = st.file_uploader(
                            "Upload protected areas GeoJSON", 
                            type=['geojson', 'json'], 
                            key='protected_file_ui',
                            help="GeoJSON file with protected area boundaries"
                        )
                        
                        if protected_file:
                            try:
                                protected_geojson = json.load(protected_file)
                                validation = validate_polygon_geojson(protected_geojson, "protected area")
                                render_geojson_validation_feedback(validation, "protected areas")
                                
                                if validation.valid:
                                    protected_file_data = validation.data
                                    st.session_state['protected_constraint_data'] = protected_file_data
                                    
                                    polygon_count = (validation.geometry_types.get('Polygon', 0) + 
                                                    validation.geometry_types.get('MultiPolygon', 0))
                                    st.success(f"Ready to process {polygon_count} protected area polygons")
                                else:
                                    st.session_state['protected_constraint_data'] = None
                            except json.JSONDecodeError as e:
                                st.error(f"Invalid JSON format: {str(e)[:100]}")
                            except Exception as e:
                                st.error(f"Error processing file: {str(e)[:100]}")
                        else:
                            st.warning("Please upload a protected areas GeoJSON file to apply this constraint")
                            if st.session_state.get('protected_constraint_data'):
                                st.info("Using previously loaded protected areas data")
                
                # ===============================================================
                # ELECTRIC STATIONS PROXIMITY CONSTRAINT
                # ===============================================================
                elif constraint_type == "Electric Stations Proximity":
                    st.markdown("### Electric Stations Proximity")
                    st.caption("Include only locations within a specified distance of electrical stations")
                    
                    elec_stations_enabled = st.checkbox(
                        "Enable electric stations proximity constraint", 
                        value=elec_stations_enabled, 
                        key='elec_stations_enabled_ui',
                        help="When enabled, only locations near electrical stations are considered valid"
                    )
                    st.session_state['elec_stations_constraint_enabled'] = elec_stations_enabled
                    
                    if elec_stations_enabled:
                        elec_stations_distance = st.number_input(
                            "Maximum distance from station (m)", 
                            min_value=100, 
                            max_value=5000, 
                            value=int(elec_stations_distance), 
                            step=100,
                            key='elec_stations_dist_ui',
                            help="Sites must be within this distance from an electrical station"
                        )
                        st.session_state['elec_stations_constraint_distance'] = elec_stations_distance
                        
                        # Geometry info
                        st.info("**Expected format:** JSON file with electrical station data (Enedis format with 'results' containing '_geopoint' fields)")
                        
                        # File upload
                        elec_file = st.file_uploader(
                            "Upload electrical stations JSON", 
                            type=['json'], 
                            key='elec_stations_file_ui',
                            help="JSON file with electrical station locations"
                        )
                        
                        if elec_file:
                            try:
                                elec_json = json.load(elec_file)
                                
                                # Validate structure
                                if 'results' not in elec_json:
                                    st.error("Invalid format: JSON must contain 'results' array")
                                    st.session_state['elec_stations_constraint_data'] = None
                                else:
                                    # Parse stations and convert to metric coordinates
                                    stations_count = len(elec_json['results'])
                                    
                                    # Parse stations inline (similar to load_stations_and_convert_to_enu)
                                    lat_ref_rad = np.radians(REF_LAT)
                                    meters_per_deg_lat = (np.pi / 180.0) * GEO_EARTH_RADIUS_M
                                    meters_per_deg_lon = (np.pi / 180.0) * GEO_EARTH_RADIUS_M * np.cos(lat_ref_rad)
                                    
                                    stations_enu = []
                                    for result in elec_json['results']:
                                        geopoint = result.get('_geopoint')
                                        if geopoint:
                                            if isinstance(geopoint, str):
                                                lat, lon = map(float, geopoint.split(','))
                                            else:
                                                lat = geopoint.get('lat')
                                                lon = geopoint.get('lon')
                                            
                                            if lat is not None and lon is not None:
                                                y_m = (lat - REF_LAT) * meters_per_deg_lat
                                                x_m = (lon - REF_LON) * meters_per_deg_lon
                                                stations_enu.append({'x_m': x_m, 'y_m': y_m})
                                    
                                    if stations_enu:
                                        elec_stations_data = stations_enu
                                        st.session_state['elec_stations_constraint_data'] = elec_stations_data
                                        st.success(f"Loaded {len(stations_enu)} electrical stations for mask computation")
                                    else:
                                        st.warning("No valid station coordinates found in the file")
                                        st.session_state['elec_stations_constraint_data'] = None
                                        
                            except json.JSONDecodeError as e:
                                st.error(f"Invalid JSON format: {str(e)[:100]}")
                            except Exception as e:
                                st.error(f"Error processing file: {str(e)[:100]}")
                        else:
                            st.warning("Please upload an electrical stations JSON file to apply this constraint")
                            if st.session_state.get('elec_stations_constraint_data'):
                                st.info("Using previously loaded electrical stations data")
                
                # ===============================================================
                # ACTIVE CONSTRAINTS SUMMARY
                # ===============================================================
                st.markdown("---")
                st.markdown("### Active User Constraints")
                
                active_constraints = []
                if roads_enabled and roads_data:
                    active_constraints.append(f"Roads Proximity (<{roads_max_distance}m, {len(roads_data)} segments)")
                elif roads_enabled:
                    active_constraints.append("Roads Proximity (no data loaded)")
                
                if buildings_enabled and buildings_data:
                    eff_rad = max(0, buildings_exclusion - buildings_buffer)
                    active_constraints.append(f"Buildings Exclusion (>{eff_rad}m, {len(buildings_data)} buildings)")
                elif buildings_enabled:
                    active_constraints.append("Buildings Exclusion (no data loaded)")
                
                if residential_enabled and residential_file_data:
                    n_feat = len(residential_file_data.get('features', []))
                    active_constraints.append(f"Residential Exclusion ({n_feat} zones)")
                elif residential_enabled:
                    active_constraints.append("Residential Exclusion (no data loaded)")
                
                if protected_enabled and protected_file_data:
                    n_feat = len(protected_file_data.get('features', []))
                    active_constraints.append(f"Protected Areas ({n_feat} areas)")
                elif protected_enabled:
                    active_constraints.append("Protected Areas (no data loaded)")
                
                if elec_stations_enabled and elec_stations_data:
                    active_constraints.append(f"Electric Stations Proximity (<{elec_stations_distance}m, {len(elec_stations_data)} stations)")
                elif elec_stations_enabled:
                    active_constraints.append("Electric Stations Proximity (no data loaded)")
                
                if active_constraints:
                    for constraint in active_constraints:
                        if "no data" in constraint:
                            st.markdown(f"- {constraint}")
                        else:
                            st.markdown(f"- {constraint}")
                else:
                    st.caption("No user-defined constraints active")
            
            # =================================================================
            # COMPUTE BUTTON
            # =================================================================
            st.markdown("---")
            
            # Retrieve user constraint data from session state
            user_roads_enabled = st.session_state.get('roads_constraint_enabled', False)
            user_roads_data = st.session_state.get('roads_constraint_data', None)
            user_roads_distance = st.session_state.get('roads_constraint_distance', 500)
            
            user_buildings_enabled = st.session_state.get('buildings_constraint_enabled', False)
            user_buildings_data = st.session_state.get('buildings_constraint_data', None)
            user_buildings_exclusion = st.session_state.get('buildings_constraint_exclusion', 1000)
            user_buildings_buffer = st.session_state.get('buildings_constraint_buffer', 100)
            
            user_residential_enabled = st.session_state.get('residential_constraint_enabled', False)
            user_residential_data = st.session_state.get('residential_constraint_data', None)
            
            user_protected_enabled = st.session_state.get('protected_constraint_enabled', False)
            user_protected_data = st.session_state.get('protected_constraint_data', None)
            
            user_elec_stations_enabled = st.session_state.get('elec_stations_constraint_enabled', False)
            user_elec_stations_data = st.session_state.get('elec_stations_constraint_data', None)
            user_elec_stations_distance = st.session_state.get('elec_stations_constraint_distance', 500)
            
            # Count active constraints for button feedback
            n_predefined = sum([land_enabled, radius_enabled, french_enabled, coastline_enabled, slope_enabled])
            n_user = sum([
                user_roads_enabled and user_roads_data is not None,
                user_buildings_enabled and user_buildings_data is not None,
                user_residential_enabled and user_residential_data is not None,
                user_protected_enabled and user_protected_data is not None,
                user_elec_stations_enabled and user_elec_stations_data is not None
            ])
            
            st.caption(f"Ready to apply: {n_predefined} predefined + {n_user} user-defined constraints")
            
            if st.button("Apply Constraints", type="primary", use_container_width=True, key='compute_masks'):
                with st.spinner("Computing masks..."):
                    # Build configuration
                    predefined_config = {
                        'land_enabled': land_enabled,
                        'radius_enabled': radius_enabled,
                        'radius_km': radius_km,
                        'french_enabled': french_enabled,
                        'coastline_enabled': coastline_enabled,
                        'coastline_buffer_m': coastline_buffer_m,
                        'slope_enabled': slope_enabled,
                        'max_slope_percent': max_slope_percent,
                    }
                    
                    user_config = {
                        'roads_enabled': user_roads_enabled,
                        'roads_data': user_roads_data,
                        'roads_max_distance_m': user_roads_distance,
                        'buildings_enabled': user_buildings_enabled,
                        'buildings_data': user_buildings_data,
                        'buildings_exclusion_m': user_buildings_exclusion,
                        'buildings_buffer_m': user_buildings_buffer,
                        'residential_enabled': user_residential_enabled,
                        'residential_file': user_residential_data,
                        'protected_enabled': user_protected_enabled,
                        'protected_file': user_protected_data,
                        'elec_stations_enabled': user_elec_stations_enabled,
                        'elec_stations_data': user_elec_stations_data,
                        'elec_stations_distance_m': user_elec_stations_distance,
                    }
                    
                    # Compute masks
                    predefined_masks = compute_predefined_masks(terrain_data, predefined_config)
                    user_masks = compute_user_masks(terrain_data, user_config)
                    
                    # Combine all masks
                    all_masks = {**predefined_masks, **user_masks}
                    
                    if all_masks:
                        combined = combine_masks(*all_masks.values())
                    else:
                        combined = np.ones(terrain_data['Z_raw'].shape, dtype=bool)
                    
                    # Store in session state
                    # Note: bg_style, excluded_alpha, admissible_alpha are managed by their widgets
                    # and are automatically stored in session state via their keys
                    st.session_state.site_masks_dict = all_masks
                    st.session_state.site_combined_mask = combined
                    st.session_state.site_masks_computed = True
                    st.session_state.site_terrain_data = terrain_data
                    
                st.success(f"Constraints applied! ({len(all_masks)} total)")
                st.rerun()
    
    # =========================================================================
    # MAIN CONTENT AREA
    # =========================================================================
    
    if terrain_data is None:
        st.info("Upload terrain data or enable sample terrain in the sidebar to begin.")
        return
    
    if not st.session_state.site_masks_computed:
        st.info("Configure constraints in the sidebar and click **Apply Constraints** to see results.")
        return
    
    # Retrieve computed data
    masks_dict = st.session_state.site_masks_dict
    combined_mask = st.session_state.site_combined_mask
    terrain_data = st.session_state.site_terrain_data
    
    # =========================================================================
    # SECTION 1: INTERACTIVE MAP VISUALIZATION
    # =========================================================================
    st.header("Interactive Map Visualization")
    
    # Display options (moved from sidebar)
    with st.expander("Display Options", expanded=False):
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            bg_style = st.selectbox(
                "Background style",
                ["terrain", "satellite", "streets"],
                index=0,
                key='site_bg_style'
            )
        with opt_col2:
            excluded_alpha = st.slider("Excluded opacity", 0.2, 1.0, 0.6, 0.1, key='site_excluded_alpha')
        # Note: admissible_alpha not needed since valid areas are transparent
        admissible_alpha = 0.0  # Not used but kept for function signature compatibility
    
    # View mode selector
    view_mode = st.radio(
        "View Mode",
        ["Masks Overview", "Single Mask Focus"],
        horizontal=True,
        index=0,
        key="view_mode",
        help="Choose how to visualize the constraints"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # VIEW MODE: MASKS OVERVIEW (with combined at the end)
    # -------------------------------------------------------------------------
    if view_mode == "Masks Overview":
        st.subheader("All Constraint Maps")
        
        if masks_dict:
            # Mask visibility toggles in an expander
            with st.expander("Toggle Mask Visibility", expanded=True):
                st.caption("Select which constraints to display in the grid")
                
                # Create columns for toggle checkboxes (4 per row for better layout)
                mask_names = list(masks_dict.keys())
                n_toggle_cols = min(4, len(mask_names) + 1)  # +1 for combined
                toggle_cols = st.columns(n_toggle_cols)
                
                visible_masks = {}
                for idx, name in enumerate(mask_names):
                    col_idx = idx % n_toggle_cols
                    with toggle_cols[col_idx]:
                        visible_masks[name] = st.checkbox(
                            name, 
                            value=True, 
                            key=f"vis_{name.replace(' ', '_')}"
                        )
                # Add combined checkbox
                with toggle_cols[len(mask_names) % n_toggle_cols]:
                    show_combined = st.checkbox("Combined Result", value=True, key="vis_combined")
            
            # Filter to only visible masks
            masks_to_show = {k: v for k, v in masks_dict.items() if visible_masks.get(k, True)}
            
            if masks_to_show or show_combined:
                # Display in a grid (2 per row)
                mask_names_visible = list(masks_to_show.keys())
                # Add combined at the end if enabled
                if show_combined:
                    mask_names_visible.append("Combined Result")
                
                n_masks = len(mask_names_visible)
                n_cols = 2
                
                for i in range(0, n_masks, n_cols):
                    cols = st.columns(n_cols)
                    for j in range(n_cols):
                        idx = i + j
                        if idx < n_masks:
                            name = mask_names_visible[idx]
                            # Get the appropriate mask
                            if name == "Combined Result":
                                mask = combined_mask
                            else:
                                mask = masks_to_show[name]
                            
                            with cols[j]:
                                fig = plot_single_mask(
                                    mask,
                                    terrain_data['lats'],
                                    terrain_data['lons'],
                                    terrain_data['Z_raw'],
                                    name,
                                    radar_lat=None,
                                    radar_lon=None,
                                    bg_style=bg_style,
                                    excluded_alpha=excluded_alpha,
                                    figsize=(6, 5)
                                )
                                st.pyplot(fig, use_container_width=True)
                                plt.close(fig)
            else:
                st.info("No masks selected. Use the toggles above to show constraint maps.")
        else:
            st.info("No individual constraints to display.")
    
    # -------------------------------------------------------------------------
    # VIEW MODE: SINGLE MASK FOCUS (always with comparison)
    # -------------------------------------------------------------------------
    elif view_mode == "Single Mask Focus":
        st.subheader("Detailed Constraint View")
        
        if masks_dict:
            mask_names = list(masks_dict.keys())
            
            # Dropdown to select which mask to view
            selected_mask_name = st.selectbox(
                "Select constraint to view",
                mask_names,
                key="focus_mask_select"
            )
            
            if selected_mask_name:
                selected_mask = masks_dict[selected_mask_name]
                
                # Calculate detailed statistics
                mask_admissible = np.sum(selected_mask)
                mask_excluded = selected_mask.size - mask_admissible
                mask_pct = mask_admissible / selected_mask.size * 100
                
                # Stats row
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    st.metric("Total Grid Points", f"{selected_mask.size:,}")
                with stat_cols[1]:
                    st.metric("Valid Points", f"{mask_admissible:,}")
                with stat_cols[2]:
                    st.metric("Excluded Points", f"{mask_excluded:,}")
                with stat_cols[3]:
                    st.metric("Valid Percentage", f"{mask_pct:.2f}%")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Side-by-side comparison with combined mask (always shown)
                comp_cols = st.columns(2)
                
                with comp_cols[0]:
                    st.markdown(f"**{selected_mask_name}** (this constraint alone)")
                    fig_single = plot_combined_mask(
                        selected_mask,
                        terrain_data['lats'],
                        terrain_data['lons'],
                        terrain_data['Z_raw'],
                        radar_lat=None,
                        radar_lon=None,
                        bg_style=bg_style,
                        admissible_alpha=admissible_alpha,
                        excluded_alpha=excluded_alpha,
                        figsize=(8, 6)
                    )
                    fig_single.axes[0].set_title(selected_mask_name, fontsize=12, fontweight='600', color=ACCENT_PRIMARY)
                    st.pyplot(fig_single, use_container_width=True)
                    plt.close(fig_single)
                
                with comp_cols[1]:
                    st.markdown("**Combined Result** (all constraints)")
                    fig_combined = plot_combined_mask(
                        combined_mask,
                        terrain_data['lats'],
                        terrain_data['lons'],
                        terrain_data['Z_raw'],
                        radar_lat=None,
                        radar_lon=None,
                        bg_style=bg_style,
                        admissible_alpha=admissible_alpha,
                        excluded_alpha=excluded_alpha,
                        figsize=(8, 6)
                    )
                    fig_combined.axes[0].set_title("All Constraints Combined", fontsize=12, fontweight='600', color=ACCENT_PRIMARY)
                    st.pyplot(fig_combined, use_container_width=True)
                    plt.close(fig_combined)
                
                # Contribution analysis
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='background-color: {BG_CARD}; padding: 15px; border-radius: 8px; border: 1px solid {BORDER_COLOR};'>
                        <h5 style='margin-top: 0; color: {TEXT_PRIMARY};'>Constraint Contribution Analysis</h5>
                        <p style='color: {TEXT_SECONDARY}; margin-bottom: 0;'>
                            <strong>{selected_mask_name}</strong> excludes <strong>{mask_excluded:,}</strong> points on its own.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No constraints available. Apply constraints first.")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 2: CONSTRAINT IMPACT SUMMARY TABLE
    # =========================================================================
    st.header("Constraint Impact Summary")
    
    # Build statistics table
    total_points = combined_mask.size
    stats_data = []
    
    for name, mask in masks_dict.items():
        # Determine type
        if 'Proximity' in name or 'km Radius' in name:
            ctype = 'Inclusion'
        else:
            ctype = 'Exclusion/Requirement'
        
        # Points for this constraint alone
        mask_valid = np.sum(mask)
        mask_excluded = total_points - mask_valid
        mask_pct = mask_valid / total_points * 100
        
        stats_data.append({
            'Constraint': name,
            'Type': ctype,
            'Valid Points': f'{mask_valid:,}',
            'Excluded Points': f'{mask_excluded:,}',
            'Valid Percentage': f'{mask_pct:.2f}%'
        })
    
    st.dataframe(stats_data, use_container_width=True, hide_index=True)
    
    # Final summary metrics
    final_admissible = np.sum(combined_mask)
    final_pct = final_admissible / total_points * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Grid Points", f"{total_points:,}")
    with col2:
        st.metric("Valid Sites", f"{final_admissible:,}")
    with col3:
        st.metric("Excluded Sites", f"{total_points - final_admissible:,}")
    with col4:
        st.metric("Valid Percentage", f"{final_pct:.2f}%")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 3: EXPORT
    # =========================================================================
    st.header("Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("NPZ Data")
        st.caption("All masks and coordinates")
        st.markdown("""
        - Combined mask array
        - Individual constraint masks
        - Valid site coordinates
        """)
        npz_data = export_results_npz(masks_dict, combined_mask, terrain_data)
        st.download_button(
            "Download NPZ",
            npz_data,
            "site_selection_results.npz",
            "application/octet-stream",
            use_container_width=True,
            key="npz_download"
        )
    
    with col2:
        st.subheader("Statistics CSV")
        st.caption("Constraint analysis table")
        st.markdown("""
        - Constraint impact statistics
        - Points excluded per constraint
        - Cumulative effect analysis
        """)
        csv_stats = export_results_csv(masks_dict, combined_mask, terrain_data)
        st.download_button(
            "Download Statistics CSV",
            csv_stats,
            "constraint_statistics.csv",
            "text/csv",
            use_container_width=True,
            key="stats_csv_download"
        )
    
    with col3:
        st.subheader("Candidates CSV")
        st.caption("Valid site coordinates")
        st.markdown("""
        - Latitude/Longitude
        - Elevation data
        - Up to 10,000 points
        """)
        csv_candidates = export_candidates_csv(combined_mask, terrain_data)
        st.download_button(
            "Download Candidates CSV",
            csv_candidates,
            "valid_sites.csv",
            "text/csv",
            use_container_width=True,
            key="candidates_csv_download"
        )
    
    # KMZ Export (if available)
    if HAS_KMZ_EXPORT:
        st.markdown("---")
        st.subheader("KMZ Export (Google Earth)")
        st.caption("Visualize constraints and candidate points in Google Earth")
        
        kmz_col1, kmz_col2 = st.columns([2, 1])
        with kmz_col1:
            st.markdown("""
            Export to KMZ file for visualization in Google Earth:
            - Each constraint as a separate toggleable layer
            - Combined result layer
            - **All candidate points** (toggleable folder)
            - Reference markers (Airport)
            """)
            
            # Option for candidate point density
            max_candidates = st.slider(
                "Max candidate points to export",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=1000,
                key="kmz_max_candidates",
                help="Limit the number of candidate points to avoid large files"
            )
        
        with kmz_col2:
            if st.button("Generate KMZ", use_container_width=True, key="gen_kmz"):
                with st.spinner("Generating KMZ with candidate points..."):
                    try:
                        import xml.etree.ElementTree as ET
                        
                        # Downsample masks for KMZ (to reduce file size)
                        step = max(1, min(terrain_data['lats'].shape[0] // 100, terrain_data['lons'].shape[0] // 100))
                        
                        masks_for_kmz = {k: v[::step, ::step] for k, v in masks_dict.items()}
                        masks_for_kmz['COMBINED - Valid Sites'] = combined_mask[::step, ::step]
                        
                        # Export base KMZ to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.kmz') as tmp:
                            tmp_path = tmp.name
                        
                        export_masks_to_kmz(
                            masks_for_kmz,
                            terrain_data['lats'][::step],
                            terrain_data['lons'][::step],
                            tmp_path,
                            nice_lat=AIRPORT_LAT,
                            nice_lon=AIRPORT_LON
                        )
                        
                        # Read the KMZ, extract KML, add candidate points, repackage
                        with zipfile.ZipFile(tmp_path, 'r') as kmz_in:
                            kml_content = kmz_in.read('doc.kml')
                        
                        # Parse KML and add candidate points folder
                        kml_root = ET.fromstring(kml_content)
                        document = kml_root.find('.//{http://www.opengis.net/kml/2.2}Document')
                        
                        if document is not None:
                            # Extract candidate points first to get count
                            lats = terrain_data['lats']
                            lons = terrain_data['lons']
                            Z_raw = terrain_data['Z_raw']
                            
                            i_rows, j_cols = np.where(combined_mask)
                            n_candidates = len(i_rows)
                            
                            # Downsample if too many points
                            if n_candidates > max_candidates:
                                step_pts = n_candidates // max_candidates
                                i_rows = i_rows[::step_pts]
                                j_cols = j_cols[::step_pts]
                            
                            n_exported = len(i_rows)
                            
                            # Create candidate points folder with description including count
                            candidates_folder = ET.SubElement(document, "Folder")
                            ET.SubElement(candidates_folder, "name").text = "Candidate Points"
                            desc_text = f"Valid radar site candidates: {n_exported:,} points"
                            if n_candidates > max_candidates:
                                desc_text += f" (downsampled from {n_candidates:,})"
                            ET.SubElement(candidates_folder, "description").text = desc_text
                            
                            # Create style for candidate points
                            candidate_style = ET.SubElement(document, "Style", id="candidate_style")
                            icon_style = ET.SubElement(candidate_style, "IconStyle")
                            ET.SubElement(icon_style, "color").text = "ff00ff00"  # Green (AABBGGRR)
                            ET.SubElement(icon_style, "scale").text = "0.5"
                            icon = ET.SubElement(icon_style, "Icon")
                            ET.SubElement(icon, "href").text = "http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png"
                            
                            # Add placemarks for each candidate
                            for idx, (i, j) in enumerate(zip(i_rows, j_cols)):
                                lat = lats[i]
                                lon = lons[j]
                                elev = Z_raw[i, j]
                                
                                placemark = ET.SubElement(candidates_folder, "Placemark")
                                ET.SubElement(placemark, "name").text = f"Site {idx+1}"
                                ET.SubElement(placemark, "description").text = f"Lat: {lat:.6f}°N\nLon: {lon:.6f}°E\nElevation: {elev:.1f}m"
                                ET.SubElement(placemark, "styleUrl").text = "#candidate_style"
                                
                                point = ET.SubElement(placemark, "Point")
                                coords = ET.SubElement(point, "coordinates")
                                coords.text = f"{lon},{lat},{elev}"
                        
                        # Repackage as KMZ
                        kml_str = ET.tostring(kml_root, encoding='utf-8', xml_declaration=True)
                        
                        kmz_buffer = BytesIO()
                        with zipfile.ZipFile(kmz_buffer, 'w', zipfile.ZIP_DEFLATED) as kmz_out:
                            kmz_out.writestr('doc.kml', kml_str)
                        
                        kmz_data = kmz_buffer.getvalue()
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                        st.download_button(
                            "Download KMZ",
                            kmz_data,
                            "site_selection.kmz",
                            "application/vnd.google-earth.kmz",
                            use_container_width=True,
                            key="kmz_download"
                        )
                        
                        st.success("KMZ generated with candidate points")
                    except Exception as e:
                        st.error(f"KMZ export failed: {e}")
    
    # Footer with summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='background-color: {BG_CARD}; padding: 20px; border-radius: 12px; border: 1px solid {BORDER_COLOR}; text-align: center;'>
            <h4 style='margin-top: 0; color: {ACCENT_PRIMARY};'>Analysis Summary</h4>
            <p style='color: {TEXT_SECONDARY}; margin-bottom: 5px;'>
                {len(masks_dict)} constraints applied | {final_admissible:,} valid sites identified ({final_pct:.2f}% of territory)
            </p>
            <p style='color: {TEXT_SECONDARY}; margin-bottom: 0;'>
                Reference: {AIRPORT_NAME} ({AIRPORT_LAT}°N, {AIRPORT_LON}°E)
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
