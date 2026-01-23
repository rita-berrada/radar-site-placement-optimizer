"""
Radar Coverage Analysis Software

A comprehensive application for radar coverage analysis that:
1. Loads terrain data from NPZ files
2. Visualizes terrain in 2D and 3D
3. Computes coverage on Flight Levels (FL)
4. Outputs coverage maps with various backgrounds
5. Exports results to KMZ format

Author: Radar Analysis Team

NOTE: This app uses the same Earth curvature-corrected algorithms as main_coverage.py
to ensure consistent coverage computation results.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as path_effects
from io import BytesIO
from typing import Dict, List, Optional, Tuple
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Numba import
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Contextily import
try:
    import contextily as cx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

# Import Earth curvature-corrected modules (same as main_coverage.py)
try:
    from geo_utils_earth_curvature import load_and_convert_to_enu, REF_LAT, REF_LON, EARTH_RADIUS_M as GEO_EARTH_RADIUS_M
    from LOS_numba_enu import coverage_map_numba_xy, latlon_to_xy_m as los_latlon_to_xy_m, fl_to_m as los_fl_to_m
    HAS_CURVATURE_MODULES = True
except ImportError:
    HAS_CURVATURE_MODULES = False

# ============================================================================
# MEDIUM DARK BLUE-GREY PALETTE
# ============================================================================
BG_APP = "#1A2332"          # Medium dark blue-grey (improved readability)
BG_PANEL = "#243447"        # Sidebar/panel background
BG_CARD = "#2D4156"         # Card/container background
BORDER_COLOR = "#3D5A73"    # Visible borders

TEXT_PRIMARY = "#F0F4F8"    # High contrast white
TEXT_SECONDARY = "#B8C9DB"  # Readable secondary text

ACCENT_PRIMARY = "#4A90D9"  # Professional blue
ACCENT_HOVER = "#6BA3E3"    # Lighter blue hover

# Coverage Data Colors (Professional scientific)
COVERAGE_VISIBLE = "#4CAF50"  # Standard green for visible
BLOCKED_AREA = "#EF5350"      # Standard red for blocked


# Metric color
METRIC_VALUE = "#6BA3E3"    # Professional light blue for metrics (matches ACCENT_HOVER)

# ============================================================================
# CONSTANTS
# ============================================================================
DEFAULT_REF_LAT = 43.6584
DEFAULT_REF_LON = 7.2159
EARTH_RADIUS_M = 6371000.0

AIRPORT_LAT = 43.6584
AIRPORT_LON = 7.2159
AIRPORT_NAME = "Nice Airport (LFMN)"

FLIGHT_LEVELS = [5, 10, 20, 50, 100, 200, 300, 400]

# Airport range rings (km) for specific flight levels
AIRPORT_RANGE_RINGS = {
    5: 2.9,    # FL5 → 2.9 km
    10: 5.8,   # FL10 → 5.8 km
    20: 11.6,  # FL20 → 11.6 km
    50: 29.0,  # FL50 → 29 km
}

def draw_airport_range_ring(ax, fl, linewidth=1.5, linestyle='--', color='white', alpha=0.9):
    """
    Draw a range ring centered on the airport for specific flight levels.
    Only draws for FL5, FL10, FL20, FL50 with predefined radii.
    
    Args:
        ax: matplotlib axis
        fl: flight level (int)
        linewidth: line width for the circle
        linestyle: line style ('--' for dashed, '-' for solid)
        color: line color
        alpha: transparency
    """
    if fl not in AIRPORT_RANGE_RINGS:
        return None
    
    radius_km = AIRPORT_RANGE_RINGS[fl]
    
    # Convert km to degrees (approximate at this latitude)
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 * cos(lat) km
    lat_rad = np.radians(AIRPORT_LAT)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(lat_rad)
    
    # Radius in degrees
    radius_lat = radius_km / km_per_deg_lat
    radius_lon = radius_km / km_per_deg_lon
    
    # Draw circle using parametric plot (cleaner than Ellipse patch)
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_lon = AIRPORT_LON + radius_lon * np.cos(theta)
    circle_lat = AIRPORT_LAT + radius_lat * np.sin(theta)
    
    line, = ax.plot(circle_lon, circle_lat, linestyle=linestyle, linewidth=linewidth, 
                    color=color, alpha=alpha, zorder=8)
    
    return line

# ============================================================================
# CUSTOM CSS - Minimal Essential Styling
# ============================================================================
def apply_theme():
    """Apply professional theme with modern UI elements.
    
    Theme colors are configured in .streamlit/config.toml
    """
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: {TEXT_PRIMARY};
        }}
        
        /* Main title styling */
        .main-title {{
            text-align: center;
            font-size: 3.5rem;
            font-weight: 800;
            color: {TEXT_PRIMARY};
            margin-bottom: 0.1rem;
            letter-spacing: -1px;
        }}
        
        .main-subtitle {{
            text-align: center;
            font-size: 1rem;
            color: {TEXT_SECONDARY};
            margin-bottom: 2.5rem;
            font-weight: 500;
            letter-spacing: 2px;
            text-transform: uppercase;
            opacity: 0.8;
        }}

        /* Section headers - aligned to the side */
        h1, h2 {{
            text-align: left !important;
            color: {ACCENT_PRIMARY};
            margin-top: 2.5rem !important;
            margin-bottom: 1.5rem !important;
            font-weight: 700;
            padding-left: 0;
        }}

        h3 {{
            color: {TEXT_SECONDARY};
            font-weight: 600;
            text-align: left !important;
        }}
        
        /* KPI card styling with glassmorphism effects */
        div[data-testid="stMetric"] {{
            background: {BG_CARD};
            border-radius: 12px;
            padding: 15px 20px;
            border: 1px solid {BORDER_COLOR};
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 15px rgba(0, 0, 0, 0.2);
            border-color: {ACCENT_PRIMARY};
        }}

        /* Section containers */
        .stButton button {{
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        .stButton button:hover {{
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
            transform: translateY(-1px);
        }}

        /* Expander styling */
        .streamlit-expanderHeader {{
            background-color: {BG_PANEL};
            border-radius: 8px;
            border: 1px solid {BORDER_COLOR};
        }}

        /* Scrollbar styling */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: {BG_APP};
        }}
        ::-webkit-scrollbar-thumb {{
            background: {BORDER_COLOR};
            border-radius: 10px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {ACCENT_PRIMARY};
        }}

        /* Centered button container */
        .centered-compute {{
            display: flex;
            justify-content: center;
            margin: 2rem 0;
        }}
    </style>
    """, unsafe_allow_html=True)


def render_title():
    """Render centered application title with modern styling."""
    st.markdown(f"""
        <div style="margin-bottom: 2.5rem;">
            <div class="main-title">
                Radar Coverage Analysis
            </div>
            <div class="main-subtitle">
                Advanced Terrain Modeling & Line-of-Sight Computing
            </div>
        </div>
    """, unsafe_allow_html=True)


# ============================================================================
# TERRAIN LOADING
# ============================================================================

def load_terrain_npz(npz_file) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw terrain data from NPZ file (without curvature correction)."""
    data = np.load(npz_file)
    lats = data['lat'].astype(float)
    lons = data['lon'].astype(float)
    Z = data['ter'].astype(float)
    if Z.shape != (len(lats), len(lons)):
        raise ValueError(f"Shape mismatch: Z{Z.shape} vs ({len(lats)}, {len(lons)})")
    return lats, lons, Z


def load_terrain_with_curvature(npz_file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load terrain data with Earth curvature correction applied to Z.
    This is the same method used in main_coverage.py for consistency.
    
    Returns:
        X_m: East-West distance in meters (1D array)
        Y_m: North-South distance in meters (1D array)
        Z_corrected: Terrain elevation corrected for Earth curvature (2D array)
        lats: Original latitudes (1D array)
        lons: Original longitudes (1D array)
    """
    if HAS_CURVATURE_MODULES:
        # Use the same loading function as main_coverage.py
        return load_and_convert_to_enu(npz_file_path)
    else:
        # Fallback: Apply curvature correction manually
        data = np.load(npz_file_path)
        lats = data['lat'].astype(float)
        lons = data['lon'].astype(float)
        Z_terrain = data['ter'].astype(float)
        
        # Use Nice Airport as reference (same as geo_utils_earth_curvature)
        ref_lat = 43.6584
        ref_lon = 7.2159
        
        lat_ref_rad = np.radians(ref_lat)
        meters_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
        meters_per_deg_lon = (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(lat_ref_rad)
        
        Y_m = (lats - ref_lat) * meters_per_deg_lat
        X_m = (lons - ref_lon) * meters_per_deg_lon
        
        # Apply Earth curvature correction to terrain
        X_grid, Y_grid = np.meshgrid(X_m, Y_m)
        dist_sq = X_grid**2 + Y_grid**2
        curvature_drop = dist_sq / (2.0 * EARTH_RADIUS_M)
        Z_corrected = Z_terrain - curvature_drop
        
        return X_m, Y_m, Z_corrected, lats, lons


def convert_to_enu(lats, lons, ref_lat, ref_lon):
    """Convert lat/lon to ENU meters (legacy function for visualization)."""
    lat_ref_rad = np.radians(ref_lat)
    meters_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
    meters_per_deg_lon = (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(lat_ref_rad)
    Y_m = (lats - ref_lat) * meters_per_deg_lat
    X_m = (lons - ref_lon) * meters_per_deg_lon
    return X_m, Y_m


def normalize_xy_grid(X_m, Y_m, Z):
    """Ensure X_m and Y_m are strictly increasing, reorder Z accordingly."""
    X_m = np.asarray(X_m)
    Y_m = np.asarray(Y_m)
    Z = np.asarray(Z)
    if Y_m[0] > Y_m[-1]:
        Y_m = Y_m[::-1].copy()
        Z = Z[::-1, :].copy()
    if X_m[0] > X_m[-1]:
        X_m = X_m[::-1].copy()
        Z = Z[:, ::-1].copy()
    return np.ascontiguousarray(X_m), np.ascontiguousarray(Y_m), np.ascontiguousarray(Z)


def normalize_all(X_m, Y_m, Z, lats, lons):
    """
    Ensure X_m and Y_m are strictly increasing (monotonic).
    Applies the necessary flips to Z, lats, and lons to maintain alignment.
    Returns contiguous arrays for Numba performance.
    (Same as main_coverage.py)
    """
    X_m = np.asarray(X_m)
    Y_m = np.asarray(Y_m)
    Z = np.asarray(Z)
    lats = np.asarray(lats)
    lons = np.asarray(lons)

    # Check Y axis (North-South)
    if Y_m[0] > Y_m[-1]:
        Y_m = Y_m[::-1].copy()
        lats = lats[::-1].copy()
        Z = Z[::-1, :].copy()

    # Check X axis (East-West)
    if X_m[0] > X_m[-1]:
        X_m = X_m[::-1].copy()
        lons = lons[::-1].copy()
        Z = Z[:, ::-1].copy()
    
    return (np.ascontiguousarray(X_m), 
            np.ascontiguousarray(Y_m), 
            np.ascontiguousarray(Z), 
            np.ascontiguousarray(lats), 
            np.ascontiguousarray(lons))


# ============================================================================
# COVERAGE COMPUTATION (WITH EARTH CURVATURE CORRECTION)
# ============================================================================

def fl_to_m(FL):
    """Convert Flight Level to meters."""
    if HAS_CURVATURE_MODULES:
        return los_fl_to_m(FL)
    return FL * 100.0 * 0.3048


def latlon_to_xy_m(lat, lon, ref_lat=None, ref_lon=None):
    """
    Convert lat/lon to ENU meters.
    When curvature modules are available, uses the fixed reference point
    (Nice Airport) for consistency with main_coverage.py.
    """
    if HAS_CURVATURE_MODULES:
        # Use the same conversion as LOS_numba_enu (fixed reference point)
        return los_latlon_to_xy_m(lat, lon)
    else:
        # Fallback: use provided ref or defaults
        if ref_lat is None:
            ref_lat = 43.6584
        if ref_lon is None:
            ref_lon = 7.2159
        lat_ref_rad = np.radians(ref_lat)
        meters_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
        meters_per_deg_lon = (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(lat_ref_rad)
        y = (float(lat) - float(ref_lat)) * meters_per_deg_lat
        x = (float(lon) - float(ref_lon)) * meters_per_deg_lon
        return x, y


# Legacy Numba functions (fallback when curvature modules not available)
if HAS_NUMBA:
    @njit
    def z_bilinear(x, y, x0, y0, dx, dy, Z):
        nrow, ncol = Z.shape
        x_max = x0 + dx * (ncol - 1)
        y_max = y0 + dy * (nrow - 1)
        if x < x0 or x > x_max or y < y0 or y > y_max:
            return np.nan
        fj = (x - x0) / dx
        fi = (y - y0) / dy
        j0 = int(fj)
        i0 = int(fi)
        if j0 < 0: j0 = 0
        if i0 < 0: i0 = 0
        if j0 > ncol - 2: j0 = ncol - 2
        if i0 > nrow - 2: i0 = nrow - 2
        u = fj - j0
        t = fi - i0
        z0 = (1.0 - u) * Z[i0, j0] + u * Z[i0, j0 + 1]
        z1 = (1.0 - u) * Z[i0 + 1, j0] + u * Z[i0 + 1, j0 + 1]
        return (1.0 - t) * z0 + t * z1

    @njit
    def los_check_with_curvature(radar_x, radar_y, radar_h, target_x, target_y, target_alt_msl, 
                                  x0, y0, dx, dy, Z_corrected, n_samples, margin, earth_radius):
        """
        LOS check with Earth curvature correction on target altitude.
        Z_corrected: terrain already has curvature drop applied.
        target_alt_msl: must also have curvature drop applied.
        """
        z_r = z_bilinear(radar_x, radar_y, x0, y0, dx, dy, Z_corrected)
        if np.isnan(z_r):
            return False
        z_radar = z_r + radar_h
        
        # Apply curvature drop to target altitude (same as LOS_numba_enu.py)
        dist_sq_target = target_x**2 + target_y**2
        drop_target = dist_sq_target / (2.0 * earth_radius)
        z_target_enu = target_alt_msl - drop_target
        
        for k in range(1, n_samples):
            s = k / n_samples
            x = radar_x + s * (target_x - radar_x)
            y = radar_y + s * (target_y - radar_y)
            z_g = z_bilinear(x, y, x0, y0, dx, dy, Z_corrected)
            if np.isnan(z_g):
                return False
            z_line = z_radar + s * (z_target_enu - z_radar)
            if z_g + margin >= z_line:
                return False
        return True

    @njit(parallel=True)
    def compute_coverage_with_curvature(radar_x, radar_y, radar_h, target_alt_msl, X_m, Y_m, 
                                         x0, y0, dx, dy, Z_corrected, n_samples, margin, earth_radius):
        """
        Coverage computation with Earth curvature correction.
        This matches the physics in LOS_numba_enu.py / main_coverage.py.
        """
        N, M = Y_m.size, X_m.size
        out = np.zeros((N, M), dtype=np.bool_)
        for i in prange(N):
            for j in range(M):
                out[i, j] = los_check_with_curvature(
                    radar_x, radar_y, radar_h, X_m[j], Y_m[i], target_alt_msl,
                    x0, y0, dx, dy, Z_corrected, n_samples, margin, earth_radius
                )
        return out
else:
    def z_bilinear(x, y, x0, y0, dx, dy, Z):
        nrow, ncol = Z.shape
        x_max = x0 + dx * (ncol - 1)
        y_max = y0 + dy * (nrow - 1)
        if x < x0 or x > x_max or y < y0 or y > y_max:
            return np.nan
        fj = (x - x0) / dx
        fi = (y - y0) / dy
        j0 = max(0, min(int(fj), ncol - 2))
        i0 = max(0, min(int(fi), nrow - 2))
        u = fj - j0
        t = fi - i0
        z0 = (1.0 - u) * Z[i0, j0] + u * Z[i0, j0 + 1]
        z1 = (1.0 - u) * Z[i0 + 1, j0] + u * Z[i0 + 1, j0 + 1]
        return (1.0 - t) * z0 + t * z1

    def compute_coverage_with_curvature(radar_x, radar_y, radar_h, target_alt_msl, X_m, Y_m,
                                         x0, y0, dx, dy, Z_corrected, n_samples, margin, earth_radius):
        """Fallback non-Numba version with curvature correction."""
        N, M = Y_m.size, X_m.size
        out = np.zeros((N, M), dtype=np.bool_)
        z_r = z_bilinear(radar_x, radar_y, x0, y0, dx, dy, Z_corrected)
        if np.isnan(z_r):
            return out
        z_radar = z_r + radar_h
        
        for i in range(N):
            for j in range(M):
                target_x, target_y = X_m[j], Y_m[i]
                
                # Apply curvature drop to target altitude
                dist_sq_target = target_x**2 + target_y**2
                drop_target = dist_sq_target / (2.0 * earth_radius)
                z_target_enu = target_alt_msl - drop_target
                
                visible = True
                for k in range(1, n_samples):
                    s = k / n_samples
                    x = radar_x + s * (target_x - radar_x)
                    y = radar_y + s * (target_y - radar_y)
                    z_g = z_bilinear(x, y, x0, y0, dx, dy, Z_corrected)
                    if np.isnan(z_g):
                        visible = False
                        break
                    z_line = z_radar + s * (z_target_enu - z_radar)
                    if z_g + margin >= z_line:
                        visible = False
                        break
                out[i, j] = visible
        return out


def compute_fl_coverage_curvature(radar_lat, radar_lon, radar_h, fl, X_m, Y_m, Z_corrected, n_samples=400, margin=0.0):
    """
    Compute coverage for a single FL using Earth curvature-corrected algorithms.
    This matches main_coverage.py exactly.
    
    Args:
        radar_lat, radar_lon: Radar position in degrees
        radar_h: Radar height above ground level in meters
        fl: Flight level (e.g., 50 = FL050 = 5000 ft)
        X_m, Y_m: Grid axes in ENU meters (already curvature-corrected)
        Z_corrected: Terrain elevation with curvature drop applied
        n_samples: Number of samples along LOS ray
        margin: Safety margin in meters
    """
    if HAS_CURVATURE_MODULES:
        # Use the exact same function as main_coverage.py
        radar_x, radar_y = los_latlon_to_xy_m(radar_lat, radar_lon)
        x0 = float(X_m[0])
        y0 = float(Y_m[0])
        dx = float(X_m[1] - X_m[0]) if len(X_m) > 1 else 1.0
        dy = float(Y_m[1] - Y_m[0]) if len(Y_m) > 1 else 1.0
        target_alt_msl = float(los_fl_to_m(fl))
        
        return coverage_map_numba_xy(
            float(radar_x), float(radar_y), float(radar_h),
            target_alt_msl,
            X_m, Y_m,
            x0, y0, dx, dy, Z_corrected,
            int(n_samples), float(margin)
        )
    else:
        # Fallback using local curvature-corrected functions
        radar_x, radar_y = latlon_to_xy_m(radar_lat, radar_lon)
        x0 = float(X_m[0])
        y0 = float(Y_m[0])
        dx = float(X_m[1] - X_m[0]) if len(X_m) > 1 else 1.0
        dy = float(Y_m[1] - Y_m[0]) if len(Y_m) > 1 else 1.0
        target_alt_msl = float(fl_to_m(fl))
        
        return compute_coverage_with_curvature(
            float(radar_x), float(radar_y), float(radar_h), target_alt_msl,
            X_m, Y_m, x0, y0, dx, dy, Z_corrected, int(n_samples), float(margin),
            EARTH_RADIUS_M
        )


def compute_all_fl_curvature(radar_lat, radar_lon, radar_h, fls, X_m, Y_m, Z_corrected, n_samples=400, margin=0.0, callback=None):
    """
    Compute coverage for all flight levels using Earth curvature correction.
    This is the main function to use for consistency with main_coverage.py.
    """
    maps = {}
    for i, fl in enumerate(fls):
        maps[fl] = compute_fl_coverage_curvature(radar_lat, radar_lon, radar_h, fl, X_m, Y_m, Z_corrected, n_samples, margin)
        if callback:
            callback((i + 1) / len(fls))
    return maps


# Keep legacy functions for backward compatibility (but they should not be used)
def compute_fl_coverage(radar_lat, radar_lon, radar_h, fl, X_m, Y_m, Z, ref_lat, ref_lon, n_samples=400, margin=0.0):
    """Legacy function - use compute_fl_coverage_curvature instead for accurate results."""
    return compute_fl_coverage_curvature(radar_lat, radar_lon, radar_h, fl, X_m, Y_m, Z, n_samples, margin)


def compute_all_fl(radar_lat, radar_lon, radar_h, fls, X_m, Y_m, Z, ref_lat, ref_lon, n_samples=400, margin=0.0, callback=None):
    """Legacy function - use compute_all_fl_curvature instead for accurate results."""
    return compute_all_fl_curvature(radar_lat, radar_lon, radar_h, fls, X_m, Y_m, Z, n_samples, margin, callback)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_terrain_2d(lats, lons, Z, radar_lat=None, radar_lon=None, figsize=(7, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG_PANEL)
    ax.set_facecolor(BG_PANEL)
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    Z_plot = np.ma.masked_where(Z < -100, Z)
    
    im = ax.contourf(lon_grid, lat_grid, Z_plot, levels=50, cmap='terrain')
    cbar = plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)
    cbar.ax.yaxis.label.set_color(TEXT_SECONDARY)
    cbar.ax.tick_params(colors=TEXT_SECONDARY)
    
    # Radar marker - Blue
    if radar_lat is not None and radar_lon is not None:
        ax.plot(radar_lon, radar_lat, '*', color=ACCENT_PRIMARY, markersize=10, label='Radar',
                markeredgecolor='white', markeredgewidth=1)
    
    # Airport marker - White
    ax.plot(AIRPORT_LON, AIRPORT_LAT, '^', color=TEXT_PRIMARY, markersize=8, label='Airport',
            markeredgecolor='black', markeredgewidth=0.8)
    
    ax.legend(loc='upper right', facecolor=BG_PANEL, edgecolor=BORDER_COLOR, labelcolor=TEXT_PRIMARY, fontsize=9)
    ax.set_xlabel('Longitude (°)', color=TEXT_SECONDARY)
    ax.set_ylabel('Latitude (°)', color=TEXT_SECONDARY)
    ax.set_title('2D Terrain Elevation', fontsize=12, fontweight='600', color=ACCENT_PRIMARY)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.grid(True, color=BORDER_COLOR, alpha=0.3, linestyle='-', linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    
    plt.tight_layout()
    return fig


def plot_terrain_3d(lats, lons, Z, radar_lat=None, radar_lon=None, elev=30, azim=-60, figsize=(7, 6)):
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(BG_PANEL)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BG_PANEL)
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    step = max(1, len(lats) // 200, len(lons) // 200)
    Z_plot = np.ma.masked_where(Z < -100, Z)
    
    surf = ax.plot_surface(lon_grid[::step, ::step], lat_grid[::step, ::step], Z_plot[::step, ::step],
                           cmap='terrain', alpha=0.9, linewidth=0, antialiased=True)
    
    # Radar - Blue
    if radar_lat is not None and radar_lon is not None:
        lat_idx = np.argmin(np.abs(lats - radar_lat))
        lon_idx = np.argmin(np.abs(lons - radar_lon))
        radar_elev = max(0, Z[lat_idx, lon_idx]) + 100
        ax.scatter([radar_lon], [radar_lat], [radar_elev], c=ACCENT_PRIMARY, s=200, marker='*', 
                   label='Radar', edgecolors='white', linewidths=1.5, zorder=20)
    
    # Airport - White
    ap_lat_idx = np.argmin(np.abs(lats - AIRPORT_LAT))
    ap_lon_idx = np.argmin(np.abs(lons - AIRPORT_LON))
    ap_elev = max(0, Z[ap_lat_idx, ap_lon_idx]) + 50
    ax.scatter([AIRPORT_LON], [AIRPORT_LAT], [ap_elev], c=TEXT_PRIMARY, s=150, marker='^', 
               label='Airport', edgecolors='black', linewidths=1, zorder=20)
    
    ax.legend(loc='upper left', facecolor=BG_PANEL, edgecolor=BORDER_COLOR, labelcolor=TEXT_PRIMARY, fontsize=9)
    ax.set_xlabel('Lon (°)', color=TEXT_SECONDARY)
    ax.set_ylabel('Lat (°)', color=TEXT_SECONDARY)
    ax.set_zlabel('Elev (m)', color=TEXT_SECONDARY)
    ax.set_title('3D Terrain Surface', fontsize=12, fontweight='600', color=ACCENT_PRIMARY)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(BORDER_COLOR)
    ax.yaxis.pane.set_edgecolor(BORDER_COLOR)
    ax.zaxis.pane.set_edgecolor(BORDER_COLOR)
    # Grid lines for 3D plot
    ax.xaxis._axinfo['grid']['color'] = BORDER_COLOR
    ax.yaxis._axinfo['grid']['color'] = BORDER_COLOR
    ax.zaxis._axinfo['grid']['color'] = BORDER_COLOR
    ax.xaxis._axinfo['grid']['linewidth'] = 0.5
    ax.yaxis._axinfo['grid']['linewidth'] = 0.5
    ax.zaxis._axinfo['grid']['linewidth'] = 0.5
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.ax.tick_params(colors=TEXT_SECONDARY)
    
    plt.tight_layout()
    return fig


def hillshade(z, azimuth=315.0, altitude=45.0):
    dy, dx = np.gradient(z)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    shaded = np.sin(np.deg2rad(altitude)) * np.sin(slope) + np.cos(np.deg2rad(altitude)) * np.cos(slope) * np.cos(np.deg2rad(azimuth) - aspect)
    shaded = np.clip(shaded, -1.0, 1.0)
    mn, mx = np.nanmin(shaded), np.nanmax(shaded)
    return (shaded - mn) / (mx - mn + 1e-12)


def colored_relief(z):
    z_valid = z[np.isfinite(z)]
    z_min = np.nanmin(z_valid) if len(z_valid) > 0 else 0
    z_max = np.nanmax(z_valid) if len(z_valid) > 0 else 1
    zn = np.clip((z - z_min) / (z_max - z_min + 1e-12), 0, 1)
    rgb = cm.get_cmap('terrain')(zn)[..., :3]
    hs = hillshade(z)
    rgb = rgb * (0.75 + 0.25 * hs)[..., None]
    return np.clip(rgb, 0, 1)


def _remove_contextily_attribution(ax):
    """Best-effort removal of attribution text added by contextily (if present)."""
    for t in list(ax.texts):
        txt = (t.get_text() or "").lower()
        if any(k in txt for k in ["tiles", "esri", "openstreetmap", "carto", "i-cubed", "usgs", "aex", "getmapping"]):
            try:
                t.remove()
            except Exception:
                pass


def _add_basemap(ax, extent, provider="carto", add_labels=False):
    """
    Add basemap tiles using contextily on an EPSG:4326 axis.
    provider: "esri" | "carto"
    add_labels: if True and provider == "esri", overlays Esri labels on top of imagery
    """
    if not HAS_CONTEXTILY:
        return

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # Choose tile provider
    if provider == "esri":
        base_src = cx.providers.Esri.WorldImagery
    else:
        # "google-like" streets with labels
        base_src = cx.providers.CartoDB.Voyager  # or Positron for minimal look

    # Try attribution=False if supported by your contextily version
    try:
        cx.add_basemap(ax, crs="EPSG:4326", source=base_src, attribution=False)
    except TypeError:
        # Older contextily versions may not accept attribution=
        cx.add_basemap(ax, crs="EPSG:4326", source=base_src)
        _remove_contextily_attribution(ax)

    # Optional labels overlay (works great on satellite imagery)
    if add_labels and provider == "esri":
        try:
            try:
                cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.Esri.WorldReferenceOverlay, attribution=False)
            except TypeError:
                cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.Esri.WorldReferenceOverlay)
                _remove_contextily_attribution(ax)
        except Exception:
            # If overlay isn't available in your contextily providers, just ignore
            pass


def plot_coverage(
    cov, lats, lons, fl, terrain=None, radar_lat=None, radar_lon=None,
    bg="relief",
    basemap_provider="carto",   # <-- NEW ("esri" or "carto")
    basemap_labels=False,       # <-- NEW (useful for esri satellite + labels)
    show_blocked=True, green_alpha=0.5, red_alpha=0.3, figsize=(5, 4)
):
    """Small coverage map for grid."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG_PANEL)
    ax.set_facecolor(BG_PANEL)

    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    # Get terrain image based on background type
    if bg == "terrain" and terrain is not None:
        # Terrain colormap using imshow (no contour artifacts)
        Z_plot = np.ma.masked_where(terrain < -100, terrain)
        im = ax.imshow(Z_plot, extent=extent, aspect='auto', origin='lower', 
                       cmap='terrain', interpolation='bilinear', zorder=1)
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('Elevation (m)', fontsize=8, color=TEXT_SECONDARY)
        cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)

    elif bg == "relief" and terrain is not None:
        ax.imshow(
            colored_relief(terrain),
            aspect='auto', origin='lower', extent=extent,
            interpolation='bilinear'
        )

    elif bg == "basemap":
        try:
            _add_basemap(ax, extent, provider=basemap_provider, add_labels=basemap_labels)
        except Exception:
            # fallback to relief if basemap fails
            if terrain is not None:
                ax.imshow(colored_relief(terrain), aspect='auto', origin='lower', extent=extent, interpolation='bilinear')

    # Coverage overlay - use RGBA array to avoid transparency artifacts
    # Create RGBA overlay directly
    overlay = np.zeros((*cov.shape, 4), dtype=np.float32)
    
    # Get colors
    green_rgb = [int(COVERAGE_VISIBLE[i:i+2], 16)/255 for i in (1, 3, 5)]
    red_rgb = [int(BLOCKED_AREA[i:i+2], 16)/255 for i in (1, 3, 5)]
    
    # Set visible areas (cov=1) to green
    visible_mask = cov == 1
    overlay[visible_mask, 0] = green_rgb[0]
    overlay[visible_mask, 1] = green_rgb[1]
    overlay[visible_mask, 2] = green_rgb[2]
    overlay[visible_mask, 3] = green_alpha
    
    # Set blocked areas (cov=0) 
    if show_blocked:
        blocked_mask = cov == 0
        overlay[blocked_mask, 0] = red_rgb[0]
        overlay[blocked_mask, 1] = red_rgb[1]
        overlay[blocked_mask, 2] = red_rgb[2]
        overlay[blocked_mask, 3] = red_alpha
    # else: blocked areas stay transparent (alpha=0)

    ax.imshow(
        overlay,
        aspect='auto',
        origin='lower',
        extent=extent,
        interpolation='none',  # No interpolation to avoid artifacts
        zorder=2
    )

    # Markers
    if radar_lat is not None and radar_lon is not None:
        ax.plot(radar_lon, radar_lat, '*', color=ACCENT_PRIMARY, markersize=7, label='Radar',
                markeredgecolor='white', markeredgewidth=0.6, zorder=10)

    ax.plot(AIRPORT_LON, AIRPORT_LAT, '^', color=TEXT_PRIMARY, markersize=6, label='Airport',
            markeredgecolor='black', markeredgewidth=0.4, zorder=10)

    # Draw airport range ring for FL5, FL10, FL20, FL50
    if fl in AIRPORT_RANGE_RINGS:
        radius_km = AIRPORT_RANGE_RINGS[fl]
        lat_rad = np.radians(AIRPORT_LAT)
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(lat_rad)
        radius_lat = radius_km / km_per_deg_lat
        radius_lon = radius_km / km_per_deg_lon
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_lon = AIRPORT_LON + radius_lon * np.cos(theta)
        circle_lat = AIRPORT_LAT + radius_lat * np.sin(theta)
        ax.plot(circle_lon, circle_lat, 'w--', linewidth=1.0, alpha=1.0, zorder=15,
                label=f'{radius_km:.1f}km radius')

    ax.legend(loc='upper right', fontsize=7, facecolor=BG_PANEL, edgecolor=BORDER_COLOR, labelcolor=TEXT_PRIMARY)

    pct = np.sum(cov) / cov.size * 100
    ax.text(
        0.02, 0.98, f'{pct:.1f}%', transform=ax.transAxes, fontsize=9, va='top',
        color=TEXT_PRIMARY, fontweight='600',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=BG_PANEL, alpha=0.9, edgecolor=BORDER_COLOR)
    )

    ax.set_xlabel('Lon', fontsize=8, color=TEXT_SECONDARY)
    ax.set_ylabel('Lat', fontsize=8, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)

    ax.set_title(f'FL{int(fl)}', fontsize=10, fontweight='600', color=ACCENT_PRIMARY)
    plt.tight_layout()
    return fig


def plot_coverage_large(
    cov, lats, lons, fl, terrain=None, radar_lat=None, radar_lon=None,
    bg="relief",
    basemap_provider="carto",   # <-- NEW
    basemap_labels=False,       # <-- NEW
    show_blocked=True, green_alpha=0.5, red_alpha=0.3, figsize=(11, 9),
    for_export=False  # If True, use white background with dark text
):
    """Large coverage map for expanded view."""
    # Color scheme based on context
    if for_export:
        bg_color = 'white'
        text_color = '#1a1a1a'
        text_secondary_color = '#333333'
        border_color = '#cccccc'
        legend_bg = 'white'
    else:
        bg_color = BG_PANEL
        text_color = TEXT_PRIMARY
        text_secondary_color = TEXT_SECONDARY
        border_color = BORDER_COLOR
        legend_bg = BG_PANEL
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    if bg == "terrain" and terrain is not None:
        # Terrain colormap using imshow
        Z_plot = np.ma.masked_where(terrain < -100, terrain)
        ax.imshow(Z_plot, extent=extent, aspect='auto', origin='lower',
                  cmap='terrain', interpolation='bilinear', zorder=1)
    elif bg == "relief" and terrain is not None:
        ax.imshow(colored_relief(terrain), aspect='auto', origin='lower', extent=extent, interpolation='bilinear')
    elif bg == "basemap":
        try:
            _add_basemap(ax, extent, provider=basemap_provider, add_labels=basemap_labels)
        except Exception:
            if terrain is not None:
                ax.imshow(colored_relief(terrain), aspect='auto', origin='lower', extent=extent, interpolation='bilinear')

    if show_blocked:
        colors = [
            (*[int(BLOCKED_AREA[i:i+2], 16)/255 for i in (1, 3, 5)], red_alpha),
            (*[int(COVERAGE_VISIBLE[i:i+2], 16)/255 for i in (1, 3, 5)], green_alpha),
        ]
    else:
        colors = [
            (0, 0, 0, 0),
            (*[int(COVERAGE_VISIBLE[i:i+2], 16)/255 for i in (1, 3, 5)], green_alpha),
        ]

    im = ax.imshow(
        cov.astype(float),
        cmap=ListedColormap(colors),
        aspect='auto',
        origin='lower',
        extent=extent,
        interpolation='nearest',
        vmin=0, vmax=1,
        zorder=2
    )

    if radar_lat is not None and radar_lon is not None:
        ax.plot(radar_lon, radar_lat, '*', color=ACCENT_PRIMARY, markersize=12, label='Radar',
                markeredgecolor='white', markeredgewidth=1, zorder=10)

    ax.plot(AIRPORT_LON, AIRPORT_LAT, '^', color=TEXT_PRIMARY, markersize=10, label='Airport',
            markeredgecolor='black', markeredgewidth=0.8, zorder=10)

    # Draw airport range ring for FL5, FL10, FL20, FL50
    if fl in AIRPORT_RANGE_RINGS:
        radius_km = AIRPORT_RANGE_RINGS[fl]
        lat_rad = np.radians(AIRPORT_LAT)
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(lat_rad)
        radius_lat = radius_km / km_per_deg_lat
        radius_lon = radius_km / km_per_deg_lon
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_lon = AIRPORT_LON + radius_lon * np.cos(theta)
        circle_lat = AIRPORT_LAT + radius_lat * np.sin(theta)
        ax.plot(circle_lon, circle_lat, 'w--', linewidth=1.5, alpha=1.0, zorder=15,
                label=f'{radius_km:.1f}km radius')

    ax.legend(loc='upper right', fontsize=11, facecolor=legend_bg, edgecolor=border_color, labelcolor=text_color)

    pct = np.sum(cov) / cov.size * 100
    ax.text(
        0.02, 0.98, f'{pct:.1f}% visible', transform=ax.transAxes, fontsize=12, va='top',
        color=text_color, fontweight='600',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=legend_bg, alpha=0.95, edgecolor=border_color)
    )

    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    if show_blocked:
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['Blocked', 'Visible'])
    else:
        cbar.set_ticks([0.75])
        cbar.set_ticklabels(['Visible'])
    cbar.ax.tick_params(colors=text_secondary_color)

    ax.set_xlabel('Longitude (°)', fontsize=11, color=text_secondary_color)
    ax.set_ylabel('Latitude (°)', fontsize=11, color=text_secondary_color)
    ax.tick_params(colors=text_secondary_color, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(border_color)

    ax.set_title(f'Radar Coverage - FL{int(fl)}', fontsize=14, fontweight='600', color=ACCENT_PRIMARY)
    plt.tight_layout()
    return fig



def plot_all_coverage_grid(
    maps,
    lats,
    lons,
    terrain=None,
    radar_lat=None,
    radar_lon=None,
    bg="relief",
    basemap_provider="carto",
    basemap_labels=False,
    show_blocked=True,
    green_alpha=0.5,
    red_alpha=0.3,
    figsize=(24, 12),
    for_export=False  # If True, use white background with dark text
):
    """
    Create a single figure with all 8 flight levels in a 4x2 grid.
    Always shows all 8 standard FLs: 5, 10, 20, 50, 100, 200, 300, 400.
    Layout: 4 columns x 2 rows.
    """
    # Color scheme based on context
    if for_export:
        bg_color = 'white'
        text_color = '#1a1a1a'
        text_secondary_color = '#333333'
        border_color = '#cccccc'
        legend_bg = 'white'
    else:
        bg_color = BG_PANEL
        text_color = TEXT_PRIMARY
        text_secondary_color = TEXT_SECONDARY
        border_color = BORDER_COLOR
        legend_bg = BG_PANEL
    
    # Always use all 8 standard FLs
    all_fls = [5, 10, 20, 50, 100, 200, 300, 400]
    
    # Fixed layout: 2 rows x 4 columns
    n_rows, n_cols = 2, 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.patch.set_facecolor(bg_color)
    
    axes_flat = axes.flatten()
    
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    
    # Coverage colors
    if show_blocked:
        colors = [
            (*[int(BLOCKED_AREA[i:i+2], 16)/255 for i in (1, 3, 5)], red_alpha),
            (*[int(COVERAGE_VISIBLE[i:i+2], 16)/255 for i in (1, 3, 5)], green_alpha),
        ]
    else:
        colors = [
            (0, 0, 0, 0),
            (*[int(COVERAGE_VISIBLE[i:i+2], 16)/255 for i in (1, 3, 5)], green_alpha),
        ]
    cmap = ListedColormap(colors)
    
    for idx, fl in enumerate(all_fls):
        ax = axes_flat[idx]
        ax.set_facecolor(bg_color)
        
        # Check if we have coverage data for this FL
        if fl in maps:
            cov = maps[fl]
            
            # Background
            if bg == "terrain" and terrain is not None:
                Z_plot = np.ma.masked_where(terrain < -100, terrain)
                ax.imshow(Z_plot, extent=extent, aspect='auto', origin='lower',
                          cmap='terrain', interpolation='bilinear', zorder=1)
            elif bg == "relief" and terrain is not None:
                ax.imshow(
                    colored_relief(terrain),
                    aspect='auto', origin='lower', extent=extent,
                    interpolation='bilinear'
                )
            elif bg == "basemap":
                try:
                    _add_basemap(ax, extent, provider=basemap_provider, add_labels=basemap_labels)
                except Exception:
                    if terrain is not None:
                        ax.imshow(colored_relief(terrain), aspect='auto', origin='lower', extent=extent, interpolation='bilinear')
            
            # Coverage overlay
            ax.imshow(
                cov.astype(float),
                cmap=cmap,
                aspect='auto',
                origin='lower',
                extent=extent,
                interpolation='nearest',
                vmin=0, vmax=1,
                zorder=2
            )
            
            # Coverage percentage
            pct = np.sum(cov) / cov.size * 100
            ax.text(
                0.02, 0.98, f'{pct:.1f}%', transform=ax.transAxes, fontsize=12, va='top',
                color=text_color, fontweight='700',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=legend_bg, alpha=0.9, edgecolor=border_color)
            )
        else:
            # No data for this FL - show terrain only with "Not computed" label
            if bg == "terrain" and terrain is not None:
                Z_plot = np.ma.masked_where(terrain < -100, terrain)
                ax.imshow(Z_plot, extent=extent, aspect='auto', origin='lower',
                          cmap='terrain', interpolation='bilinear', zorder=1)
            elif bg == "relief" and terrain is not None:
                ax.imshow(
                    colored_relief(terrain),
                    aspect='auto', origin='lower', extent=extent,
                    interpolation='bilinear'
                )
            elif bg == "basemap":
                try:
                    _add_basemap(ax, extent, provider=basemap_provider, add_labels=basemap_labels)
                except Exception:
                    if terrain is not None:
                        ax.imshow(colored_relief(terrain), aspect='auto', origin='lower', extent=extent, interpolation='bilinear')
            
            ax.text(
                0.5, 0.5, 'Not computed', transform=ax.transAxes, fontsize=14, 
                ha='center', va='center', color=text_secondary_color, fontweight='600',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=legend_bg, alpha=0.9, edgecolor=border_color)
            )
        
        # Markers (always show)
        if radar_lat is not None and radar_lon is not None:
            ax.plot(radar_lon, radar_lat, '*', color=ACCENT_PRIMARY, markersize=9,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=10)
        
        ax.plot(AIRPORT_LON, AIRPORT_LAT, '^', color=TEXT_PRIMARY, markersize=7,
                markeredgecolor='black', markeredgewidth=0.6, zorder=10)
        
        # Draw airport range ring for FL5, FL10, FL20, FL50
        if fl in AIRPORT_RANGE_RINGS:
            radius_km = AIRPORT_RANGE_RINGS[fl]
            lat_rad = np.radians(AIRPORT_LAT)
            km_per_deg_lat = 111.0
            km_per_deg_lon = 111.0 * np.cos(lat_rad)
            radius_lat = radius_km / km_per_deg_lat
            radius_lon = radius_km / km_per_deg_lon
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_lon = AIRPORT_LON + radius_lon * np.cos(theta)
            circle_lat = AIRPORT_LAT + radius_lat * np.sin(theta)
            ax.plot(circle_lon, circle_lat, 'w--', linewidth=1.2, alpha=1.0, zorder=15)
            # Add text label near the circle showing the radius
            # Place label at top of circle
            label_lon = AIRPORT_LON
            label_lat = AIRPORT_LAT + radius_lat * 1.1
            ax.text(label_lon, label_lat, f'{radius_km:.1f}km', fontsize=8, color='white',
                    ha='center', va='bottom', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6, edgecolor='none'))
        
        # Title with FL and altitude
        alt_m = fl_to_m(fl)
        ax.set_title(f'FL{int(fl)} ({alt_m:.0f}m)', fontsize=14, fontweight='700', color=ACCENT_PRIMARY, pad=10)
        
        # Axis styling
        ax.tick_params(colors=text_secondary_color, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(border_color)
    
    # Add overall title
    fig.suptitle(
        'Radar Coverage Analysis - All Flight Levels',
        fontsize=18, fontweight='700', color=text_color, y=0.98
    )
    
    # Add legend for the whole figure
    from matplotlib.patches import Patch
    # Use appropriate marker background for legend visibility
    marker_bg = bg_color if for_export else 'w'
    legend_elements = [
        Patch(facecolor=COVERAGE_VISIBLE, alpha=green_alpha, label='Visible'),
        plt.Line2D([0], [0], marker='*', color=marker_bg, markerfacecolor=ACCENT_PRIMARY, 
                   markersize=9, label='Radar', markeredgecolor='white'),
        plt.Line2D([0], [0], marker='^', color=marker_bg, markerfacecolor=TEXT_PRIMARY, 
                   markersize=7, label='Airport', markeredgecolor='black'),
        plt.Line2D([0], [0], linestyle='--', color='white', linewidth=1.5, 
                   label='Airport Range'),
    ]
    if show_blocked:
        legend_elements.insert(1, Patch(facecolor=BLOCKED_AREA, alpha=red_alpha, label='Blocked'))
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements),
               fontsize=11, facecolor=legend_bg, edgecolor=border_color, labelcolor=text_color,
               bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


def create_png_zip(
    maps,
    lats,
    lons,
    terrain,
    radar_lat,
    radar_lon,
    radar_h,
    bg,
    show_blocked,
    green_alpha,
    red_alpha,
    basemap_provider="carto",
    basemap_labels=False,
    fls_to_export=None,
    dpi=150,
    X_m=None,
    Y_m=None,
    n_samples=200,
    margin=0.0,
):
    """
    Create a ZIP file with PNG coverage maps.
    
    The combined image always shows all 8 standard FLs.
    Missing FLs are computed on-the-fly.
    """
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        fls = sorted(maps.keys()) if fls_to_export is None else list(fls_to_export)
        
        # Individual FL images (only for selected FLs)
        for fl in fls:
            fig = plot_coverage_large(
                cov=maps[fl],
                lats=lats,
                lons=lons,
                fl=fl,
                terrain=terrain,
                radar_lat=radar_lat,
                radar_lon=radar_lon,
                bg=bg,
                basemap_provider=basemap_provider,
                basemap_labels=basemap_labels,
                show_blocked=show_blocked,
                green_alpha=green_alpha,
                red_alpha=red_alpha,
                for_export=True,  # Use white background for export
            )
            img_buf = BytesIO()
            fig.savefig(img_buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            img_buf.seek(0)
            zf.writestr(f'coverage_FL{int(fl)}.png', img_buf.getvalue())
        
        # Combined grid image with ALL 8 FLs (always computed)
        all_fls = [5, 10, 20, 50, 100, 200, 300, 400]
        maps_all = dict(maps)  # Copy existing maps
        
        # Compute missing FLs if we have the required data
        if X_m is not None and Y_m is not None:
            missing_fls = [fl for fl in all_fls if fl not in maps_all]
            for fl in missing_fls:
                cov = compute_fl_coverage_curvature(
                    radar_lat, radar_lon, radar_h, fl,
                    X_m, Y_m, terrain,
                    n_samples=n_samples, margin=margin
                )
                maps_all[fl] = cov
        
        fig_grid = plot_all_coverage_grid(
            maps=maps_all,
            lats=lats,
            lons=lons,
            terrain=terrain,
            radar_lat=radar_lat,
            radar_lon=radar_lon,
            bg=bg,
            basemap_provider=basemap_provider,
            basemap_labels=basemap_labels,
            show_blocked=show_blocked,
            green_alpha=green_alpha,
            red_alpha=red_alpha,
            for_export=True,  # Use white background for export
        )
        img_buf_grid = BytesIO()
        fig_grid.savefig(img_buf_grid, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig_grid)
        img_buf_grid.seek(0)
        zf.writestr('coverage_ALL_FLs.png', img_buf_grid.getvalue())
    
    return buf.getvalue()


def create_coverage_csv(maps, radar_lat, radar_lon, radar_h):
    """Create CSV content with coverage statistics."""
    lines = []
    lines.append("# Radar Coverage Analysis - Statistics Export")
    lines.append(f"# Radar Position: {radar_lat:.6f}N, {radar_lon:.6f}E")
    lines.append(f"# Radar Height: {radar_h}m AGL")
    lines.append("")
    lines.append("Flight Level,Altitude (m),Altitude (ft),Visible Points,Total Points,Coverage (%)")
    
    for fl in sorted(maps.keys()):
        cov = maps[fl]
        vis = np.sum(cov)
        tot = cov.size
        pct = 100 * vis / tot
        alt_m = fl_to_m(fl)
        alt_ft = int(fl) * 100
        lines.append(f"FL{int(fl)},{alt_m:.0f},{alt_ft},{vis},{tot},{pct:.4f}")
    
    return "\n".join(lines)


# ============================================================================
# KMZ EXPORT
# ============================================================================

def export_kmz(maps, lats, lons, radar_lat=None, radar_lon=None, show_blocked=True):
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = "Radar Coverage"
    
    vis = ET.SubElement(doc, "Style", id="vis")
    ps = ET.SubElement(vis, "PolyStyle")
    ET.SubElement(ps, "color").text = "7f4cff9a" # Thales green
    ET.SubElement(ps, "fill").text = "1"
    ET.SubElement(ps, "outline").text = "0"
    
    if show_blocked:
        blk = ET.SubElement(doc, "Style", id="blk")
        ps2 = ET.SubElement(blk, "PolyStyle")
        ET.SubElement(ps2, "color").text = "7f3939e6" # Thales alert red
        ET.SubElement(ps2, "fill").text = "1"
        ET.SubElement(ps2, "outline").text = "0"
    
    step = max(1, min(len(lats) // 100, len(lons) // 100))
    for fl in sorted(maps.keys()):
        folder = ET.SubElement(doc, "Folder")
        ET.SubElement(folder, "name").text = f"FL{int(fl)}"
        cov = maps[fl]
        for i in range(0, len(lats) - 1, step):
            for j in range(0, len(lons) - 1, step):
                lat0, lat1 = lats[i], lats[min(i + step, len(lats) - 1)]
                lon0, lon1 = lons[j], lons[min(j + step, len(lons) - 1)]
                st = cov[i, j]
                if not st and not show_blocked:
                    continue
                pm = ET.SubElement(folder, "Placemark")
                ET.SubElement(pm, "styleUrl").text = "#vis" if st else "#blk"
                poly = ET.SubElement(pm, "Polygon")
                ob = ET.SubElement(poly, "outerBoundaryIs")
                lr = ET.SubElement(ob, "LinearRing")
                coords = ET.SubElement(lr, "coordinates")
                coords.text = f"{lon0},{lat0},0 {lon1},{lat0},0 {lon1},{lat1},0 {lon0},{lat1},0 {lon0},{lat0},0"
    
    ref = ET.SubElement(doc, "Folder")
    ET.SubElement(ref, "name").text = "Reference"
    if radar_lat and radar_lon:
        rpm = ET.SubElement(ref, "Placemark")
        ET.SubElement(rpm, "name").text = "Radar"
        rpt = ET.SubElement(rpm, "Point")
        ET.SubElement(rpt, "coordinates").text = f"{radar_lon},{radar_lat},0"
    apm = ET.SubElement(ref, "Placemark")
    ET.SubElement(apm, "name").text = AIRPORT_NAME
    apt = ET.SubElement(apm, "Point")
    ET.SubElement(apt, "coordinates").text = f"{AIRPORT_LON},{AIRPORT_LAT},0"
    
    kml_str = ET.tostring(kml, encoding='utf-8', xml_declaration=True)
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as kmz:
        kmz.writestr("doc.kml", kml_str)
    return buf.getvalue()


# ============================================================================
# MAIN APP - Landing Page
# ============================================================================

def main():
    """
    Landing page for the Radar Coverage Analysis multi-page application.
    
    This serves as the home page with navigation to:
    - Coverage Analysis (terrain visualization & LOS computation)
    - Site Selection (coming soon - constraint-based site selection)
    """
    st.set_page_config(
        page_title="Radar Coverage Analysis",
        page_icon="📡",
        layout="wide"
    )
    apply_theme()
    render_title()
    
    # Welcome section
    st.markdown(f"""
        <div style='background-color: {BG_CARD}; padding: 30px; border-radius: 16px; border: 1px solid {BORDER_COLOR}; margin-bottom: 2rem;'>
            <p style='color: {TEXT_SECONDARY}; font-size: 1.1rem; line-height: 1.6; margin: 0;'>
                Welcome to the Radar Coverage Analysis Software. This application provides comprehensive tools for 
                analyzing radar line-of-sight coverage over terrain, with Earth curvature correction for accurate results.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    st.header("Available Modules")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div style='background-color: {BG_CARD}; padding: 25px; border-radius: 12px; border: 1px solid {BORDER_COLOR}; height: 100%;'>
                <h3 style='color: {ACCENT_PRIMARY}; margin-top: 0;'>Coverage Analysis</h3>
                <p style='color: {TEXT_SECONDARY}; margin-bottom: 1rem;'>
                    Compute and visualize radar coverage across multiple flight levels with Earth curvature correction.
                </p>
                <ul style='color: {TEXT_SECONDARY}; margin-bottom: 1.5rem;'>
                    <li>Load terrain data from NPZ files</li>
                    <li>2D and 3D terrain visualization</li>
                    <li>Coverage computation for FL5-FL400</li>
                    <li>Export to KMZ, PNG, and CSV</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Coverage Analysis", type="primary", use_container_width=True, key="btn_coverage"):
            st.switch_page("pages/1_Coverage_Analysis.py")
    
    with col2:
        st.markdown(f"""
            <div style='background-color: {BG_CARD}; padding: 25px; border-radius: 12px; border: 1px solid {BORDER_COLOR}; height: 100%;'>
                <h3 style='color: {ACCENT_PRIMARY}; margin-top: 0;'>Site Selection</h3>
                <p style='color: {TEXT_SECONDARY}; margin-bottom: 1rem;'>
                    Find optimal radar installation sites using geographic constraints and coverage scoring.
                </p>
                <ul style='color: {TEXT_SECONDARY}; margin-bottom: 1.5rem;'>
                    <li>Predefined constraints (land, slope, coastline)</li>
                    <li>User-defined constraints (GeoJSON upload)</li>
                    <li>Interactive constraint visualization</li>
                    <li>Candidate site ranking and export</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Site Selection", type="primary", use_container_width=True, key="btn_site"):
            st.switch_page("pages/2_Site_Selection.py")
    
    # Technical info section
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("Technical Information")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown(f"""
            <div style='background-color: {BG_PANEL}; padding: 20px; border-radius: 10px; border: 1px solid {BORDER_COLOR};'>
                <h4 style='color: {ACCENT_PRIMARY}; margin-top: 0;'>Earth Curvature</h4>
                <p style='color: {TEXT_SECONDARY}; font-size: 0.9rem; margin-bottom: 0;'>
                    All coverage computations include Earth curvature correction for accurate 
                    line-of-sight calculations at long ranges (up to 50km radius).
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with tech_col2:
        st.markdown(f"""
            <div style='background-color: {BG_PANEL}; padding: 20px; border-radius: 10px; border: 1px solid {BORDER_COLOR};'>
                <h4 style='color: {ACCENT_PRIMARY}; margin-top: 0;'>Reference Point</h4>
                <p style='color: {TEXT_SECONDARY}; font-size: 0.9rem; margin-bottom: 0;'>
                    Nice Airport (LFMN) at {AIRPORT_LAT}°N, {AIRPORT_LON}°E serves as the 
                    reference point for coordinate transformations.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with tech_col3:
        numba_status = "Available" if HAS_NUMBA else "Not installed"
        ctx_status = "Available" if HAS_CONTEXTILY else "Not installed"
        curv_status = "Available" if HAS_CURVATURE_MODULES else "Fallback mode"
        
        st.markdown(f"""
            <div style='background-color: {BG_PANEL}; padding: 20px; border-radius: 10px; border: 1px solid {BORDER_COLOR};'>
                <h4 style='color: {ACCENT_PRIMARY}; margin-top: 0;'>Dependencies</h4>
                <p style='color: {TEXT_SECONDARY}; font-size: 0.9rem; margin-bottom: 0;'>
                    Numba: {numba_status}<br>
                    Contextily: {ctx_status}<br>
                    Curvature modules: {curv_status}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("Quick Start")
    
    st.markdown(f"""
        <div style='background-color: {BG_CARD}; padding: 25px; border-radius: 12px; border: 1px solid {BORDER_COLOR};'>
            <ol style='color: {TEXT_SECONDARY}; font-size: 1rem; line-height: 1.8; margin: 0; padding-left: 1.5rem;'>
                <li>Navigate to <strong style='color: {TEXT_PRIMARY};'>Coverage Analysis</strong> using the sidebar or button above</li>
                <li>Load terrain data (sample terrain available) and configure radar position</li>
                <li>Select desired flight levels and click <strong style='color: {TEXT_PRIMARY};'>Compute Coverage</strong></li>
                <li>View coverage maps and statistics, then export results in your preferred format</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='text-align: center; color: {TEXT_SECONDARY}; font-size: 0.85rem; opacity: 0.7;'>
            Radar Coverage Analysis Software | Advanced Terrain Modeling & Line-of-Sight Computing
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
