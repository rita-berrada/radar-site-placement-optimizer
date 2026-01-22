"""
Radar Coverage Analysis Software

A comprehensive application for radar coverage analysis that:
1. Loads terrain data from NPZ files
2. Visualizes terrain in 2D and 3D
3. Computes coverage on Flight Levels (FL)
4. Outputs coverage maps with various backgrounds
5. Exports results to KMZ format

Author: Radar Analysis Team
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
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
    data = np.load(npz_file)
    lats = data['lat'].astype(float)
    lons = data['lon'].astype(float)
    Z = data['ter'].astype(float)
    if Z.shape != (len(lats), len(lons)):
        raise ValueError(f"Shape mismatch: Z{Z.shape} vs ({len(lats)}, {len(lons)})")
    return lats, lons, Z


def convert_to_enu(lats, lons, ref_lat, ref_lon):
    lat_ref_rad = np.radians(ref_lat)
    meters_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
    meters_per_deg_lon = (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(lat_ref_rad)
    Y_m = (lats - ref_lat) * meters_per_deg_lat
    X_m = (lons - ref_lon) * meters_per_deg_lon
    return X_m, Y_m


def normalize_xy_grid(X_m, Y_m, Z):
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


# ============================================================================
# COVERAGE COMPUTATION
# ============================================================================

def fl_to_m(FL):
    return FL * 100.0 * 0.3048


def latlon_to_xy_m(lat, lon, ref_lat, ref_lon):
    lat_ref_rad = np.radians(ref_lat)
    meters_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
    meters_per_deg_lon = (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(lat_ref_rad)
    y = (float(lat) - float(ref_lat)) * meters_per_deg_lat
    x = (float(lon) - float(ref_lon)) * meters_per_deg_lon
    return x, y


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
    def los_check(radar_x, radar_y, radar_h, target_x, target_y, target_alt, x0, y0, dx, dy, Z, n_samples, margin):
        z_r = z_bilinear(radar_x, radar_y, x0, y0, dx, dy, Z)
        if np.isnan(z_r):
            return False
        z_radar = z_r + radar_h
        for k in range(1, n_samples):
            s = k / n_samples
            x = radar_x + s * (target_x - radar_x)
            y = radar_y + s * (target_y - radar_y)
            z_g = z_bilinear(x, y, x0, y0, dx, dy, Z)
            if np.isnan(z_g):
                return False
            z_line = z_radar + s * (target_alt - z_radar)
            if z_g + margin >= z_line:
                return False
        return True

    @njit(parallel=True)
    def compute_coverage(radar_x, radar_y, radar_h, target_alt, X_m, Y_m, x0, y0, dx, dy, Z, n_samples, margin):
        N, M = Y_m.size, X_m.size
        out = np.zeros((N, M), dtype=np.bool_)
        for i in prange(N):
            for j in range(M):
                out[i, j] = los_check(radar_x, radar_y, radar_h, X_m[j], Y_m[i], target_alt, x0, y0, dx, dy, Z, n_samples, margin)
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

    def compute_coverage(radar_x, radar_y, radar_h, target_alt, X_m, Y_m, x0, y0, dx, dy, Z, n_samples, margin):
        N, M = Y_m.size, X_m.size
        out = np.zeros((N, M), dtype=np.bool_)
        z_r = z_bilinear(radar_x, radar_y, x0, y0, dx, dy, Z)
        if np.isnan(z_r):
            return out
        z_radar = z_r + radar_h
        for i in range(N):
            for j in range(M):
                visible = True
                for k in range(1, n_samples):
                    s = k / n_samples
                    x = radar_x + s * (X_m[j] - radar_x)
                    y = radar_y + s * (Y_m[i] - radar_y)
                    z_g = z_bilinear(x, y, x0, y0, dx, dy, Z)
                    if np.isnan(z_g):
                        visible = False
                        break
                    z_line = z_radar + s * (target_alt - z_radar)
                    if z_g + margin >= z_line:
                        visible = False
                        break
                out[i, j] = visible
        return out


def compute_fl_coverage(radar_lat, radar_lon, radar_h, fl, X_m, Y_m, Z, ref_lat, ref_lon, n_samples=400, margin=0.0):
    radar_x, radar_y = latlon_to_xy_m(radar_lat, radar_lon, ref_lat, ref_lon)
    x0, y0 = float(X_m[0]), float(Y_m[0])
    dx = float(X_m[1] - X_m[0]) if len(X_m) > 1 else 1.0
    dy = float(Y_m[1] - Y_m[0]) if len(Y_m) > 1 else 1.0
    return compute_coverage(float(radar_x), float(radar_y), float(radar_h), float(fl_to_m(fl)),
                            X_m, Y_m, x0, y0, dx, dy, Z, int(n_samples), float(margin))


def compute_all_fl(radar_lat, radar_lon, radar_h, fls, X_m, Y_m, Z, ref_lat, ref_lon, n_samples=400, margin=0.0, callback=None):
    maps = {}
    for i, fl in enumerate(fls):
        maps[fl] = compute_fl_coverage(radar_lat, radar_lon, radar_h, fl, X_m, Y_m, Z, ref_lat, ref_lon, n_samples, margin)
        if callback:
            callback((i + 1) / len(fls))
    return maps


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
        ax.plot(radar_lon, radar_lat, '*', color=ACCENT_PRIMARY, markersize=16, label='Radar',
                markeredgecolor='white', markeredgewidth=1.5)
    
    # Airport marker - White
    ax.plot(AIRPORT_LON, AIRPORT_LAT, '^', color=TEXT_PRIMARY, markersize=12, label='Airport',
            markeredgecolor='black', markeredgewidth=1)
    
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

    if bg == "relief" and terrain is not None:
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

    # Coverage overlay
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

    ax.imshow(
        cov.astype(float),
        cmap=ListedColormap(colors),
        aspect='auto',
        origin='lower',
        extent=extent,
        interpolation='nearest',
        vmin=0, vmax=1,
        zorder=2
    )

    # Markers
    if radar_lat is not None and radar_lon is not None:
        ax.plot(radar_lon, radar_lat, '*', color=ACCENT_PRIMARY, markersize=10, label='Radar',
                markeredgecolor='white', markeredgewidth=0.8, zorder=10)

    ax.plot(AIRPORT_LON, AIRPORT_LAT, '^', color=TEXT_PRIMARY, markersize=8, label='Airport',
            markeredgecolor='black', markeredgewidth=0.5, zorder=10)

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
    show_blocked=True, green_alpha=0.5, red_alpha=0.3, figsize=(11, 9)
):
    """Large coverage map for expanded view."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG_PANEL)
    ax.set_facecolor(BG_PANEL)

    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    if bg == "relief" and terrain is not None:
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
        ax.plot(radar_lon, radar_lat, '*', color=ACCENT_PRIMARY, markersize=18, label='Radar',
                markeredgecolor='white', markeredgewidth=1.5, zorder=10)

    ax.plot(AIRPORT_LON, AIRPORT_LAT, '^', color=TEXT_PRIMARY, markersize=14, label='Airport',
            markeredgecolor='black', markeredgewidth=1, zorder=10)

    ax.legend(loc='upper right', fontsize=11, facecolor=BG_PANEL, edgecolor=BORDER_COLOR, labelcolor=TEXT_PRIMARY)

    pct = np.sum(cov) / cov.size * 100
    ax.text(
        0.02, 0.98, f'{pct:.1f}% visible', transform=ax.transAxes, fontsize=12, va='top',
        color=TEXT_PRIMARY, fontweight='600',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=BG_PANEL, alpha=0.95, edgecolor=BORDER_COLOR)
    )

    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    if show_blocked:
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['Blocked', 'Visible'])
    else:
        cbar.set_ticks([0.75])
        cbar.set_ticklabels(['Visible'])
    cbar.ax.tick_params(colors=TEXT_SECONDARY)

    ax.set_xlabel('Longitude (°)', fontsize=11, color=TEXT_SECONDARY)
    ax.set_ylabel('Latitude (°)', fontsize=11, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)

    ax.set_title(f'Radar Coverage - FL{int(fl)}', fontsize=14, fontweight='600', color=ACCENT_PRIMARY)
    plt.tight_layout()
    return fig



def create_png_zip(
    maps,
    lats,
    lons,
    terrain,
    radar_lat,
    radar_lon,
    bg,
    show_blocked,
    green_alpha,
    red_alpha,
    basemap_provider="carto",
    basemap_labels=False,
    fls_to_export=None,
    dpi=150,
):
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        fls = sorted(maps.keys()) if fls_to_export is None else list(fls_to_export)
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
            )
            img_buf = BytesIO()
            fig.savefig(img_buf, format='png', dpi=dpi, bbox_inches='tight', facecolor=BG_PANEL)
            plt.close(fig)
            img_buf.seek(0)
            zf.writestr(f'coverage_FL{int(fl)}.png', img_buf.getvalue())
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
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(page_title="Radar Coverage Analysis", page_icon="📡", layout="wide")
    apply_theme()
    render_title()
    
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
                lats, lons, Z = load_terrain_npz(uploaded)
                terrain_data = {'lats': lats, 'lons': lons, 'Z': Z}
            except Exception as e:
                st.error(f"Error loading terrain: {e}")
        elif use_sample:
            sample_path = Path(__file__).parent / "terrain_mat.npz"
            if sample_path.exists():
                lats, lons, Z = load_terrain_npz(str(sample_path))
                terrain_data = {'lats': lats, 'lons': lons, 'Z': Z}
        
        if terrain_data:
            lats, lons, Z = terrain_data['lats'], terrain_data['lons'], terrain_data['Z']
            
            # Radar Position Expander
            with st.expander("Radar Position", expanded=True):
                radar_lat = st.number_input("Latitude", value=DEFAULT_REF_LAT, format="%.6f")
                radar_lon = st.number_input("Longitude", value=DEFAULT_REF_LON, format="%.6f")
                radar_h = st.number_input("Height (m AGL)", value=20.0, min_value=0.0)
            
            # Reference coordinates (derived from radar position by default)
            ref_lat = radar_lat
            ref_lon = radar_lon
            X_m, Y_m = convert_to_enu(lats, lons, ref_lat, ref_lon)
            X_m, Y_m, Z_enu = normalize_xy_grid(X_m, Y_m, Z.copy())
            
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
        lats, lons, Z = terrain_data['lats'], terrain_data['lons'], terrain_data['Z']
        
        # Reference coordinates (derived from radar position)
        ref_lat = radar_lat
        ref_lon = radar_lon
        X_m, Y_m = convert_to_enu(lats, lons, ref_lat, ref_lon)
        X_m, Y_m, Z_enu = normalize_xy_grid(X_m, Y_m, Z.copy())
        
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
        if (main_compute or sidebar_compute) and selected_fls:
            prog = st.progress(0)
            stat = st.empty()
            def upd(p):
                prog.progress(p)
                stat.text(f"Computing... {p*100:.0f}%")
            maps = compute_all_fl(radar_lat, radar_lon, radar_h, sorted(selected_fls), X_m, Y_m, Z_enu, ref_lat, ref_lon, n_samples, margin, upd)
            st.session_state.coverage_maps = maps
            st.session_state.coverage_computed = True
            st.session_state.coverage_lats = lats
            st.session_state.coverage_lons = lons
            st.session_state.coverage_terrain = Z
            st.session_state.radar_lat = radar_lat
            st.session_state.radar_lon = radar_lon
            st.session_state.radar_h = radar_h
            st.session_state.expanded_fl = None
            prog.progress(1.0)
            stat.text("Done!")
        
        #         # =====================================================================
        # SECTION 2: COVERAGE ANALYSIS (After computation)
        # =====================================================================
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.header("Coverage Analysis")

        if not st.session_state.coverage_computed or 'coverage_maps' not in st.session_state:
            st.info("Configure settings in the sidebar and click **Compute Coverage** to see results.")
        else:
            maps = st.session_state.coverage_maps
            sorted_fls = sorted(maps.keys())
            
            # Coverage statistics table (all FLs) - shown FIRST
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
            
            st.markdown("---")
            
            # Coverage maps display options
            st.subheader("Coverage Maps")
            map_opt_col1, map_opt_col2, map_opt_col3, map_opt_col4 = st.columns(4)
            with map_opt_col1:
                map_style = st.selectbox(
                    "Background",
                    ["Satellite (Esri)", "Street (Carto)"],
                    index=0,  # default to satellite
                    key="coverage_map_style"
                    )
            with map_opt_col2:
                map_show_blocked = st.toggle("Show blocked areas", value=False, key="coverage_map_blocked")
            with map_opt_col3:
                map_green_alpha = st.slider("Visible opacity", 0.1, 1.0, 0.6, 0.05, key="coverage_green_alpha")
            with map_opt_col4:
                map_red_alpha = st.slider("Blocked opacity", 0.1, 1.0, 0.5, 0.05, key="coverage_red_alpha")

            # Map UI style -> plotting params
            if map_style == "Satellite (Esri)":
                plot_bg = "basemap"
                basemap_provider = "esri"
                basemap_labels = False
            else:  # "Street (Carto)"
                plot_bg = "basemap"
                basemap_provider = "carto"
                basemap_labels = False

            
            # All coverage maps in a grid (max 2 per row)
            # st.markdown("<h4 style='text-align: center;'>All Flight Levels</h4>", unsafe_allow_html=True)
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
                            terrain=st.session_state.coverage_terrain,     # on garde toujours le terrain dispo pour fallback
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
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # =====================================================================
        # SECTION 3: EXPORT (After computation)
        # =====================================================================
        st.header("Export Results")
        
        if not st.session_state.coverage_computed or 'coverage_maps' not in st.session_state:
            st.info("Compute coverage first to enable exports.")
        else:
            maps = st.session_state.coverage_maps
            
            # st.caption("Download coverage analysis results in various formats")
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
                st.markdown("</div>", unsafe_allow_html=True)
            
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
                    ["Satellite (Esri)", "Street (Carto)"],
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

                if png_style == "Satellite (Esri)":
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
                    bg=png_plot_bg,
                    show_blocked=png_show_blocked,
                    green_alpha=png_green_alpha,
                    red_alpha=png_red_alpha,
                    basemap_provider=png_basemap_provider,
                    basemap_labels=png_basemap_labels,
                    fls_to_export=png_fls,
                    dpi=png_dpi,
                )
                st.download_button(
                    "Download PNG (ZIP)",
                    png_zip,
                    "coverage_maps.zip",
                    "application/zip",
                    use_container_width=True,
                    key="png_download"
                )
                st.markdown("</div>", unsafe_allow_html=True)
            
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
