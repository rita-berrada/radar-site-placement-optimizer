"""
Coverage Visualization Module

This module provides functions to visualize radar coverage maps using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from matplotlib.colors import ListedColormap
from matplotlib import cm

try:
    import contextily as cx  # optional (adds real basemap tiles)

    _HAS_CONTEXTILY = True
except Exception:  # pragma: no cover
    cx = None
    _HAS_CONTEXTILY = False


def _get_basemap_source(name: str):
    """
    Map a friendly provider name to a contextily source.
    Defaults to a neutral, "Google Maps-like" basemap.
    """
    if not _HAS_CONTEXTILY:
        return None

    sources = {
        # Neutral "Google Maps-like" basemap (public tiles)
        "CartoDB.VoyagerNoLabels": getattr(cx.providers.CartoDB, "VoyagerNoLabels", cx.providers.OpenStreetMap.Mapnik),
        "CartoDB.Voyager": getattr(cx.providers.CartoDB, "Voyager", cx.providers.OpenStreetMap.Mapnik),
        "OpenStreetMap.Mapnik": cx.providers.OpenStreetMap.Mapnik,
    }
    return sources.get(name, getattr(cx.providers.CartoDB, "VoyagerNoLabels", cx.providers.OpenStreetMap.Mapnik))


def _add_basemap_latlon(
    ax,
    extent,
    basemap_source,
    basemap_zoom: Optional[int] = None,
    warn: bool = True,
) -> bool:
    """
    Add a web basemap under the current axes (assumes lon/lat degrees, EPSG:4326).
    Returns True if a basemap was drawn, False otherwise.
    """
    if not _HAS_CONTEXTILY or basemap_source is None:
        return False

    # Ensure axes bounds are set before fetching tiles
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    try:
        if basemap_zoom is None:
            cx.add_basemap(ax, crs="EPSG:4326", source=basemap_source)
        else:
            cx.add_basemap(ax, crs="EPSG:4326", source=basemap_source, zoom=int(basemap_zoom))
        return True
    except Exception as e:
        # Network/tiles/optional deps might not be available; fall back gracefully.
        # Don't fail silently so users know why they're seeing the fallback.
        if warn and _HAS_CONTEXTILY:
            print(f"[visualize_coverage] Basemap failed, falling back (reason: {e})")
        return False


def _hillshade(z: np.ndarray, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    """
    Simple hillshade from an elevation grid. Produces a neutral grayscale background
    that makes coverage overlays easier to interpret.

    Notes:
    - Works best when z is a reasonably smooth terrain grid.
    - This is a visualization aid only; it does not affect computations.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError("terrain must be a 2D array")

    # Gradient in grid coordinates (not true meters; sufficient for visual shading)
    dy, dx = np.gradient(z)

    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)

    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)

    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shaded = np.clip(shaded, -1.0, 1.0)

    # Normalize to [0,1]
    mn = float(np.nanmin(shaded))
    mx = float(np.nanmax(shaded))
    return (shaded - mn) / (mx - mn + 1e-12)


def _colored_relief(
    z: np.ndarray,
    cmap_name: str = "terrain",
    hs_azimuth_deg: float = 315.0,
    hs_altitude_deg: float = 45.0,
    desaturate: float = 0.25,
) -> np.ndarray:
    """
    Build a colored shaded-relief image from terrain elevations.

    Returns an RGB image (H,W,3) in [0,1]. This is intended as a realistic-ish
    background that highlights mountains/elevation.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError("terrain must be a 2D array")

    # Normalize elevation for colormap
    z_min = float(np.nanmin(z))
    z_max = float(np.nanmax(z))
    zn = (z - z_min) / (z_max - z_min + 1e-12)
    zn = np.clip(zn, 0.0, 1.0)

    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(zn)[..., :3]  # drop alpha

    # Hillshade modulation
    hs = _hillshade(z, azimuth_deg=hs_azimuth_deg, altitude_deg=hs_altitude_deg)
    shade = 0.55 + 0.45 * hs  # keep it bright enough under overlays
    rgb = rgb * shade[..., None]

    # Slightly desaturate toward gray for a more "map-like" look
    d = float(np.clip(desaturate, 0.0, 1.0))
    gray = np.mean(rgb, axis=2, keepdims=True)
    rgb = (1.0 - d) * rgb + d * gray

    return np.clip(rgb, 0.0, 1.0)


def plot_coverage_map(
    coverage_map: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    flight_level: float,
    radar_lat: Optional[float] = None,
    radar_lon: Optional[float] = None,
    terrain: Optional[np.ndarray] = None,
    visible_alpha: float = 0.45,
    blocked_alpha: float = 0.22,
    visible_color: tuple = (0.00, 0.70, 0.00),  # green
    blocked_color: tuple = (0.85, 0.05, 0.05),  # red
    basemap: bool = True,
    basemap_provider: str = "CartoDB.VoyagerNoLabels",
    basemap_zoom: Optional[int] = None,
    basemap_warn: bool = True,
    background: str = "basemap+relief",  # "basemap", "relief", "hillshade", "basemap+relief"
    relief_alpha: float = 0.70,
    airport_lat: float = 43.6584,
    airport_lon: float = 7.2159,
    airport_label: str = "Nice Airport (LFMN)",
    show_airport: bool = True,
    return_fig: bool = False,
    save_path: Optional[str] = None
) -> Optional["plt.Figure"]:
    """
    Plot a single coverage map as a simple 2D map.
    
    Parameters:
    -----------
    coverage_map : np.ndarray
        2D boolean array (True=visible, False=blocked)
    lats : np.ndarray
        1D array of latitude values
    lons : np.ndarray
        1D array of longitude values
    flight_level : float
        Flight level for title
    radar_lat : float, optional
        Radar latitude to mark on map
    radar_lon : float, optional
        Radar longitude to mark on map
    save_path : str, optional
        If provided, save figure to this path
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Convert boolean to float for visualization
    coverage_float = coverage_map.astype(float)
    
    # Create subtle overlay colormap (separate alphas for readability on basemap)
    va = float(np.clip(visible_alpha, 0.0, 1.0))
    ba = float(np.clip(blocked_alpha, 0.0, 1.0))
    colors = [
        (float(blocked_color[0]), float(blocked_color[1]), float(blocked_color[2]), ba),  # often fully transparent
        (float(visible_color[0]), float(visible_color[1]), float(visible_color[2]), va),
    ]
    cmap = ListedColormap(colors)
    
    # Use imshow for simple 2D map (extent sets the coordinate system)
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    # Background: prefer basemap and/or terrain relief depending on requested mode.
    drew_basemap = False
    bg = (background or "").lower().strip()

    if basemap and ("basemap" in bg):
        drew_basemap = _add_basemap_latlon(
            ax,
            extent,
            basemap_source=_get_basemap_source(basemap_provider),
            basemap_zoom=basemap_zoom,
            warn=basemap_warn,
        )

    # Terrain-based relief overlays mountains/elevations.
    # background:
    # - "relief": colored elevation + hillshade (from terrain Z)
    # - "basemap+relief": basemap tiles + colored relief overlay
    # - "basemap": basemap only
    # - "hillshade": grayscale hillshade only
    if terrain is not None and ("relief" in bg):
        rel = _colored_relief(terrain, cmap_name="terrain")
        ax.imshow(
            rel,
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation="bilinear",
            alpha=float(np.clip(relief_alpha, 0.0, 1.0)),
            zorder=1,
        )
    elif terrain is not None and ("hillshade" in bg) and ("relief" not in bg):
        hs = _hillshade(terrain)
        ax.imshow(
            hs,
            cmap="Greys",
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation="bilinear",
            vmin=0,
            vmax=1,
            alpha=0.95,
            zorder=1,
        )

    # Coverage overlay (transparent)
    # Force fixed normalization so constant arrays (all 0 or all 1) render correctly.
    im = ax.imshow(
        coverage_float,
        cmap=cmap,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
        vmin=0,
        vmax=1,
        zorder=2,
    )
    
    # Mark radar position if provided
    if radar_lat is not None and radar_lon is not None:
        ax.plot(radar_lon, radar_lat, "k*", markersize=20, label="Radar", zorder=10, markeredgewidth=2, markeredgecolor="white")

    # Mark airport position (default: Nice)
    if show_airport and airport_lat is not None and airport_lon is not None:
        ax.plot(airport_lon, airport_lat, marker="o", color="#1f77b4", markersize=10, label="Airport", zorder=11,
                markeredgewidth=1.5, markeredgecolor="white")
        ax.text(
            airport_lon,
            airport_lat,
            f"  {airport_label}",
            fontsize=11,
            va="bottom",
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
    
    # Calculate and display statistics
    coverage_pct = np.sum(coverage_map) / coverage_map.size * 100
    stats_text = f'Coverage: {coverage_pct:.1f}% visible'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Radar Coverage - FL{flight_level}', fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar (simple 2-class legend)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(["Blocked", "Visible"])
    cbar.set_label("Coverage Status", fontsize=11)
    
    # Add legend if radar is marked
    if (radar_lat is not None and radar_lon is not None) or show_airport:
        ax.legend(loc="upper right", fontsize=10)
    
    # Gridlines reduce realism on basemaps; keep them only when no basemap is drawn.
    if not drew_basemap:
        ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
    else:
        ax.grid(False)
    
    plt.tight_layout()
    
    # Save if path provided, otherwise show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        if not return_fig:
            plt.close(fig)
            return None

    if return_fig:
        return fig

    else:
        plt.show()  # This blocks until window is closed
        plt.close(fig)
        return None


def plot_all_coverage_maps(
    coverage_maps: Dict[float, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    radar_lat: Optional[float] = None,
    radar_lon: Optional[float] = None,
    terrain: Optional[np.ndarray] = None,
    visible_alpha: float = 0.45,
    blocked_alpha: float = 0.22,
    visible_color: tuple = (0.00, 0.70, 0.00),
    blocked_color: tuple = (0.85, 0.05, 0.05),
    basemap: bool = True,
    basemap_provider: str = "CartoDB.VoyagerNoLabels",
    basemap_zoom: Optional[int] = None,
    basemap_warn: bool = True,
    background: str = "basemap+relief",
    relief_alpha: float = 0.70,
    airport_lat: float = 43.6584,
    airport_lon: float = 7.2159,
    airport_label: str = "Nice Airport (LFMN)",
    show_airport: bool = True,
    return_fig: bool = False,
) -> Optional["plt.Figure"]:
    """
    Plot all coverage maps in a grid layout (2 rows x 4 columns).
    
    Parameters:
    -----------
    coverage_maps : Dict[float, np.ndarray]
        Dictionary mapping flight level to coverage map
    lats : np.ndarray
        1D array of latitude values
    lons : np.ndarray
        1D array of longitude values
    radar_lat : float, optional
        Radar latitude to mark on maps
    radar_lon : float, optional
        Radar longitude to mark on maps
    """
    flight_levels = sorted(coverage_maps.keys())
    n_maps = len(flight_levels)
    
    # Create subplot grid: 2 rows x 4 columns for 8 maps
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Convert boolean to float for visualization
    coverage_float_base = coverage_maps[flight_levels[0]].astype(float)
    
    # Create custom colormap
    va = float(np.clip(visible_alpha, 0.0, 1.0))
    ba = float(np.clip(blocked_alpha, 0.0, 1.0))
    colors = [
        (float(blocked_color[0]), float(blocked_color[1]), float(blocked_color[2]), ba),
        (float(visible_color[0]), float(visible_color[1]), float(visible_color[2]), va),
    ]
    cmap = ListedColormap(colors)
    
    # Set extent for all maps
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    # Background (compute once)
    bg = (background or "").lower().strip()
    hs = None
    rel = None
    if terrain is not None:
        if "relief" in bg:
            rel = _colored_relief(terrain, cmap_name="terrain")
        elif "hillshade" in bg:
            hs = _hillshade(terrain)

    basemap_source = _get_basemap_source(basemap_provider) if (basemap and ("basemap" in bg)) else None
    
    for idx, fl in enumerate(flight_levels):
        ax = axes[idx]
        coverage_float = coverage_maps[fl].astype(float)

        drew_basemap = False
        if basemap_source is not None:
            drew_basemap = _add_basemap_latlon(
                ax,
                extent,
                basemap_source,
                basemap_zoom=basemap_zoom,
                warn=basemap_warn,
            )

        if rel is not None:
            ax.imshow(
                rel,
                aspect="auto",
                origin="lower",
                extent=extent,
                interpolation="bilinear",
                alpha=float(np.clip(relief_alpha, 0.0, 1.0)),
                zorder=1,
            )
        elif hs is not None:
            ax.imshow(
                hs,
                cmap="Greys",
                aspect="auto",
                origin="lower",
                extent=extent,
                interpolation="bilinear",
                vmin=0,
                vmax=1,
                alpha=0.95,
                zorder=1,
            )
        
        # Use imshow for simple 2D map
        im = ax.imshow(
            coverage_float,
            cmap=cmap,
            aspect='auto',
            origin='lower',
            extent=extent,
            interpolation='nearest',
            vmin=0,
            vmax=1,
            zorder=2,
        )
        
        # Mark radar position if provided
        if radar_lat is not None and radar_lon is not None:
            ax.plot(radar_lon, radar_lat, 'k*', markersize=12, label='Radar', 
                   zorder=10, markeredgewidth=1.5, markeredgecolor='white')

        # Mark airport position (default: Nice)
        if show_airport and airport_lat is not None and airport_lon is not None:
            ax.plot(
                airport_lon,
                airport_lat,
                marker="o",
                color="#1f77b4",
                markersize=6,
                label="Airport",
                zorder=11,
                markeredgewidth=1.0,
                markeredgecolor="white",
            )
        
        # Calculate and display statistics
        coverage_pct = np.sum(coverage_maps[fl]) / coverage_maps[fl].size * 100
        stats_text = f'Coverage: {coverage_pct:.1f}% visible'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Set labels and title
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.set_title(f'Radar Coverage - FL{fl}', fontsize=12, fontweight='bold')
        if drew_basemap:
            ax.grid(False)
        else:
            ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
        
        # Add legend for first subplot only
        if idx == 0 and ((radar_lat is not None and radar_lon is not None) or show_airport):
            ax.legend(loc="upper right", fontsize=9)
    
    # Add single colorbar for all subplots
    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(["Blocked", "Visible"])
    cbar.set_label("Coverage Status", fontsize=11)
    
    plt.suptitle('Radar Coverage Maps - All Flight Levels', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])
    
    if return_fig:
        return fig

    plt.show()
    plt.close(fig)
    return None