"""
Main Coverage (NUMBA ENU)

This script is the NUMBA-based Lot 1 driver to compute full-grid coverage maps for
all tender flight levels from a single radar location, using ENU (East-North-Up) 
coordinate system for better accuracy.

Pipeline:
1) Load terrain grid from terrain_mat.npz
2) Convert lat/lon grid to ENU meters
3) Normalize grid orientation + make arrays contiguous (Numba-friendly)
4) For each FL: compute full-grid LOS coverage via LOS_numba_enu.coverage_map_numba_xy
5) Visualize (matplotlib) and/or export KMZ
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from LOS_numba_enu import (
    coverage_map_numba_xy, 
    fl_to_m, 
    normalize_xy_grid,
    latlon_to_xy_m
)
from visualize_coverage import plot_all_coverage_maps, plot_coverage_map
from export_kml import export_all_coverage_to_kmz


def load_terrain_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lightweight loader for terrain_mat.npz.

    Expected NPZ keys:
      - lat: (N,) latitude array
      - lon: (M,) longitude array
      - ter: (N,M) terrain elevation (meters MSL)
    """
    d = np.load(npz_path)
    lats = d["lat"].astype(float)
    lons = d["lon"].astype(float)
    Z = d["ter"].astype(float)

    if Z.shape != (len(lats), len(lons)):
        raise ValueError(f"Incohérence: Z{Z.shape} vs ({len(lats)}, {len(lons)})")

    return lats, lons, Z


def convert_terrain_to_enu(
    lats: np.ndarray,
    lons: np.ndarray,
    Z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert lat/lon terrain grid to ENU (East-North-Up) meters.
    
    Returns:
        X_m: 1D array of X coordinates (east) in meters, len M
        Y_m: 1D array of Y coordinates (north) in meters, len N
        Z: 2D terrain elevation array (N, M) - unchanged
    """
    # Convert longitude axis to X (east) in meters
    X_m = np.zeros(len(lons), dtype=float)
    for j, lon in enumerate(lons):
        x, _ = latlon_to_xy_m(lats[0], lon)  # Use first lat for reference
        X_m[j] = x
    
    # Convert latitude axis to Y (north) in meters
    Y_m = np.zeros(len(lats), dtype=float)
    for i, lat in enumerate(lats):
        _, y = latlon_to_xy_m(lat, lons[0])  # Use first lon for reference
        Y_m[i] = y
    
    return X_m, Y_m, Z


def compute_all_fls_numba_fullgrid(
    radar_lat: float,
    radar_lon: float,
    radar_height_agl_m: float,
    flight_levels: List[float],
    lats: np.ndarray,
    lons: np.ndarray,
    Z: np.ndarray,
    n_samples: int = 400,
    margin_m: float = 0.0,
) -> Dict[float, np.ndarray]:
    """
    Compute full-grid coverage maps for all FLs using LOS_numba_enu (ENU coordinates).

    Returns:
      Dict[float, np.ndarray]: {FL: coverage_map_bool} where coverage_map has shape (N,M).
    """
    # Convert terrain grid to ENU coordinates
    print("   Converting to ENU coordinates...")
    X_m, Y_m, Z_enu = convert_terrain_to_enu(lats, lons, Z)
    
    # Normalize grid (ensure increasing order)
    X_m, Y_m, Z_enu = normalize_xy_grid(X_m, Y_m, Z_enu)
    
    # Convert radar position to ENU
    radar_x, radar_y = latlon_to_xy_m(radar_lat, radar_lon)
    
    # Get grid parameters
    if X_m.size < 2 or Y_m.size < 2:
        raise ValueError("Terrain grid too small: need at least 2x2 points.")

    x0 = float(X_m[0])
    y0 = float(Y_m[0])
    dx = float(X_m[1] - X_m[0])
    dy = float(Y_m[1] - Y_m[0])

    if dx == 0.0 or dy == 0.0:
        raise ValueError("Invalid grid step: dx/dy is zero.")

    print(f"   Grid in ENU: X=[{X_m[0]:.1f}, {X_m[-1]:.1f}]m, Y=[{Y_m[0]:.1f}, {Y_m[-1]:.1f}]m")
    print(f"   Radar ENU: x={radar_x:.1f}m, y={radar_y:.1f}m")

    coverage_maps: Dict[float, np.ndarray] = {}

    for idx, fl in enumerate([float(x) for x in flight_levels], start=1):
        target_alt_msl = float(fl_to_m(fl))
        print(f"→ FL{int(fl):3d} ({idx}/{len(flight_levels)}) ...", end="", flush=True)

        cov = coverage_map_numba_xy(
            float(radar_x), float(radar_y), float(radar_height_agl_m),
            float(target_alt_msl),
            X_m, Y_m,
            float(x0), float(y0), float(dx), float(dy), Z_enu,
            int(n_samples), float(margin_m),
        )

        visible = int(cov.sum())
        total = int(cov.size)
        pct = 100.0 * visible / total if total else 0.0
        print(f" ✓ {pct:6.2f}% visible")

        coverage_maps[fl] = cov

    return coverage_maps


def main() -> None:
    # ---------------- Configuration ----------------
    terrain_file = "terrain_mat.npz"
    kmz_output = "radar_coverage_numba_enu.kmz"

    radar_lat = 43.66375
    radar_lon = 7.07868
    radar_height_agl_m = 20.0

    flight_levels = [5, 10, 20, 50, 100, 200, 300, 400]

    # LOS sampling parameters
    n_samples = 200
    margin_m = 0.0

    # Output toggles (non-interactive)
    SHOW_PLOTS = True
    SHOW_PLOTS_FL_BY_FL = False
    EXPORT_KMZ = False

    print("=" * 70)
    print("NUMBA FULL GRID COVERAGE — ALL FLs (ENU COORDINATES)")
    print("=" * 70)

    # ---------------- Load terrain ----------------
    print("\n1) Loading terrain...")
    lats_full, lons_full, Z_full = load_terrain_npz(terrain_file)

    # Always use full grid (Numba provides the acceleration)
    lats, lons, Z = lats_full, lons_full, Z_full

    print(f"   Grid: {len(lats)} x {len(lons)} = {len(lats) * len(lons):,} points")

    # ---------------- Compute coverage maps (Numba ENU) ----------------
    print("\n2) Computing coverage maps (Numba + parallel + ENU)...")
    print("   (First run compiles Numba and can be slower.)")

    coverage_maps = compute_all_fls_numba_fullgrid(
        radar_lat=radar_lat,
        radar_lon=radar_lon,
        radar_height_agl_m=radar_height_agl_m,
        flight_levels=flight_levels,
        lats=lats,
        lons=lons,
        Z=Z,
        n_samples=n_samples,
        margin_m=margin_m,
    )

    # ---------------- Visualize ----------------
    if SHOW_PLOTS:
        print("\n3) Plotting all FL maps (BASEMAP + RELIEF)...")
        plot_all_coverage_maps(
            coverage_maps,
            lats,
            lons,
            radar_lat,
            radar_lon,
            terrain=Z,
            basemap=True,
            basemap_provider="CartoDB.VoyagerNoLabels",
            basemap_zoom=None,
            visible_alpha=0.45,
            blocked_alpha=0.22,
            visible_color=(0.0, 0.70, 0.0),
            blocked_color=(0.85, 0.05, 0.05),
            background="basemap+relief",
            relief_alpha=0.70,
            airport_lat=43.6584,
            airport_lon=7.2159,
            airport_label="Nice Airport (LFMN)",
            show_airport=True,
        )

    # Visualize FL by FL (one window per flight level)
    if SHOW_PLOTS_FL_BY_FL:
        print("\n3b) Plotting maps FL by FL...")
        for fl in sorted(coverage_maps.keys()):
            plot_coverage_map(
                coverage_maps[fl],
                lats,
                lons,
                float(fl),
                radar_lat=radar_lat,
                radar_lon=radar_lon,
                terrain=Z,
                basemap=True,
                basemap_provider="CartoDB.VoyagerNoLabels",
                basemap_zoom=None,
                visible_alpha=0.45,
                blocked_alpha=0.22,
                visible_color=(0.0, 0.70, 0.0),
                blocked_color=(0.85, 0.05, 0.05),
                background="basemap+relief",
                relief_alpha=0.70,
                airport_lat=43.6584,
                airport_lon=7.2159,
                airport_label="Nice Airport (LFMN)",
                show_airport=True,
                save_path=None,
            )

    # ---------------- Export KMZ ----------------
    if EXPORT_KMZ:
        print("\n4) Exporting KMZ...")
        export_all_coverage_to_kmz(
            coverage_maps,
            lats,
            lons,
            radar_lat=radar_lat,
            radar_lon=radar_lon,
            output_path=kmz_output,
        )

    print("\nDone ✓")


if __name__ == "__main__":
    main()