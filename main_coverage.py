"""
Main Coverage (NUMBA ENU)

This script is the NUMBA-based driver to compute full-grid coverage maps for
all tender flight levels from a single radar location.

It uses:
1. geo_utils_earth_curvature: To load terrain with Earth curvature correction applied to Z.
2. LOS_numba_enu: To compute Line of Sight, applying the same curvature logic to the Target.
3. Now the FLs also take into account earth curvature
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

# --- Geometry Engine ---
# Loads terrain and converts to ENU metrics with Z corrected for Earth curvature
from geo_utils_earth_curvature import load_and_convert_to_enu

# --- Physics Engine ---
# Numba-accelerated LOS logic
from LOS_numba_enu import (
    coverage_map_numba_xy, 
    fl_to_m, 
    latlon_to_xy_m
)

# --- Visualization Modules ---
from visualize_coverage import plot_all_coverage_maps, plot_coverage_map
from export_kml import export_all_coverage_to_kmz


def normalize_all(X_m, Y_m, Z, lats, lons):
    """
    Ensure X_m and Y_m are strictly increasing (monotonic).
    Applies the necessary flips to Z, lats, and lons to maintain alignment.
    Returns contiguous arrays for Numba performance.
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


def compute_all_fls_numba_fullgrid(
    radar_lat: float,
    radar_lon: float,
    radar_height_agl_m: float,
    flight_levels: List[float],
    X_m: np.ndarray,
    Y_m: np.ndarray,
    Z_corrected: np.ndarray,
    n_samples: int = 400,
    margin_m: float = 0.0,
) -> Dict[float, np.ndarray]:
    """
    Compute full-grid coverage maps using the ENU metric system.
    Expects Z_corrected (Terrain with curvature drop applied).
    """
    
    # 1. Convert radar position to ENU
    radar_x, radar_y = latlon_to_xy_m(radar_lat, radar_lon)
    
    # 2. Get Grid Parameters
    x0 = float(X_m[0])
    y0 = float(Y_m[0])
    dx = float(X_m[1] - X_m[0])
    dy = float(Y_m[1] - Y_m[0])

    print(f"   Grid ENU Bounds: X=[{X_m[0]:.0f}, {X_m[-1]:.0f}]m, Y=[{Y_m[0]:.0f}, {Y_m[-1]:.0f}]m")
    print(f"   Radar Position:  x={radar_x:.0f}m, y={radar_y:.0f}m")

    coverage_maps: Dict[float, np.ndarray] = {}

    # 3. Loop over Flight Levels
    for idx, fl in enumerate([float(x) for x in flight_levels], start=1):
        # Convert FL to Meters MSL
        # Note: The curvature drop for the TARGET is handled inside LOS_numba_enu
        target_alt_msl = float(fl_to_m(fl))
        
        print(f"→ FL{int(fl):3d} ({idx}/{len(flight_levels)}) ...", end="", flush=True)

        # 4. Run Numba Engine
        cov = coverage_map_numba_xy(
            float(radar_x), float(radar_y), float(radar_height_agl_m),
            float(target_alt_msl),
            X_m, Y_m,
            float(x0), float(y0), float(dx), float(dy), Z_corrected,
            int(n_samples), float(margin_m),
        )

        # Stats
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

    # Radar coordinates (Example location)
    radar_lat = 43.66375
    radar_lon = 7.07868
    radar_height_agl_m = 20.0

    # Tender Flight Levels
    flight_levels = [5, 10, 20, 50, 100, 200, 300, 400]

    # Simulation parameters
    n_samples = 200
    margin_m = 0.0

    # Toggles
    SHOW_PLOTS = True
    SHOW_PLOTS_FL_BY_FL = False
    EXPORT_KMZ = False

    print("=" * 70)
    print("NUMBA FULL GRID COVERAGE — EARTH CURVATURE CORRECTED")
    print("=" * 70)

    # ---------------- 1. Load & Convert Terrain ----------------
    print("\n1) Loading terrain and applying Earth Curvature Correction...")
    
    # Load using the centralized loader
    # Z_corrected has the curvature drop applied.
    X_m, Y_m, Z_corrected, lats, lons = load_and_convert_to_enu(terrain_file)

    # Normalize axes (Ensure increasing X, Y)
    X_m, Y_m, Z_corrected, lats, lons = normalize_all(X_m, Y_m, Z_corrected, lats, lons)

    print(f"   Grid size: {len(Y_m)} rows x {len(X_m)} cols")

    # ---------------- 2. Compute Coverage ----------------
    print("\n2) Computing coverage maps (Numba + Physics Correction)...")
    
    coverage_maps = compute_all_fls_numba_fullgrid(
        radar_lat=radar_lat,
        radar_lon=radar_lon,
        radar_height_agl_m=radar_height_agl_m,
        flight_levels=flight_levels,
        X_m=X_m,
        Y_m=Y_m,
        Z_corrected=Z_corrected,
        n_samples=n_samples,
        margin_m=margin_m,
    )

    # ---------------- 3. Visualization ----------------
    # We pass Z_corrected to plotting so the relief matches the calculation.
    if SHOW_PLOTS:
        print("\n3) Plotting all FL maps...")
        plot_all_coverage_maps(
            coverage_maps,
            lats,
            lons,
            radar_lat,
            radar_lon,
            terrain=Z_corrected, 
            basemap=True,
            basemap_provider="CartoDB.VoyagerNoLabels",
            show_airport=True,
        )

    if SHOW_PLOTS_FL_BY_FL:
        print("\n3b) Plotting maps FL by FL...")
        for fl in sorted(coverage_maps.keys()):
            plot_coverage_map(
                coverage_maps[fl],
                lats, lons, float(fl),
                radar_lat=radar_lat, radar_lon=radar_lon,
                terrain=Z_corrected,
                basemap=True,
                show_airport=True
            )

    # ---------------- 4. Export ----------------
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