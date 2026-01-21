"""
Main Coverage (NUMBA)

This script is the NUMBA-based Lot 1 driver to compute full-grid coverage maps for
all tender flight levels from a single radar location, optionally visualize them,
and export the results to KMZ for Google Earth.

Pipeline:
1) Load terrain grid from terrain_mat.npz
2) Normalize grid orientation + make arrays contiguous (Numba-friendly)
3) For each FL: compute full-grid LOS coverage via LOS_numba.coverage_map_numba
4) Visualize (matplotlib) and/or export KMZ
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from LOS_numba import coverage_map_numba, fl_to_m, normalize_grid
from visualize_coverage import plot_all_coverage_maps
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
    Compute full-grid coverage maps for all FLs using LOS_numba.

    Returns:
      Dict[float, np.ndarray]: {FL: coverage_map_bool} where coverage_map has shape (N,M).
    """
    # Numba kernel expects a uniform grid defined by origin and constant step
    if lats.size < 2 or lons.size < 2:
        raise ValueError("Terrain grid too small: need at least 2x2 points.")

    lats0 = float(lats[0])
    lons0 = float(lons[0])
    dlat = float(lats[1] - lats[0])
    dlon = float(lons[1] - lons[0])

    if dlat == 0.0 or dlon == 0.0:
        raise ValueError("Invalid grid step: dlat/dlon is zero.")

    coverage_maps: Dict[float, np.ndarray] = {}

    for idx, fl in enumerate([float(x) for x in flight_levels], start=1):
        target_alt_msl = float(fl_to_m(fl))
        print(f"→ FL{int(fl):3d} ({idx}/{len(flight_levels)}) ...", end="", flush=True)

        cov = coverage_map_numba(
            float(radar_lat), float(radar_lon), float(radar_height_agl_m),
            float(target_alt_msl),
            lats, lons,
            float(lats0), float(lons0), float(dlat), float(dlon), Z,
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
    kmz_output = "radar_coverage_numba.kmz"

    radar_lat = 43.66375
    radar_lon = 7.07868
    radar_height_agl_m = 20.0

    flight_levels = [5, 10, 20, 50, 100, 200, 300, 400]

    # LOS sampling parameters
    n_samples = 200
    margin_m = 0.0

    # Output toggles (non-interactive)
    SHOW_PLOTS = True
    EXPORT_KMZ = True

    print("=" * 70)
    print("NUMBA FULL GRID COVERAGE — ALL FLs")
    print("=" * 70)

    # ---------------- Load + normalize terrain ----------------
    print("\n1) Loading terrain...")
    lats_full, lons_full, Z_full = load_terrain_npz(terrain_file)

    # Always use full grid (Numba provides the acceleration)
    lats, lons, Z = lats_full, lons_full, Z_full

    lats, lons, Z = normalize_grid(lats, lons, Z)
    print(f"   Grid: {len(lats)} x {len(lons)} = {len(lats) * len(lons):,} points")

    # ---------------- Compute coverage maps (Numba) ----------------
    print("\n2) Computing coverage maps (Numba + parallel)...")
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
        print("\n3) Plotting all FL maps...")
        plot_all_coverage_maps(coverage_maps, lats, lons, radar_lat, radar_lon)

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
