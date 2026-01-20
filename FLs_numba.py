#!/usr/bin/env python3
"""
fl_fullmap_numba.py

Compute FULL GRID coverage map for ONE FL using your existing LOS_numba.py
(exact LOS logic, numba + parallel).

Requirements:
- terrain_mat.npz
- LOS_numba.py (already written)
- visualize_terrain.load_terrain_npz
- visualize_coverage.plot_coverage_map
"""

import numpy as np
from visualize_terrain import load_terrain_npz
from visualize_coverage import plot_coverage_map

# Use your existing numba module
from LOS_numba import normalize_grid, fl_to_m, coverage_map_numba


def main():
    # ---------------- Configuration ----------------
    terrain_file = "terrain_mat.npz"

    radar_lat = 43.6584
    radar_lon = 7.2159
    radar_h_agl = 20.0

    flight_level = 10.0   # change to 100.0, etc.
    n_samples = 400
    margin_m = 0.0

    print("=" * 70)
    print(f"NUMBA FULL GRID COVERAGE — FL{int(flight_level)}")
    print("=" * 70)

    # ---------------- Load + normalize terrain ----------------
    print("\n1) Loading terrain...")
    lats, lons, Z = load_terrain_npz(terrain_file)
    lats, lons, Z = normalize_grid(lats, lons, Z)

    print(f"   Grid: {len(lats)} x {len(lons)} = {len(lats)*len(lons):,} points")

    # Precompute uniform-grid params expected by LOS_numba
    lats0 = float(lats[0])
    lons0 = float(lons[0])
    dlat = float(lats[1] - lats[0])
    dlon = float(lons[1] - lons[0])

    target_alt = float(fl_to_m(float(flight_level)))

    # ---------------- Compute FULL GRID coverage ----------------
    print("\n2) Computing coverage map (Numba + parallel)...")
    print("   (First run compiles: can be slower.)")

    cov = coverage_map_numba(
        float(radar_lat), float(radar_lon), float(radar_h_agl),
        target_alt,
        lats, lons,
        lats0, lons0, dlat, dlon, Z,
        int(n_samples), float(margin_m)
    )

    visible = int(cov.sum())
    total = int(cov.size)
    pct = 100.0 * visible / total
    print(f"   ✓ Coverage: {visible:,}/{total:,} visible ({pct:.2f}%)")

    # ---------------- Plot ----------------
    print("\n3) Plotting...")
    plot_coverage_map(
        cov,
        lats, lons,
        float(flight_level),
        radar_lat=float(radar_lat),
        radar_lon=float(radar_lon),
        save_path=None
    )

    print("\nDone ✓")


if __name__ == "__main__":
    main()
