#!/usr/bin/env python3
"""
fl_fullmap_numba_enu.py

FULL GRID coverage for ONE FL using ENU meters + curvature-corrected terrain.

Uses:
- geo_utils.load_and_convert_to_enu  -> X_m, Y_m, Z_corrected, lats, lons
- LOS_numba_enu.coverage_map_numba_xy (numba + parallel, exact LOS early-exit)
- visualize_coverage.plot_coverage_map for display (still on lat/lon axes)
"""

import numpy as np
from geo_utils import load_and_convert_to_enu
from visualize_coverage import plot_coverage_map

from LOS_numba_enu import (
    normalize_xy_grid,
    latlon_to_xy_m,
    fl_to_m,
    coverage_map_numba_xy,
)


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
    print(f"NUMBA FULL GRID COVERAGE (ENU) — FL{int(flight_level)}")
    print("=" * 70)

    # ---------------- Load + convert to ENU + curvature correction ----------------
    print("\n1) Loading + converting terrain to ENU (with curvature correction)...")
    X_m, Y_m, Zcorr, lats, lons = load_and_convert_to_enu(terrain_file)
    X_m, Y_m, Zcorr = normalize_xy_grid(X_m, Y_m, Zcorr)

    print(f"   Grid: {len(Y_m)} x {len(X_m)} = {len(Y_m)*len(X_m):,} points")

    # Precompute uniform-grid params for ENU interpolation
    x0 = float(X_m[0])
    y0 = float(Y_m[0])
    dx = float(X_m[1] - X_m[0])
    dy = float(Y_m[1] - Y_m[0])

    # Radar in ENU meters
    radar_x, radar_y = latlon_to_xy_m(radar_lat, radar_lon)

    target_alt = float(fl_to_m(float(flight_level)))

    # ---------------- Compute FULL GRID coverage ----------------
    print("\n2) Computing coverage map (Numba + parallel, ENU)...")
    print("   (First run compiles: can be slower.)")

    cov = coverage_map_numba_xy(
        float(radar_x), float(radar_y), float(radar_h_agl),
        float(target_alt),
        X_m, Y_m,
        float(x0), float(y0), float(dx), float(dy), Zcorr,
        int(n_samples), float(margin_m)
    )

    visible = int(cov.sum())
    total = int(cov.size)
    pct = 100.0 * visible / total
    print(f"   ✓ Coverage: {visible:,}/{total:,} visible ({pct:.2f}%)")

    # ---------------- Plot on lat/lon axes ----------------
    # cov is indexed [i,j] = [Y_m index, X_m index]
    # lats/lons correspond to the same grid order as in geo_utils return
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
