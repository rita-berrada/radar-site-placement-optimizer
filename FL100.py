"""
FL10 Coverage Script (FULL GRID) — No explicit double loop
Uses LOS_np.los_visible (faster LOS) and computes coverage on the full DTED grid.

- No crop
- No fast coverage module
- No explicit nested for-loops: uses numpy meshgrid + fromiter + reshape
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from visualize_terrain import load_terrain_npz
from visualize_coverage import plot_coverage_map
from LOS_np import los_visible
from LOS import fl_to_m


def compute_coverage_full_grid_np(
    radar_lat: float,
    radar_lon: float,
    radar_height_agl_m: float,
    flight_level: float,
    lats: np.ndarray,
    lons: np.ndarray,
    Z: np.ndarray,
    n_samples: int = 400,
    margin_m: float = 0.0,
    point_progress_callback=None
) -> np.ndarray:
    """
    FULL GRID coverage, without explicit double for-loop.
    coverage_map[i,j] corresponds to (lats[i], lons[j]).
    """

    target_alt_m_msl = float(fl_to_m(flight_level))

    # Build full grid of target points
    lon_grid, lat_grid = np.meshgrid(lons, lats)   # shape (N,M)
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()

    total = lat_flat.size
    report_every = max(1, total // 50)

    def gen():
        for k, (lat, lon) in enumerate(zip(lat_flat, lon_flat), start=1):
            v = los_visible(
                radar_lat, radar_lon, radar_height_agl_m,
                float(lat), float(lon), target_alt_m_msl,
                lats, lons, Z,
                n_samples=n_samples,
                margin_m=margin_m
            )

            if point_progress_callback and (k % report_every == 0 or k == total):
                pct = 100.0 * k / total
                point_progress_callback(k, total, pct)

            yield v

    cov_flat = np.fromiter(gen(), dtype=np.bool_, count=total)
    return cov_flat.reshape(len(lats), len(lons))


def run_fl10_full_grid():
    print("=" * 70)
    print("FL10 Coverage (FULL GRID) — numpy style (no nested loops)")
    print("=" * 70)

    # ---------------- Configuration ----------------
    terrain_file = "terrain_mat.npz"

    radar_lat = 43.6584
    radar_lon = 7.2159
    radar_height_agl_m = 50.0

    flight_level = 10.0
    n_samples = 800
    margin_m = 0.0

    # For display only (does NOT change computation)
    DISPLAY_STEP = 1  # keep 1 if you want full resolution plot

    # ---------------- Load terrain ----------------
    print("\n1) Loading terrain...")
    lats, lons, Z = load_terrain_npz(terrain_file)
    print(f"   ✓ Grid: {len(lats)} x {len(lons)} = {len(lats)*len(lons):,} points")

    # ---------------- Progress callback ----------------
    def progress(k, total, pct):
        print(f"Progress: {k:,}/{total:,} ({pct:.1f}%)", end="\r")

    # ---------------- Compute coverage ----------------
    print(f"\n2) Computing coverage for FL{int(flight_level)} (FULL GRID)...")
    coverage_map = compute_coverage_full_grid_np(
        radar_lat, radar_lon, radar_height_agl_m,
        flight_level,
        lats, lons, Z,
        n_samples=n_samples,
        margin_m=margin_m,
        point_progress_callback=progress
    )
    print()  # newline after progress

    visible = int(np.sum(coverage_map))
    total = int(coverage_map.size)
    print(f"   ✓ Coverage: {visible:,}/{total:,} visible ({100.0*visible/total:.2f}%)")

    # ---------------- Plot ----------------
    print("\n3) Plotting...")
    if DISPLAY_STEP == 1:
        plot_coverage_map(
            coverage_map, lats, lons,
            float(flight_level),
            radar_lat=radar_lat, radar_lon=radar_lon,
            save_path=None
        )
    else:
        plot_coverage_map(
            coverage_map[::DISPLAY_STEP, ::DISPLAY_STEP],
            lats[::DISPLAY_STEP], lons[::DISPLAY_STEP],
            float(flight_level),
            radar_lat=radar_lat, radar_lon=radar_lon,
            save_path=None
        )

    return True


if __name__ == "__main__":
    ok = run_fl10_full_grid()
    exit(0 if ok else 1)
