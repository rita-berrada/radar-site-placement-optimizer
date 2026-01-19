"""
Multi-FL Coverage Script (FULL GRID) — numpy style (no explicit double loop)

- Uses LOS_np.los_visible (faster LOS)
- Computes coverage on FULL DTED grid (no crop)
- No explicit nested for-loops: numpy meshgrid + fromiter + reshape
- Displays ALL FL maps on one figure with coverage %

Requires:
- terrain_mat.npz
- LOS_np.py providing los_visible(...)
- visualize_terrain.load_terrain_npz
- visualize_coverage.plot_all_coverage_maps
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from visualize_terrain import load_terrain_npz
from visualize_coverage import plot_all_coverage_maps
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


def run_all_fls_full_grid():
    print("=" * 70)
    print("MULTI-FL Coverage (FULL GRID) — numpy style (no nested loops)")
    print("=" * 70)

    # ---------------- Configuration ----------------
    terrain_file = "terrain_mat.npz"

    radar_lat = 43.6584
    radar_lon = 7.2159
    radar_height_agl_m = 50.0

    flight_levels = [5, 10, 20, 50, 100, 200, 300, 400]

    n_samples = 200
    margin_m = 0.0

    # ---------------- Load terrain ----------------
    print("\n1) Loading terrain...")
    lats, lons, Z = load_terrain_npz(terrain_file)
    print(f"   ✓ Grid: {len(lats)} x {len(lons)} = {len(lats)*len(lons):,} points")

    coverage_maps = {}

    for idx, fl in enumerate(flight_levels, start=1):
        print(f"\n2) Computing coverage for FL{int(fl)} ({idx}/{len(flight_levels)})...")

        def progress(k, total, pct):
            print(f"   Progress FL{int(fl)}: {k:,}/{total:,} ({pct:.1f}%)", end="\r")

        cmap = compute_coverage_full_grid_np(
            radar_lat, radar_lon, radar_height_agl_m,
            float(fl),
            lats, lons, Z,
            n_samples=n_samples,
            margin_m=margin_m,
            point_progress_callback=progress
        )
        print()  # newline after progress

        visible = int(np.sum(cmap))
        total = int(cmap.size)
        pct = 100.0 * visible / total
        print(f"   ✓ FL{int(fl)} coverage: {visible:,}/{total:,} visible ({pct:.2f}%)")

        coverage_maps[float(fl)] = cmap

    # ---------------- Plot all on one figure ----------------
    print("\n3) Plotting all FL maps on one figure...")
    plot_all_coverage_maps(
        coverage_maps, lats, lons, radar_lat, radar_lon
    )

    print("\nDone ✓")
    return True


if __name__ == "__main__":
    ok = run_all_fls_full_grid()
    exit(0 if ok else 1)
