#!/usr/bin/env python3
"""
run_scoring_numba_enu.py (NUMBA + ENU)

Scores candidate sites using FULL GRID method with:
- ENU metric coordinates (X east, Y north)
- curvature-corrected terrain (Z_corrected)
- exact LOS logic + Numba parallel

Input:
- authorized_points_all_masks.npz  (lat, lon)

Output:
- scored_candidates_fullgrid_enu.npz   (lat, lon, score, cov_by_fl)
"""

import numpy as np

# IMPORTANT: import the ENU ranking function
from score_numba_enu import rank_candidates_fullgrid_numba_enu


def main():
    # -----------------------------
    # 1) Configuration
    # -----------------------------
    input_file = "authorized_points_all_masks.npz"
    output_file = "scored_candidates_fullgrid_enu.npz"
    terrain_file = "terrain_mat.npz"

    radar_height_agl_m = 20.0

    flight_levels = [5, 10, 20, 50, 100, 200, 300, 400]

    n_samples = 100
    margin_m = 0.0

    # -----------------------------
    # 2) Load authorized candidates
    # -----------------------------
    print(f"--- Loading candidates from {input_file} ---")
    try:
        data = np.load(input_file)
        lats = data["lat"]
        lons = data["lon"]
        n_points = len(lats)
        print(f"✓ Found {n_points:,} authorized candidates.")
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Run the masks script first.")
        return

    # Optional limit for testing
    # K = 10
    K = n_points

    candidates_list = [(float(lats[k]), float(lons[k]), float(radar_height_agl_m)) for k in range(K)]

    # -----------------------------
    # 3) Score candidates (Numba ENU)
    # -----------------------------
    print(f"\n--- Scoring {K} candidates (NUMBA ENU Full Grid) ---")
    print("Note: First run may be slower (Numba compilation).")

    ranked_results = rank_candidates_fullgrid_numba_enu(
        candidates=candidates_list,
        flight_levels=flight_levels,
        terrain_npz_path=terrain_file,
        n_samples=n_samples,
        margin_m=margin_m,
        show_progress=True   # prints per-candidate + per-FL %
    )

    # -----------------------------
    # 4) Display top 10
    # -----------------------------
    print("\n" + "=" * 60)
    print("TOP 10 CANDIDATES (ENU)")
    print("=" * 60)
    for i, r in enumerate(ranked_results[:10], start=1):
        print(f"#{i:<2} Score: {r['score']:6.2f}% | Lat: {r['lat']:.5f}, Lon: {r['lon']:.5f}")

    # -----------------------------
    # 5) Save results
    # -----------------------------
    final_lats = np.array([r["lat"] for r in ranked_results], dtype=float)
    final_lons = np.array([r["lon"] for r in ranked_results], dtype=float)
    final_scores = np.array([r["score"] for r in ranked_results], dtype=float)
    final_covs = np.array([r["cov_pct"] for r in ranked_results], dtype=object)

    np.savez(
        output_file,
        lat=final_lats,
        lon=final_lons,
        score=final_scores,
        cov_by_fl=final_covs
    )
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
