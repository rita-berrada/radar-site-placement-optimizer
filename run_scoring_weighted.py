#!/usr/bin/env python3
import numpy as np

# Use the WEIGHTED scoring function
from score_weighted import score_one_radar


def main():
    # -----------------------------
    # 1) Load authorized candidates
    # -----------------------------
    data = np.load("authorized_points_all_masks.npz")
    points = data["points"]  # shape (N, 2) with [lat, lon]
    n_points = len(points)

    print(f"✓ Loaded {n_points:,} authorized candidate points from authorized_points_01deg.npz")

    # -----------------------------------
    # 2) Scoring parameters (tune as needed)
    # -----------------------------------
    radar_height_agl_m = 20.0
    flight_levels = (5, 10, 20, 50, 100, 200, 300, 400)

    # Speed/quality trade-off:
    # - target_n controls downsampling resolution inside score_weighted.py
    # - n_samples controls how many LOS samples are evaluated
    target_n = 80
    n_samples = 200
    margin_m = 0.0

    # Optional: score only first K points for a quick test
    # K = min(n_points, 50)
    K = n_points

    # -----------------------------------
    # 3) Score each candidate
    # -----------------------------------
    results = []  # list of (weighted_score, lat, lon, cov_by_fl)

    for k in range(K):
        lat, lon = points[k]

        weighted_score, cov_by_fl = score_one_radar(
            radar_lat=float(lat),
            radar_lon=float(lon),
            radar_height_agl_m=radar_height_agl_m,
            flight_levels=flight_levels,
            terrain_npz_path="terrain_mat.npz",
            target_n=target_n,
            n_samples=n_samples,
            margin_m=margin_m,
        )

        results.append((float(weighted_score), float(lat), float(lon), cov_by_fl))
        print(f"[{k+1}/{K}] lat={lat:.1f}, lon={lon:.1f} -> weighted_score={weighted_score:.4f}")

    # -----------------------------------
    # 4) Sort and display Top 10
    # -----------------------------------
    results.sort(key=lambda x: x[0], reverse=True)

    print("\n=== TOP 10 CANDIDATES (WEIGHTED) ===")
    for rank, (s, lat, lon, cov) in enumerate(results[:10], start=1):
        print(f"{rank:>2}. lat={lat:.1f}, lon={lon:.1f} | weighted_score={s:.4f} | cov_by_FL={cov}")

    # -----------------------------------
    # 5) (Optional) Save results
    # -----------------------------------
    # Save score, lat, lon; cov_by_fl is kept as object array for simplicity
    scores_arr = np.array([(s, lat, lon) for (s, lat, lon, _) in results], dtype=float)
    cov_arr = np.array([cov for (_, _, _, cov) in results], dtype=object)

    np.savez("scored_candidates_weighted.npz", scores_latlon=scores_arr, cov_by_fl=cov_arr)
    print("\n✓ Saved results to scored_candidates_weighted.npz")


if __name__ == "__main__":
    main()
