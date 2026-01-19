"""
score.py — FULL GRID ranking using FLs_np.py (LOS_np pipeline)

- NO downsampling
- Uses compute_coverage_full_grid_np() from FLs_np.py
- Coverage% per FL
- Score: weighted average where FL5/FL10/FL20 have coefficient 2
- Rank candidates by score (descending)

No classes.
"""

import numpy as np
from typing import Dict, List, Tuple

from visualize_terrain import load_terrain_npz
from FLs_np import compute_coverage_full_grid_np


def coverage_rates_one_radar_full_grid(
    radar_lat: float,
    radar_lon: float,
    radar_height_agl_m: float,
    flight_levels: List[float],
    lats: np.ndarray,
    lons: np.ndarray,
    Z: np.ndarray,
    n_samples: int = 800,
    margin_m: float = 0.0
) -> Dict[float, float]:
    """
    FULL GRID coverage rates for one radar.
    Returns {FL: coverage_pct}.
    """
    cov_pct: Dict[float, float] = {}

    for fl in flight_levels:
        print(f"  → Computing FL{int(fl)}...", end="", flush=True)

        cmap = compute_coverage_full_grid_np(
            radar_lat, radar_lon, radar_height_agl_m,
            float(fl),
            lats, lons, Z,
            n_samples=n_samples,
            margin_m=margin_m,
            point_progress_callback=None
        )

        pct = float(cmap.mean() * 100.0)
        cov_pct[float(fl)] = pct
        print(f" ✓ {pct:.2f}%")

    return cov_pct


def score_from_coverages(
    flight_levels: List[float],
    cov_pct: Dict[float, float]
) -> float:
    """
    Weighted average:
    - First 3 FLs (in the given order) have weight 2
    - Others have weight 1
    """
    weights = np.ones(len(flight_levels), dtype=float)
    if len(weights) >= 3:
        weights[:3] = 2.0

    num = 0.0
    den = 0.0
    for w, fl in zip(weights, flight_levels):
        v = cov_pct.get(float(fl), None)
        if v is None:
            continue
        num += w * float(v)
        den += w

    return float(num / den) if den > 0 else 0.0


def score_one_radar_full_grid(
    radar_lat: float,
    radar_lon: float,
    radar_height_agl_m: float,
    flight_levels: List[float],
    terrain_npz_path: str = "terrain_mat.npz",
    n_samples: int = 800,
    margin_m: float = 0.0
) -> Tuple[float, Dict[float, float]]:
    """
    FULL GRID:
    returns (score, {FL: coverage_pct})
    """
    lats, lons, Z = load_terrain_npz(terrain_npz_path)

    cov_pct = coverage_rates_one_radar_full_grid(
        radar_lat, radar_lon, radar_height_agl_m,
        flight_levels,
        lats, lons, Z,
        n_samples=n_samples,
        margin_m=margin_m
    )

    score = score_from_coverages(flight_levels, cov_pct)
    return float(score), cov_pct


def rank_candidates_full_grid(
    candidates: List[Tuple[float, float, float]],   # [(lat, lon, h_agl_m), ...]
    flight_levels: List[float],
    terrain_npz_path: str = "terrain_mat.npz",
    n_samples: int = 800,
    margin_m: float = 0.0
) -> List[dict]:
    """
    Rank candidate sites by score (desc).
    Returns list of dicts.
    """
    results = []

    # load terrain once
    lats, lons, Z = load_terrain_npz(terrain_npz_path)

    for idx, (lat, lon, h) in enumerate(candidates, start=1):
        print("\n" + "=" * 60)
        print(f"Candidate {idx}/{len(candidates)} — lat={lat:.6f}, lon={lon:.6f}, h={h:.1f}m")
        print("=" * 60)

        cov_pct = coverage_rates_one_radar_full_grid(
            float(lat), float(lon), float(h),
            flight_levels,
            lats, lons, Z,
            n_samples=n_samples,
            margin_m=margin_m
        )
        score = score_from_coverages(flight_levels, cov_pct)

        results.append({
            "lat": float(lat),
            "lon": float(lon),
            "h": float(h),
            "score": float(score),
            "cov_pct": cov_pct
        })

        print(f"\n→ SCORE = {score:.2f}%")

    results.sort(key=lambda d: d["score"], reverse=True)
    return results


# =========================
# TEST
# =========================
if __name__ == "__main__":
    flight_levels = [5, 10, 20, 50, 100, 200, 300, 400]

    candidates = [
        (43.6584, 7.2159, 50.0),
        (43.7000, 7.2500, 50.0),
    ]

    ranked = rank_candidates_full_grid(
        candidates,
        flight_levels=flight_levels,
        terrain_npz_path="terrain_mat.npz",
        n_samples=800,
        margin_m=0.0
    )

    print("\n\n=== FINAL RANKING (best -> worst) ===")
    for i, r in enumerate(ranked, start=1):
        print(f"#{i} score={r['score']:.2f}%  lat={r['lat']:.6f} lon={r['lon']:.6f} h={r['h']:.1f}m")
