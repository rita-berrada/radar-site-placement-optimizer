# score.py
import numpy as np
from typing import Dict, List, Tuple

from visualize_terrain import load_terrain_npz
from LOS_numba import normalize_grid, fl_to_m, coverage_map_numba


def weights_for_fls(flight_levels: List[float]) -> Dict[float, float]:
    """
    Weight rule you requested:
    FL5/FL10/FL20 have coefficient 2, others coefficient 1.
    """
    w = {}
    for fl in flight_levels:
        fl = float(fl)
        if int(fl) in (5, 10, 20):
            w[fl] = 2.0
        else:
            w[fl] = 1.0
    return w


def score_from_cov(cov_pct: Dict[float, float], flight_levels: List[float]) -> float:
    """
    Weighted average of coverage percentages.
    """
    W = weights_for_fls(flight_levels)
    num = 0.0
    den = 0.0
    for fl in flight_levels:
        fl = float(fl)
        if fl not in cov_pct:
            continue
        num += W[fl] * float(cov_pct[fl])
        den += W[fl]
    return float(num / den) if den > 0 else 0.0


def coverage_pct_one_fl_numba_fullgrid(
    radar_lat: float,
    radar_lon: float,
    radar_height_agl_m: float,
    flight_level: float,
    lats: np.ndarray,
    lons: np.ndarray,
    Z: np.ndarray,
    lats0: float,
    lons0: float,
    dlat: float,
    dlon: float,
    n_samples: int = 400,
    margin_m: float = 0.0
) -> float:
    """
    FULL GRID coverage % for ONE FL using Numba.
    """
    target_alt = float(fl_to_m(float(flight_level)))

    cov_map = coverage_map_numba(
        float(radar_lat), float(radar_lon), float(radar_height_agl_m),
        float(target_alt),
        lats, lons,
        float(lats0), float(lons0), float(dlat), float(dlon), Z,
        int(n_samples), float(margin_m)
    )

    return float(cov_map.mean() * 100.0)


def score_one_candidate_fullgrid_numba(
    radar_lat: float,
    radar_lon: float,
    radar_height_agl_m: float,
    flight_levels: List[float],
    lats: np.ndarray,
    lons: np.ndarray,
    Z: np.ndarray,
    lats0: float,
    lons0: float,
    dlat: float,
    dlon: float,
    n_samples: int = 400,
    margin_m: float = 0.0,
    verbose: bool = True
) -> Tuple[float, Dict[float, float]]:
    """
    Returns:
      score (float),
      cov_pct (dict {FL: coverage_pct})
    """
    cov_pct: Dict[float, float] = {}

    for fl in flight_levels:
        fl = float(fl)
        if verbose:
            print(f"    → FL{int(fl)} ...", end="", flush=True)

        pct = coverage_pct_one_fl_numba_fullgrid(
            radar_lat, radar_lon, radar_height_agl_m,
            fl,
            lats, lons, Z,
            lats0, lons0, dlat, dlon,
            n_samples=n_samples,
            margin_m=margin_m
        )
        cov_pct[fl] = pct

        if verbose:
            print(f" {pct:.2f}%")

    score = score_from_cov(cov_pct, flight_levels)
    return float(score), cov_pct


def rank_candidates_fullgrid_numba(
    candidates: List[Tuple[float, float, float]],  # [(lat, lon, h_agl), ...]
    flight_levels: List[float],
    terrain_npz_path: str = "terrain_mat.npz",
    n_samples: int = 400,
    margin_m: float = 0.0,
    verbose: bool = True
) -> List[dict]:
    """
    Scores and ranks candidates by score (descending).
    Loads terrain ONCE.
    """
    # ---- Load + normalize terrain ONCE ----
    lats, lons, Z = load_terrain_npz(terrain_npz_path)
    lats, lons, Z = normalize_grid(lats, lons, Z)

    lats0 = float(lats[0])
    lons0 = float(lons[0])
    dlat = float(lats[1] - lats[0])
    dlon = float(lons[1] - lons[0])

    results: List[dict] = []
    n_cand = len(candidates)

    for idx, (lat, lon, h) in enumerate(candidates, start=1):
        if verbose:
            print("\n" + "=" * 70)
            print(f"Candidate {idx}/{n_cand}  lat={lat:.6f} lon={lon:.6f} h={h:.1f}m")
            print("=" * 70)

        s, cov = score_one_candidate_fullgrid_numba(
            float(lat), float(lon), float(h),
            flight_levels,
            lats, lons, Z,
            lats0, lons0, dlat, dlon,
            n_samples=n_samples,
            margin_m=margin_m,
            verbose=verbose
        )

        if verbose:
            print(f"  => SCORE = {s:.2f}%")

        results.append({
            "lat": float(lat),
            "lon": float(lon),
            "h": float(h),
            "score": float(s),
            "cov_pct": cov
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# -------------------------
# Quick test
# -------------------------
if __name__ == "__main__":
    flight_levels = [5, 10, 20, 50, 100, 200, 300, 400]

    # example candidates (replace with your authorized_points masks output)
    candidates = [
        (43.6584, 7.2159, 20.0),
        (43.7000, 7.2500, 20.0),
    ]

    ranked = rank_candidates_fullgrid_numba(
        candidates=candidates,
        flight_levels=flight_levels,
        terrain_npz_path="terrain_mat.npz",
        n_samples=200,     # raise to 400 for final
        margin_m=0.0,
        verbose=True
    )

    print("\n\nTOP 5:")
    for i, r in enumerate(ranked[:5], start=1):
        print(f"#{i}  score={r['score']:.2f}%  lat={r['lat']:.5f} lon={r['lon']:.5f}")
