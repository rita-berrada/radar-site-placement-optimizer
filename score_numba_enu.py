# score.py (ENU version)
import numpy as np
from typing import Dict, List, Tuple

from geo_utils import load_and_convert_to_enu
from LOS_numba_enu import latlon_to_xy_m, fl_to_m, coverage_map_numba_xy


def normalize_enu_and_geo(X_m, Y_m, Zcorr, lats, lons):
    """
    Make X_m and Y_m increasing, and apply the SAME flips to:
    - Zcorr
    - lats / lons (so plotting stays consistent if needed)
    Returns contiguous arrays (Numba-friendly).
    """
    X_m = np.asarray(X_m)
    Y_m = np.asarray(Y_m)
    Zcorr = np.asarray(Zcorr)
    lats = np.asarray(lats)
    lons = np.asarray(lons)

    # Y axis (north) corresponds to rows and to lats
    if Y_m[0] > Y_m[-1]:
        Y_m = Y_m[::-1].copy()
        lats = lats[::-1].copy()
        Zcorr = Zcorr[::-1, :].copy()

    # X axis (east) corresponds to cols and to lons
    if X_m[0] > X_m[-1]:
        X_m = X_m[::-1].copy()
        lons = lons[::-1].copy()
        Zcorr = Zcorr[:, ::-1].copy()

    return (np.ascontiguousarray(X_m),
            np.ascontiguousarray(Y_m),
            np.ascontiguousarray(Zcorr),
            np.ascontiguousarray(lats),
            np.ascontiguousarray(lons))


def weights_for_fls(flight_levels: List[float]) -> Dict[float, float]:
    """
    Weight rule: FL5/FL10/FL20 have coefficient 2, others coefficient 1.
    """
    w = {}
    for fl in flight_levels:
        fl = float(fl)
        w[fl] = 2.0 if int(fl) in (5, 10, 20) else 1.0
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


def coverage_pct_one_fl_numba_fullgrid_enu(
    radar_lat: float,
    radar_lon: float,
    radar_height_agl_m: float,
    flight_level: float,
    X_m: np.ndarray,
    Y_m: np.ndarray,
    Zcorr: np.ndarray,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    n_samples: int = 400,
    margin_m: float = 0.0,
    show_progress: bool = True,
    fl_index: int = 1,
    fl_total: int = 1
) -> float:
    """
    FULL GRID coverage % for ONE FL using ENU + curvature-corrected terrain.
    """
    # Convert radar to ENU
    radar_x, radar_y = latlon_to_xy_m(radar_lat, radar_lon)

    # Convert FL to target altitude (meters)
    target_alt = float(fl_to_m(float(flight_level)))

    if show_progress:
        print(f"    → FL{int(flight_level)} ({fl_index}/{fl_total}) ...", end="", flush=True)

    cov_map = coverage_map_numba_xy(
        float(radar_x), float(radar_y), float(radar_height_agl_m),
        float(target_alt),
        X_m, Y_m,
        float(x0), float(y0), float(dx), float(dy), Zcorr,
        int(n_samples), float(margin_m)
    )

    pct = float(cov_map.mean() * 100.0)

    if show_progress:
        print(f" {pct:.2f}%")

    return pct


def score_one_candidate_fullgrid_numba_enu(
    radar_lat: float,
    radar_lon: float,
    radar_height_agl_m: float,
    flight_levels: List[float],
    X_m: np.ndarray,
    Y_m: np.ndarray,
    Zcorr: np.ndarray,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    n_samples: int = 400,
    margin_m: float = 0.0,
    show_progress: bool = True
) -> Tuple[float, Dict[float, float]]:
    """
    Returns: (score, cov_pct_by_fl)
    """
    cov_pct: Dict[float, float] = {}
    fl_total = len(flight_levels)

    for i, fl in enumerate(flight_levels, start=1):
        fl = float(fl)
        cov_pct[fl] = coverage_pct_one_fl_numba_fullgrid_enu(
            radar_lat, radar_lon, radar_height_agl_m,
            fl,
            X_m, Y_m, Zcorr,
            x0, y0, dx, dy,
            n_samples=n_samples,
            margin_m=margin_m,
            show_progress=show_progress,
            fl_index=i,
            fl_total=fl_total
        )

    score = score_from_cov(cov_pct, flight_levels)
    return float(score), cov_pct


def rank_candidates_fullgrid_numba_enu(
    candidates: List[Tuple[float, float, float]],  # [(lat, lon, h_agl), ...]
    flight_levels: List[float],
    terrain_npz_path: str = "terrain_mat.npz",
    n_samples: int = 400,
    margin_m: float = 0.0,
    show_progress: bool = True
) -> List[dict]:
    """
    Scores and ranks candidates by score (descending).
    Loads ENU terrain ONCE.
    """
    # ---- Load once: ENU axes + curvature corrected Z ----
    X_m, Y_m, Zcorr, lats, lons = load_and_convert_to_enu(terrain_npz_path)
    X_m, Y_m, Zcorr, lats, lons = normalize_enu_and_geo(X_m, Y_m, Zcorr, lats, lons)

    # uniform ENU grid params
    x0 = float(X_m[0])
    y0 = float(Y_m[0])
    dx = float(X_m[1] - X_m[0])
    dy = float(Y_m[1] - Y_m[0])

    results: List[dict] = []
    n_cand = len(candidates)

    for idx, (lat, lon, h) in enumerate(candidates, start=1):
        if show_progress:
            print("\n" + "=" * 70)
            print(f"Candidate {idx}/{n_cand}  lat={lat:.6f} lon={lon:.6f} h={h:.1f}m")
            print("=" * 70)

        s, cov = score_one_candidate_fullgrid_numba_enu(
            float(lat), float(lon), float(h),
            flight_levels,
            X_m, Y_m, Zcorr,
            x0, y0, dx, dy,
            n_samples=n_samples,
            margin_m=margin_m,
            show_progress=show_progress
        )

        if show_progress:
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


# Quick test
if __name__ == "__main__":
    flight_levels = [5, 10, 20, 50, 100, 200, 300, 400]
    candidates = [
        (43.6584, 7.2159, 20.0),
        (43.7000, 7.2500, 20.0),
    ]

    ranked = rank_candidates_fullgrid_numba_enu(
        candidates=candidates,
        flight_levels=flight_levels,
        terrain_npz_path="terrain_mat.npz",
        n_samples=200,
        margin_m=0.0,
        show_progress=True
    )

    print("\nTOP 5:")
    for i, r in enumerate(ranked[:5], start=1):
        print(f"#{i}  score={r['score']:.2f}%  lat={r['lat']:.5f} lon={r['lon']:.5f}")
