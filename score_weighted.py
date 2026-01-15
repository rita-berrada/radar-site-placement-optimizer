import numpy as np

from LOS import load_terrain_npz
from coverage_analysis import compute_all_coverage_maps


def downsample_grid(lats, lons, Z, target_n=80):
    """
    Downsample the terrain grid to approximately target_n x target_n
    in order to speed up the coverage computation.
    """
    lat_step = max(1, len(lats) // target_n)
    lon_step = max(1, len(lons) // target_n)

    lats_s = lats[::lat_step][:target_n]
    lons_s = lons[::lon_step][:target_n]
    Z_s = Z[::lat_step, ::lon_step][:target_n, :target_n]

    # Safety check on shapes
    nlat = min(len(lats_s), Z_s.shape[0])
    nlon = min(len(lons_s), Z_s.shape[1])
    return lats_s[:nlat], lons_s[:nlon], Z_s[:nlat, :nlon]


def coverage_percent_by_fl(coverage_maps):
    """
    Convert boolean visibility maps into coverage percentages.

    Args:
        coverage_maps (dict):
            coverage_maps[FL] = boolean matrix (True = visible)

    Returns:
        dict: {FL: coverage_percentage}
    """
    return {
        float(fl): float(np.sum(m) / m.size * 100.0)
        for fl, m in coverage_maps.items()
    }


def score_one_radar(
    radar_lat, radar_lon, radar_height_agl_m,
    flight_levels=(5, 10, 20, 50, 100, 200, 300, 400),
    terrain_npz_path="terrain_mat.npz",
    target_n=80,
    n_samples=200,
    margin_m=0.0
):
    """
    Compute a GLOBAL radar score as a WEIGHTED average of coverage percentages
    over the different Flight Levels (FL).

    Weighting rationale (PROJECT_REQ_12):
    - Lowest flight levels (FL5 / FL10 / FL20) are critical for terminal approach
      and runway surveillance → highest weights.
    - Medium FLs (FL50 / FL100) correspond to transition phases.
    - High FLs (FL200+) mainly ensure en-route and long-range surveillance and
      are therefore secondary.

    Returns:
        score (float): weighted global score (%)
        cov_pct (dict): {FL: coverage_percentage}
    """

    # Flight Level weights (sum = 1.0)
    FL_WEIGHTS = {
        5: 0.30,
        10: 0.20,
        20: 0.15,
        50: 0.10,
        100: 0.08,
        200: 0.07,
        300: 0.05,
        400: 0.05,
    }

    # Load terrain data and downsample it
    lats, lons, Z = load_terrain_npz(terrain_npz_path)
    lats_s, lons_s, Z_s = downsample_grid(lats, lons, Z, target_n=target_n)

    # Compute coverage maps for all flight levels
    coverage_maps = compute_all_coverage_maps(
        radar_lat, radar_lon, radar_height_agl_m,
        list(flight_levels),
        lats_s, lons_s, Z_s,
        n_samples=n_samples,
        margin_m=margin_m
    )

    # Compute coverage percentage per FL
    cov_pct = coverage_percent_by_fl(coverage_maps)

    # Weighted score (renormalized if some FLs are missing)
    score = 0.0
    w_sum = 0.0
    for fl, cov in cov_pct.items():
        w = FL_WEIGHTS.get(int(fl), 0.0)
        score += w * cov
        w_sum += w

    if w_sum > 0:
        score /= w_sum

    return float(score), cov_pct


# =========================================================
# TEST (run: python score_weighted.py)
# =========================================================
if __name__ == "__main__":
    # Flight Levels required by PROJECT_REQ_12
    FLs = [5, 10, 20, 50, 100, 200, 300, 400]

    # Example radar position (replace with your candidate site)
    radar_lat = 43.6584
    radar_lon = 7.2159
    radar_h = 20.0  # antenna height above ground level (m)

    score, cov = score_one_radar(
        radar_lat, radar_lon, radar_h,
        flight_levels=FLs,
        terrain_npz_path="terrain_mat.npz",
        target_n=80,     # lower = faster, higher = more accurate
        n_samples=200,   # 100–200 for scoring, 300–400 for final analysis
        margin_m=0.0
    )

    print("\n=== SCORE RESULT ===")
    print("Radar:", radar_lat, radar_lon, "h_agl:", radar_h)
    print("Coverage % per FL:", cov)
    print("WEIGHTED SCORE:", score)
