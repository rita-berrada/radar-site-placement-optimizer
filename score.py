import numpy as np

from LOS import load_terrain_npz
from coverage_analysis import compute_all_coverage_maps


def downsample_grid(lats, lons, Z, target_n=80):
    """
    Réduit la grille à ~target_n x ~target_n pour accélérer (comme vos tests).
    """
    lat_step = max(1, len(lats) // target_n)
    lon_step = max(1, len(lons) // target_n)

    lats_s = lats[::lat_step][:target_n]
    lons_s = lons[::lon_step][:target_n]
    Z_s = Z[::lat_step, ::lon_step][:target_n, :target_n]

    # sécurité shape
    nlat = min(len(lats_s), Z_s.shape[0])
    nlon = min(len(lons_s), Z_s.shape[1])
    return lats_s[:nlat], lons_s[:nlon], Z_s[:nlat, :nlon]


def coverage_percent_by_fl(coverage_maps):
    """
    coverage_maps[FL] = matrice bool (True=visible)
    -> retourne {FL: coverage_en_%}
    """
    return {float(fl): float(np.sum(m) / m.size * 100.0) for fl, m in coverage_maps.items()}


def score_one_radar(
    radar_lat, radar_lon, radar_height_agl_m,
    flight_levels=(5, 10, 20, 50, 100, 200, 300, 400),
    terrain_npz_path="terrain_mat.npz",
    target_n=80,
    n_samples=200,
    margin_m=0.0
):
    """
    Score d'un radar = moyenne des taux de coverage (%) sur les FL.
    Retourne (score, {FL: coverage_%})
    """
    lats, lons, Z = load_terrain_npz(terrain_npz_path)
    lats_s, lons_s, Z_s = downsample_grid(lats, lons, Z, target_n=target_n)

    coverage_maps = compute_all_coverage_maps(
        radar_lat, radar_lon, radar_height_agl_m,
        list(flight_levels),
        lats_s, lons_s, Z_s,
        n_samples=n_samples,
        margin_m=margin_m
    )

    cov_pct = coverage_percent_by_fl(coverage_maps)
    score = float(np.mean(list(cov_pct.values())))
    return score, cov_pct


# =========================================================
# TEST (lance: python score.py)
# =========================================================
if __name__ == "__main__":
    FLs = [5, 10, 20, 50, 100, 200, 300, 400]

    # Exemple radar (mets ta position)
    radar_lat = 43.6584
    radar_lon = 7.2159
    radar_h = 20.0

    score, cov = score_one_radar(
        radar_lat, radar_lon, radar_h,
        flight_levels=FLs,
        terrain_npz_path="terrain_mat.npz",
        target_n=80,     # 50 = très rapide, 80 = bon, 120 = plus précis
        n_samples=200,   # 100-200 ok pour score, 300-400 pour final
        margin_m=0.0
    )

    print("\n=== SCORE RESULT ===")
    print("Radar:", radar_lat, radar_lon, "h_agl:", radar_h)
    print("Coverage % par FL:", cov)
    print("SCORE (moyenne):", score)
