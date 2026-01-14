# score.py
import numpy as np

from LOS import load_terrain_npz, los_visible, fl_to_m


def _subsample_grid(lats_full, lons_full, Z_full, target_n=80):
    """
    Réduit la grille à ~target_n x ~target_n (comme test_coverage.py) pour aller vite.
    """
    lat_step = max(1, len(lats_full) // target_n)
    lon_step = max(1, len(lons_full) // target_n)

    lats = lats_full[::lat_step][:target_n]
    lons = lons_full[::lon_step][:target_n]
    Z = Z_full[::lat_step, ::lon_step][:target_n, :target_n]

    # sécurité shape
    if Z.shape != (len(lats), len(lons)):
        min_lat = min(len(lats), Z.shape[0])
        min_lon = min(len(lons), Z.shape[1])
        lats = lats[:min_lat]
        lons = lons[:min_lon]
        Z = Z[:min_lat, :min_lon]

    return lats, lons, Z


def score_radar(
    radar_lat: float,
    radar_lon: float,
    radar_height_agl_m: float,
    flight_levels=(5, 10, 20, 50, 100, 200, 300, 400),
    terrain_npz_path="terrain_mat.npz",
    grid_n=80,
    n_samples=40,
    margin_m=0.0,
    verbose=True
):
    """
    Calcule un score pour UN radar = moyenne des pourcentages de coverage
    sur les flight levels donnés.

    - grid_n: taille de la grille réduite (ex: 50, 80, 120)
    - n_samples: points échantillonnés le long du LOS (ex: 30, 40, 60)
    """
    lats_full, lons_full, Z_full = load_terrain_npz(terrain_npz_path)

    # Check radar dans les bornes (sinon tout sera False)
    lat_min, lat_max = float(np.min(lats_full)), float(np.max(lats_full))
    lon_min, lon_max = float(np.min(lons_full)), float(np.max(lons_full))
    if not (lat_min <= radar_lat <= lat_max and lon_min <= radar_lon <= lon_max):
        raise ValueError(
            f"Radar hors terrain: lat {radar_lat} not in [{lat_min},{lat_max}], "
            f"lon {radar_lon} not in [{lon_min},{lon_max}]"
        )

    # Grille réduite pour aller vite
    lats, lons, Z = _subsample_grid(lats_full, lons_full, Z_full, target_n=grid_n)

    total_points = len(lats) * len(lons)
    if verbose:
        print(f"[score] grid utilisée: {len(lats)} x {len(lons)} = {total_points} points")
        print(f"[score] n_samples LOS = {n_samples}, margin = {margin_m} m")
        print(f"[score] FLs = {list(flight_levels)}")

    coverage_pct_by_fl = {}
    for fl in flight_levels:
        target_alt_m = fl_to_m(float(fl))

        visible_count = 0
        done = 0
        # boucle sur points (lat, lon)
        for i in range(len(lats)):
            for j in range(len(lons)):
                ok = los_visible(
                    radar_lat, radar_lon, radar_height_agl_m,
                    float(lats[i]), float(lons[j]), target_alt_m,
                    lats, lons, Z,
                    n_samples=n_samples,
                    margin_m=margin_m
                )
                visible_count += int(ok)
                done += 1

        pct = 100.0 * visible_count / total_points
        coverage_pct_by_fl[float(fl)] = pct

        if verbose:
            print(f"[score] FL{float(fl):.0f}: {pct:.2f}% visible")

    # score = moyenne simple
    score = float(np.mean(list(coverage_pct_by_fl.values())))

    if verbose:
        print(f"[score] SCORE (moyenne FL) = {score:.2f}%")

    return score, coverage_pct_by_fl


if __name__ == "__main__":
    # Exemple rapide (tu changes les coords)
    s, details = score_radar(
        radar_lat=43.6584,
        radar_lon=7.2159,
        radar_height_agl_m=50.0,
        grid_n=80,       # 50 si tu veux encore + rapide
        n_samples=40,    # 30 si tu veux encore + rapide
        verbose=True
    )
