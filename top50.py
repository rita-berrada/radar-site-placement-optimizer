import numpy as np

from coverage_analysis import compute_all_coverage_maps
from score import coverage_percent_by_fl
from LOS import load_terrain_npz


def top_k_sites(
    lats_f, lons_f, Z_f,
    k=50,
    candidate_step=20,
    radar_height_agl_m=20.0,
    flight_levels=(5, 10, 20, 50, 100, 200, 300, 400),
    n_samples=120,
    margin_m=0.0,
    candidate_mask=None,
    min_ground_m=None,
    verbose=False
):
    """
    Retourne les k meilleurs sites radar sur un terrain (déjà filtré).
    - lats_f, lons_f, Z_f : terrain filtré
    - candidate_step : pas sur la grille (plus grand = plus rapide)
    - candidate_mask : bool (même shape que Z_f), True = autorisé
    - min_ground_m : filtre altitude (ex: 5m) pour éviter mer/basse altitude
    """
    results = []  # on maintient un top-k

    nlat = len(lats_f)
    nlon = len(lons_f)

    for i in range(0, nlat, candidate_step):
        for j in range(0, nlon, candidate_step):

            z = float(Z_f[i, j])

            # no-data
            if z < 0:
                continue

            # filtre altitude (optionnel)
            if (min_ground_m is not None) and (z < float(min_ground_m)):
                continue

            # masque (optionnel)
            if candidate_mask is not None and not bool(candidate_mask[i, j]):
                continue

            radar_lat = float(lats_f[i])
            radar_lon = float(lons_f[j])

            # cartes coverage via vos fonctions
            coverage_maps = compute_all_coverage_maps(
                radar_lat, radar_lon, radar_height_agl_m,
                list(flight_levels),
                lats_f, lons_f, Z_f,
                n_samples=n_samples,
                margin_m=margin_m
            )

            # % coverage via score.py
            cov_pct = coverage_percent_by_fl(coverage_maps)

            # score = moyenne des % sur les FL demandés
            score = float(np.mean([cov_pct[float(fl)] for fl in flight_levels]))

            item = {"lat": radar_lat, "lon": radar_lon, "score": score, "cov": cov_pct}

            # maintien top-k
            if len(results) < k:
                results.append(item)
                results.sort(key=lambda d: d["score"])
            else:
                if score > results[0]["score"]:
                    results[0] = item
                    results.sort(key=lambda d: d["score"])

            if verbose:
                print(f"site ({radar_lat:.5f},{radar_lon:.5f}) score={score:.2f}")

    results.sort(key=lambda d: d["score"], reverse=True)
    return results


# =========================================================
# TEST intégré (lance: python top50.py)
# =========================================================
if __name__ == "__main__":
    # 1) Charger terrain
    lats, lons, Z = load_terrain_npz("terrain_mat.npz")

    # 2) Terrain filtré (test: on garde tout)
    lats_f, lons_f, Z_f = lats, lons, Z

    # 3) Appel rapide (fait exprès pour ne pas être trop long)
    top = top_k_sites(
        lats_f, lons_f, Z_f,
        k=5,                  # top 5 pour le test
        candidate_step=120,   # énorme pas => très peu de candidats => rapide
        radar_height_agl_m=20.0,
        flight_levels=(5, 10),  # 2 FL pour aller vite
        n_samples=25,         # LOS rapide
        margin_m=0.0,
        min_ground_m=5.0,
        verbose=True
    )

    # 4) Vérifs basiques
    assert isinstance(top, list)
    assert len(top) > 0
    assert len(top) <= 5

    best = top[0]
    assert "lat" in best and "lon" in best and "score" in best and "cov" in best
    assert isinstance(best["score"], float)
    assert 0.0 <= best["score"] <= 100.0
    assert 5.0 in best["cov"] and 10.0 in best["cov"]

    print("\nTEST OK ✅")
    print("Best site:", best)
