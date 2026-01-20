import numpy as np
from numba import njit, prange

def normalize_grid(lats, lons, Z):
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    Z = np.asarray(Z)

    if lats[0] > lats[-1]:
        lats = lats[::-1].copy()
        Z = Z[::-1, :].copy()
    if lons[0] > lons[-1]:
        lons = lons[::-1].copy()
        Z = Z[:, ::-1].copy()

    # IMPORTANT: rendre contigu (Numba aime ça)
    return np.ascontiguousarray(lats), np.ascontiguousarray(lons), np.ascontiguousarray(Z)

@njit
def fl_to_m(FL):
    return FL * 100.0 * 0.3048

@njit
def z_bilinear_uniform(lat, lon, lats0, lons0, dlat, dlon, Z):
    # Out of bounds => nan
    if lat < lats0 or lon < lons0:
        return np.nan

    nlat = Z.shape[0]
    nlon = Z.shape[1]
    lat_max = lats0 + dlat * (nlat - 1)
    lon_max = lons0 + dlon * (nlon - 1)
    if lat > lat_max or lon > lon_max:
        return np.nan

    fi = (lat - lats0) / dlat
    fj = (lon - lons0) / dlon

    i0 = int(fi)
    j0 = int(fj)

    if i0 < 0: i0 = 0
    if j0 < 0: j0 = 0
    if i0 > nlat - 2: i0 = nlat - 2
    if j0 > nlon - 2: j0 = nlon - 2

    i1 = i0 + 1
    j1 = j0 + 1

    t = fi - i0
    u = fj - j0

    z00 = Z[i0, j0]
    z01 = Z[i0, j1]
    z10 = Z[i1, j0]
    z11 = Z[i1, j1]

    z0 = (1.0 - u) * z00 + u * z01
    z1 = (1.0 - u) * z10 + u * z11
    return (1.0 - t) * z0 + t * z1


@njit
def los_visible_numba(
    radar_lat, radar_lon, radar_h_agl,
    target_lat, target_lon, target_alt_msl,
    lats0, lons0, dlat, dlon, Z,
    n_samples, margin_m
):
    # radar ground
    z_ground_r = z_bilinear_uniform(radar_lat, radar_lon, lats0, lons0, dlat, dlon, Z)
    if np.isnan(z_ground_r):
        return False
    z_radar = z_ground_r + radar_h_agl

    # Early-exit exact (comme LOS.py)
    for k in range(1, n_samples):
        s = k / n_samples
        lat = radar_lat + s * (target_lat - radar_lat)
        lon = radar_lon + s * (target_lon - radar_lon)

        z_ground = z_bilinear_uniform(lat, lon, lats0, lons0, dlat, dlon, Z)
        if np.isnan(z_ground):
            return False

        z_line = z_radar + s * (target_alt_msl - z_radar)
        if z_ground + margin_m >= z_line:
            return False

    return True


@njit(parallel=True)
def coverage_map_numba(
    radar_lat, radar_lon, radar_h_agl,
    target_alt_msl,
    lats, lons,
    lats0, lons0, dlat, dlon, Z,
    n_samples, margin_m
):
    N = lats.size
    M = lons.size
    out = np.zeros((N, M), dtype=np.bool_)

    for i in prange(N):
        for j in range(M):
            out[i, j] = los_visible_numba(
                radar_lat, radar_lon, radar_h_agl,
                lats[i], lons[j], target_alt_msl,
                lats0, lons0, dlat, dlon, Z,
                n_samples, margin_m
            )
    return out
