import numpy as np

def _normalize_grid(lats, lons, Z):
    """
    Force lats and lons to be increasing and reorders Z accordingly.
    This removes a LOT of subtle bugs (searchsorted, plotting, etc).
    """
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    Z = np.asarray(Z)

    if lats[0] > lats[-1]:
        lats = lats[::-1]
        Z = Z[::-1, :]

    if lons[0] > lons[-1]:
        lons = lons[::-1]
        Z = Z[:, ::-1]

    return lats, lons, Z


def z_terrain_vec(lat, lon, lats, lons, Z):
    """
    Bilinear interpolation, vectorized.
    Negative terrain is treated as VALID altitude (sea is valid).
    Out-of-bounds -> np.nan
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    lats, lons, Z = _normalize_grid(lats, lons, Z)

    out = np.full(lat.shape, np.nan, dtype=float)

    inb = (lat >= lats[0]) & (lat <= lats[-1]) & (lon >= lons[0]) & (lon <= lons[-1])
    if not np.any(inb):
        return out

    lat_i = lat[inb]
    lon_i = lon[inb]

    i1 = np.searchsorted(lats, lat_i, side="right")
    j1 = np.searchsorted(lons, lon_i, side="right")

    i1 = np.clip(i1, 1, len(lats) - 1)
    j1 = np.clip(j1, 1, len(lons) - 1)
    i0 = i1 - 1
    j0 = j1 - 1

    lat0 = lats[i0]; lat1 = lats[i1]
    lon0 = lons[j0]; lon1 = lons[j1]

    z00 = Z[i0, j0]
    z01 = Z[i0, j1]
    z10 = Z[i1, j0]
    z11 = Z[i1, j1]

    t = (lat_i - lat0) / (lat1 - lat0 + 1e-12)
    u = (lon_i - lon0) / (lon1 - lon0 + 1e-12)

    z0 = (1 - u) * z00 + u * z01
    z1 = (1 - u) * z10 + u * z11
    z  = (1 - t) * z0  + t * z1

    out[inb] = z
    return out


def fl_to_m(FL: float) -> float:
    return FL * 100.0 * 0.3048


def los_visible(radar_lat, radar_lon, radar_height_agl_m,
                target_lat, target_lon, target_alt_m_msl,
                lats, lons, Z,
                n_samples=400, margin_m=0.0):
    """
    EXACT same logic as your LOS.py, but the inner sampling loop is vectorized.
    No azimuth bins, no ray casting, no viewshed approximation.
    """

    # normalize grid once (important)
    lats, lons, Z = _normalize_grid(lats, lons, Z)

    # radar ground
    z_ground_r = z_terrain_vec(np.array([radar_lat]), np.array([radar_lon]), lats, lons, Z)[0]
    if np.isnan(z_ground_r):
        return False
    z_radar = float(z_ground_r) + float(radar_height_agl_m)

    # samples s in (0,1)
    s = np.arange(1, n_samples, dtype=float) / float(n_samples)

    # line points
    lat = radar_lat + s * (target_lat - radar_lat)
    lon = radar_lon + s * (target_lon - radar_lon)

    # terrain along the line
    z_ground = z_terrain_vec(lat, lon, lats, lons, Z)
    if np.any(np.isnan(z_ground)):
        return False

    # line altitude along LOS
    z_line = z_radar + s * (float(target_alt_m_msl) - z_radar)

    # blocked?
    return not np.any((z_ground + margin_m) >= z_line)
