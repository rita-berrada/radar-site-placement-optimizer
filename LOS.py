import numpy as np
from visualize_terrain import load_terrain_npz


# Terrain interpolation
def z_terrain(lat: float, lon: float,
              lats: np.ndarray, lons: np.ndarray, Z: np.ndarray):
    """
    Terrain altitude (m) at point (lat, lon) via bilinear interpolation.
    Returns None if out of bounds or no-data (values < 0).
    """

    # Check if axes are increasing or decreasing
    lats_inc = lats[0] < lats[-1]
    lons_inc = lons[0] < lons[-1]

    lat_min, lat_max = (lats[0], lats[-1]) if lats_inc else (lats[-1], lats[0])
    lon_min, lon_max = (lons[0], lons[-1]) if lons_inc else (lons[-1], lons[0])

    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
        return None

    # Make increasing for searchsorted
    lats_s = lats if lats_inc else lats[::-1]
    lons_s = lons if lons_inc else lons[::-1]

    # Reorder Z
    if lats_inc and lons_inc:
        Z_s = Z
    elif (not lats_inc) and lons_inc:
        Z_s = Z[::-1, :]
    elif lats_inc and (not lons_inc):
        Z_s = Z[:, ::-1]
    else:
        Z_s = Z[::-1, ::-1]

    i1 = int(np.searchsorted(lats_s, lat))
    j1 = int(np.searchsorted(lons_s, lon))
    i1 = int(np.clip(i1, 1, len(lats_s) - 1))
    j1 = int(np.clip(j1, 1, len(lons_s) - 1))
    i0, j0 = i1 - 1, j1 - 1

    lat0, lat1 = lats_s[i0], lats_s[i1]
    lon0, lon1 = lons_s[j0], lons_s[j1]

    z00 = Z_s[i0, j0]
    z01 = Z_s[i0, j1]
    z10 = Z_s[i1, j0]
    z11 = Z_s[i1, j1]

    # No-data: negative values
    if min(z00, z01, z10, z11) < 0:
        return None

    t = (lat - lat0) / (lat1 - lat0 + 1e-12)
    u = (lon - lon0) / (lon1 - lon0 + 1e-12)

    z0 = (1 - u) * z00 + u * z01
    z1 = (1 - u) * z10 + u * z11
    return float((1 - t) * z0 + t * z1)


# Line altitude
def z_ligne(s: float, z_radar_m: float, z_target_m: float) -> float:
    """Altitude (m) on the radar->target line (s∈[0,1])."""
    return z_radar_m + s * (z_target_m - z_radar_m)


def fl_to_m(FL: float) -> float:
    """FLxxx = xxx*100 ft ; 1 ft = 0.3048 m."""
    return FL * 100.0 * 0.3048


# LOS function
def los_visible(radar_lat: float, radar_lon: float, radar_height_agl_m: float,
                target_lat: float, target_lon: float, target_alt_m_msl: float,
                lats: np.ndarray, lons: np.ndarray, Z: np.ndarray,
                n_samples: int = 400, margin_m: float = 0.0) -> bool:
    """
    Returns True if LOS is clear, False otherwise.

    - radar_height_agl_m : tower height above ground level (AGL)
    - target_alt_m_msl   : target altitude in m (MSL), e.g.: fl_to_m(50)
    - margin_m           : safety margin (0 or 10m for example)
    """

    z_ground_r = z_terrain(radar_lat, radar_lon, lats, lons, Z)
    if z_ground_r is None:
        return False
    z_radar = z_ground_r + radar_height_agl_m

    for k in range(1, n_samples):
        s = k / n_samples

        lat = radar_lat + s * (target_lat - radar_lat)
        lon = radar_lon + s * (target_lon - radar_lon)

        z_ground = z_terrain(lat, lon, lats, lons, Z)
        if z_ground is None:
            return False  # Safe: no-data => consider blocked

        z_line = z_ligne(s, z_radar, target_alt_m_msl)

        if z_ground + margin_m >= z_line:
            return False

    return True

