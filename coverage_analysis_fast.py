"""
Fast Coverage Analysis (exact same LOS logic, but much faster)

Key idea:
For each target point, compute ONCE the minimum target altitude (MSL) required
to clear all terrain along the segment (radar -> target) using the same sampled LOS model.

Then for each FL: visible iff fl_to_m(FL) > required_altitude.

This removes the x8 repeated LOS work across flight levels and vectorizes the sampling.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from LOS import fl_to_m


# ----------------------------
# Terrain preparation (ensure increasing axes)
# ----------------------------
def prepare_terrain(lats: np.ndarray, lons: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    Z = np.asarray(Z, dtype=float)
    

    lats_inc = lats[0] < lats[-1]
    lons_inc = lons[0] < lons[-1]

    lats_s = lats if lats_inc else lats[::-1]
    lons_s = lons if lons_inc else lons[::-1]

    if lats_inc and lons_inc:
        Z_s = Z
    elif (not lats_inc) and lons_inc:
        Z_s = Z[::-1, :]
    elif lats_inc and (not lons_inc):
        Z_s = Z[:, ::-1]
    else:
        Z_s = Z[::-1, ::-1]

    return lats_s, lons_s, Z_s


# ----------------------------
# Fast bilinear interpolation (batch)
# Returns np.nan where out-of-bounds or no-data corners
# ----------------------------
def bilinear_terrain_batch(lat: np.ndarray, lon: np.ndarray,
                           lats: np.ndarray, lons: np.ndarray, Z: np.ndarray) -> np.ndarray:
    eps = 1e-12

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    # bounds
    inb = (lat >= lats[0]) & (lat <= lats[-1]) & (lon >= lons[0]) & (lon <= lons[-1])

    # indices
    i1 = np.searchsorted(lats, lat, side="left")
    j1 = np.searchsorted(lons, lon, side="left")

    i1 = np.clip(i1, 1, len(lats) - 1)
    j1 = np.clip(j1, 1, len(lons) - 1)
    i0 = i1 - 1
    j0 = j1 - 1

    lat0 = lats[i0]
    lat1 = lats[i1]
    lon0 = lons[j0]
    lon1 = lons[j1]

    z00 = Z[i0, j0]
    z01 = Z[i0, j1]
    z10 = Z[i1, j0]
    z11 = Z[i1, j1]

    # no-data if any corner < 0
    nodata = (z00 < 0) | (z01 < 0) | (z10 < 0) | (z11 < 0)

    t = (lat - lat0) / (lat1 - lat0 + eps)
    u = (lon - lon0) / (lon1 - lon0 + eps)

    z0 = (1 - u) * z00 + u * z01
    z1 = (1 - u) * z10 + u * z11
    z = (1 - t) * z0 + t * z1

    z = np.where(inb & (~nodata), z, np.nan)
    return z


def z_terrain_scalar(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray, Z: np.ndarray) -> Optional[float]:
    zz = bilinear_terrain_batch(np.array([lat]), np.array([lon]), lats, lons, Z)[0]
    if not np.isfinite(zz):
        return None
    return float(zz)


# ----------------------------
# Core: compute required altitude for each target (exact sampled LOS model)
# ----------------------------
def required_altitude_batch(
    radar_lat: float,
    radar_lon: float,
    z_radar_msl: float,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    Z: np.ndarray,
    n_samples: int = 400,
    margin_m: float = 0.0
) -> np.ndarray:
    """
    For each target, compute min target altitude (MSL) required so that
    z_line(s) > z_terrain(s) + margin for all sampled s in (0,1).

    Same LOS sampling model, but vectorized.
    Returns req_alt (MSL) with np.inf when path hits no-data/out-of-bounds.
    """
    target_lat = np.asarray(target_lat, dtype=float)
    target_lon = np.asarray(target_lon, dtype=float)

    # s samples (exclude 0 and 1)
    s = np.arange(1, n_samples, dtype=float) / float(n_samples)  # (S,)
    s2 = s[None, :]  # (1,S)

    # Sample points along segment for each target: (B,S)
    lat_s = radar_lat + (target_lat[:, None] - radar_lat) * s2
    lon_s = radar_lon + (target_lon[:, None] - radar_lon) * s2

    z_s = bilinear_terrain_batch(lat_s, lon_s, lats, lons, Z)  # (B,S)

    # If any NaN along the ray => invalid (safe choice)
    valid = np.isfinite(z_s).all(axis=1)

    # Required altitude at each sample:
    # z_radar + s*(z_target - z_radar) > z_s + margin
    # => z_target > z_radar + (z_s - z_radar + margin)/s
    req_k = z_radar_msl + (z_s - z_radar_msl + float(margin_m)) / s2  # (B,S)

    req = np.nanmax(req_k, axis=1)  # (B,)
    req[~valid] = np.inf
    return req


# ----------------------------
# Public API: compute all FL maps fast, exact to sampled LOS
# ----------------------------
def compute_all_coverage_maps_fast(
    radar_lat: float,
    radar_lon: float,
    radar_height_agl_m: float,
    flight_levels: List[float],
    lats: np.ndarray,
    lons: np.ndarray,
    Z: np.ndarray,
    n_samples: int = 400,
    margin_m: float = 0.0,
    batch_size: int = 1024,
    point_progress_callback: Optional[callable] = None,
) -> Dict[float, np.ndarray]:
    """
    Returns {FL: coverage_map_bool} for all FL in one pass, much faster.

    Precision: EXACT relative to the same sampled LOS rule (n_samples, margin_m),
    but computed efficiently.
    """
    lats_s, lons_s, Z_s = prepare_terrain(lats, lons, Z)
    N, M = Z_s.shape

    # Radar altitude MSL
    z_ground = z_terrain_scalar(radar_lat, radar_lon, lats_s, lons_s, Z_s)
    if z_ground is None:
        raise ValueError("Radar position out of bounds or no-data in terrain.")
    z_radar = float(z_ground) + float(radar_height_agl_m)

    # FL altitudes MSL
    fls = [float(fl) for fl in flight_levels]
    fl_alt = np.array([fl_to_m(fl) for fl in fls], dtype=float)  # (F,)
    F = len(fls)

    # Flatten grid coords
    lat_grid = np.repeat(lats_s, M)      # (N*M,)
    lon_grid = np.tile(lons_s, N)        # (N*M,)
    Z_flat = Z_s.ravel()

    # Valid target points (avoid targets on no-data)
    # IMPORTANT: on calcule la visibilité pour TOUS les points de la grille,
    # même si Z=-1 au point cible, pour rester cohérent avec los_visible (qui ne teste pas s=1).
    idx_all = np.arange(N * M)
    total = idx_all.size


    cov_flat = np.zeros((F, N * M), dtype=bool)

    done = 0
    for start in range(0, total, batch_size):
        idx = idx_all[start:start + batch_size]
        tgt_lat = lat_grid[idx]
        tgt_lon = lon_grid[idx]

        req_alt = required_altitude_batch(
            radar_lat, radar_lon, z_radar,
            tgt_lat, tgt_lon,
            lats_s, lons_s, Z_s,
            n_samples=n_samples,
            margin_m=margin_m
        )  # (B,)

        vis = fl_alt[:, None] > req_alt[None, :]  # (F,B)
        cov_flat[:, idx] = vis

        done += len(idx)
        if point_progress_callback:
            point_progress_callback(done, total, 100.0 * done / total)

    # Reshape back to 2D maps
    coverage_maps = {fls[k]: cov_flat[k].reshape(N, M) for k in range(F)}
    return coverage_maps
