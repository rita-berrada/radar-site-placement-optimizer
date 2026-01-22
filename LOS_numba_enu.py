"""
LOS_numba_enu.py

High-performance Line of Sight (LOS) engine using Numba.
Works with the centralized ENU Metric System.

CRITICAL PHYSICS FEATURE:
It applies the Earth Curvature Drop to the TARGET (Aircraft) altitude
to match the curvature drop already applied to the TERRAIN (Z_corrected).
"""

import numpy as np
from numba import njit, prange

# Import constants to ensure we use the same Earth model as the rest of the project
# Note: Ensure geo_utils.py or geo_utils_earth_curvature.py is available
from geo_utils_earth_curvature import REF_LAT, REF_LON, EARTH_RADIUS_M

# ----------------------------------------------------
# 1) Helpers: meters/degree conversion
# ----------------------------------------------------
def meters_per_degree():
    """
    Returns the scaling factors (lat, lon) for the reference point.
    """
    lat_ref_rad = np.radians(REF_LAT)
    meters_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
    meters_per_deg_lon = (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(lat_ref_rad)
    return float(meters_per_deg_lat), float(meters_per_deg_lon)


def latlon_to_xy_m(lat, lon):
    """
    Convert one point (lat,lon) to local ENU meters (x east, y north).
    Relative to the project Reference Point (0,0).
    """
    mlat, mlon = meters_per_degree()
    y = (float(lat) - float(REF_LAT)) * mlat
    x = (float(lon) - float(REF_LON)) * mlon
    return x, y


def normalize_xy_grid(X_m, Y_m, Z):
    """
    Ensure X_m and Y_m are strictly increasing, and reorder Z accordingly.
    Required for consistent bilinear interpolation indices.
    
    X_m: 1D axis (east), len M
    Y_m: 1D axis (north), len N
    Z : 2D (N,M)
    """
    X_m = np.asarray(X_m)
    Y_m = np.asarray(Y_m)
    Z = np.asarray(Z)

    # Flip Y if decreasing (North to South -> South to North)
    if Y_m[0] > Y_m[-1]:
        Y_m = Y_m[::-1].copy()
        Z = Z[::-1, :].copy()

    # Flip X if decreasing (East to West -> West to East)
    if X_m[0] > X_m[-1]:
        X_m = X_m[::-1].copy()
        Z = Z[:, ::-1].copy()

    return np.ascontiguousarray(X_m), np.ascontiguousarray(Y_m), np.ascontiguousarray(Z)


# ----------------------------
# 2) Physics helpers
# ----------------------------
@njit
def fl_to_m(FL):
    """Convert Flight Level (hundreds of feet) to Meters."""
    return FL * 100.0 * 0.3048


# ----------------------------------------------------
# 3) Bilinear interpolation in ENU meters
# ----------------------------------------------------
@njit
def z_bilinear_uniform_xy(x, y, x0, y0, dx, dy, Z):
    """
    Bilinear interpolation on uniform increasing ENU grid.
    Returns NaN if (x,y) is outside the grid.
    """
    nrow = Z.shape[0]  # Y dimension (rows)
    ncol = Z.shape[1]  # X dimension (cols)

    x_max = x0 + dx * (ncol - 1)
    y_max = y0 + dy * (nrow - 1)

    # Boundary check
    if x < x0 or x > x_max or y < y0 or y > y_max:
        return np.nan

    # Continuous index
    fj = (x - x0) / dx
    fi = (y - y0) / dy

    j0 = int(fj)
    i0 = int(fi)

    # Clamp indices to avoid index out of bounds
    if j0 < 0: j0 = 0
    if i0 < 0: i0 = 0
    if j0 > ncol - 2: j0 = ncol - 2
    if i0 > nrow - 2: i0 = nrow - 2

    j1 = j0 + 1
    i1 = i0 + 1

    u = fj - j0
    t = fi - i0

    # Fetch 4 neighbors
    z00 = Z[i0, j0]
    z01 = Z[i0, j1]
    z10 = Z[i1, j0]
    z11 = Z[i1, j1]

    # Interpolate
    z0 = (1.0 - u) * z00 + u * z01
    z1 = (1.0 - u) * z10 + u * z11
    return (1.0 - t) * z0 + t * z1


# ----------------------------------------------------
# 4) CORE ENGINE: LOS with DUAL CURVATURE CORRECTION
# ----------------------------------------------------
@njit
def los_visible_numba_xy(
    radar_x, radar_y, radar_h_agl,
    target_x, target_y, target_alt_msl,
    x0, y0, dx, dy, Zcorr,
    n_samples, margin_m
):
    """
    Checks if a Line of Sight exists between Radar and Target.
    
    Coordinate System: ENU (Meters) relative to Nice Airport (0,0).
    Zcorr: Terrain Grid ALREADY corrected for Earth curvature drop.
    
    PHYSICS CORRECTION:
    The target altitude (target_alt_msl) is also lowered by the curvature drop
    formula (d^2 / 2R) to maintain geometric consistency with the terrain.
    """

    # 1. Get Radar Ground Altitude from Zcorr
    # (Usually small drop near origin, but we interpolate for precision)
    z_ground_r = z_bilinear_uniform_xy(radar_x, radar_y, x0, y0, dx, dy, Zcorr)
    if np.isnan(z_ground_r):
        return False # Radar is off-grid
        
    z_radar = z_ground_r + radar_h_agl

    # 2. Compute Target Altitude in the Curved ENU Frame
    # Calculate distance squared from Reference Origin (0,0)
    dist_sq_target = target_x**2 + target_y**2
    
    # Calculate how much the Earth curves down at this distance
    drop_target = dist_sq_target / (2.0 * EARTH_RADIUS_M)
    
    # Apply the drop to the target's MSL altitude
    z_target_enu = target_alt_msl - drop_target

    # 3. Ray Casting (Sampling points along the line)
    for k in range(1, n_samples):
        s = k / n_samples  # normalized distance (0 to 1)

        # Interpolate X, Y along the path
        x = radar_x + s * (target_x - radar_x)
        y = radar_y + s * (target_y - radar_y)

        # Get Terrain Z at this point (already curved in the file)
        z_ground = z_bilinear_uniform_xy(x, y, x0, y0, dx, dy, Zcorr)
        if np.isnan(z_ground):
            return False # Path goes off-grid

        # Interpolate the Line of Sight Z 
        # (Linear interpolation between Radar and Lowered Target)
        z_line = z_radar + s * (z_target_enu - z_radar)

        # Check for Obstruction
        if z_ground + margin_m >= z_line:
            return False # Blocked!

    return True # Path is clear


# ----------------------------------------------------
# 5) BATCH PROCESSOR: Coverage Map
# ----------------------------------------------------
@njit(parallel=True)
def coverage_map_numba_xy(
    radar_x, radar_y, radar_h_agl,
    target_alt_msl,
    X_m, Y_m,
    x0, y0, dx, dy, Zcorr,
    n_samples, margin_m
):
    """
    Computes a full visibility grid for a given Target Altitude.
    Parallelized loop over every pixel of the map.
    """
    N = Y_m.size
    M = X_m.size
    out = np.zeros((N, M), dtype=np.bool_)

    # Loop over all pixels (each pixel is a potential target location)
    for i in prange(N):
        y = Y_m[i]
        for j in range(M):
            x = X_m[j]
            
            # Check LOS from Radar to this specific pixel (x,y)
            out[i, j] = los_visible_numba_xy(
                radar_x, radar_y, radar_h_agl,
                x, y, target_alt_msl,
                x0, y0, dx, dy, Zcorr,
                n_samples, margin_m
            )

    return out