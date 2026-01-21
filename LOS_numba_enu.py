import numpy as np
from numba import njit, prange

# Import your constants + conversion factors
from geo_utils import REF_LAT, REF_LON, EARTH_RADIUS_M


# ----------------------------------------------------
# 1) Helpers: meters/degree factors (same as geo_utils)
# ----------------------------------------------------
def meters_per_degree():
    lat_ref_rad = np.radians(REF_LAT)
    meters_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
    meters_per_deg_lon = (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(lat_ref_rad)
    return float(meters_per_deg_lat), float(meters_per_deg_lon)


def latlon_to_xy_m(lat, lon):
    """Convert one point (lat,lon) to local ENU meters (x east, y north)."""
    mlat, mlon = meters_per_degree()
    y = (float(lat) - float(REF_LAT)) * mlat
    x = (float(lon) - float(REF_LON)) * mlon
    return x, y


def normalize_xy_grid(X_m, Y_m, Z):
    """
    Ensure X_m and Y_m are increasing, and reorder Z accordingly.
    X_m: 1D axis (east), len M
    Y_m: 1D axis (north), len N
    Z : 2D (N,M)
    """
    X_m = np.asarray(X_m)
    Y_m = np.asarray(Y_m)
    Z = np.asarray(Z)

    if Y_m[0] > Y_m[-1]:
        Y_m = Y_m[::-1].copy()
        Z = Z[::-1, :].copy()

    if X_m[0] > X_m[-1]:
        X_m = X_m[::-1].copy()
        Z = Z[:, ::-1].copy()

    return np.ascontiguousarray(X_m), np.ascontiguousarray(Y_m), np.ascontiguousarray(Z)


# ----------------------------
# 2) Physics helpers
# ----------------------------
@njit
def fl_to_m(FL):
    return FL * 100.0 * 0.3048


# ----------------------------------------------------
# 3) Bilinear interpolation in ENU meters (uniform grid)
# ----------------------------------------------------
@njit
def z_bilinear_uniform_xy(x, y, x0, y0, dx, dy, Z):
    """
    Bilinear on uniform increasing ENU grid.
    - x axis corresponds to columns (east)
    - y axis corresponds to rows (north)
    Out-of-bounds -> NaN
    """
    nrow = Z.shape[0]  # Y dimension
    ncol = Z.shape[1]  # X dimension

    x_max = x0 + dx * (ncol - 1)
    y_max = y0 + dy * (nrow - 1)

    if x < x0 or x > x_max or y < y0 or y > y_max:
        return np.nan

    fj = (x - x0) / dx   # col index (float)
    fi = (y - y0) / dy   # row index (float)

    j0 = int(fj)
    i0 = int(fi)

    if j0 < 0:
        j0 = 0
    if i0 < 0:
        i0 = 0
    if j0 > ncol - 2:
        j0 = ncol - 2
    if i0 > nrow - 2:
        i0 = nrow - 2

    j1 = j0 + 1
    i1 = i0 + 1

    u = fj - j0
    t = fi - i0

    z00 = Z[i0, j0]
    z01 = Z[i0, j1]
    z10 = Z[i1, j0]
    z11 = Z[i1, j1]

    z0 = (1.0 - u) * z00 + u * z01
    z1 = (1.0 - u) * z10 + u * z11
    return (1.0 - t) * z0 + t * z1


# ----------------------------------------------------
# 4) LOS in ENU meters, using curvature-corrected Z
# ----------------------------------------------------
@njit
def los_visible_numba_xy(
    radar_x, radar_y, radar_h_agl,
    target_x, target_y, target_alt_msl,
    x0, y0, dx, dy, Zcorr,
    n_samples, margin_m
):
    """
    Exact LOS logic (early exit) in ENU meters.
    Zcorr is already curvature-corrected (same as geo_utils).
    """

    z_ground_r = z_bilinear_uniform_xy(radar_x, radar_y, x0, y0, dx, dy, Zcorr)
    if np.isnan(z_ground_r):
        return False

    z_radar = z_ground_r + radar_h_agl

    for k in range(1, n_samples):
        s = k / n_samples

        x = radar_x + s * (target_x - radar_x)
        y = radar_y + s * (target_y - radar_y)

        z_ground = z_bilinear_uniform_xy(x, y, x0, y0, dx, dy, Zcorr)
        if np.isnan(z_ground):
            return False

        z_line = z_radar + s * (target_alt_msl - z_radar)

        if z_ground + margin_m >= z_line:
            return False

    return True


# ----------------------------------------------------
# 5) FULL GRID coverage map (parallel)
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
    coverage_map[i,j] corresponds to (Y_m[i], X_m[j]).
    """
    N = Y_m.size
    M = X_m.size
    out = np.zeros((N, M), dtype=np.bool_)

    for i in prange(N):
        y = Y_m[i]
        for j in range(M):
            x = X_m[j]
            out[i, j] = los_visible_numba_xy(
                radar_x, radar_y, radar_h_agl,
                x, y, target_alt_msl,
                x0, y0, dx, dy, Zcorr,
                n_samples, margin_m
            )

    return out
