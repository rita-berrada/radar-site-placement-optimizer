# mask_50km.py
"""
PROJECT_REQ_01 mask:
Build a boolean mask representing the 50 km allowed area
around Nice Côte d’Azur Airport.

True  -> allowed zone (<= 50 km)
False -> forbidden zone (> 50 km)
"""

import numpy as np
import math


# Nice Côte d’Azur Airport (NCE)
NICE_LAT = 43.6584
NICE_LON = 7.2159
REQ01_RADIUS_KM = 50.0


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorized great-circle distance (km).
    lat1, lon1 can be arrays.
    lat2, lon2 are scalars.
    """
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def build_50km_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Build PROJECT_REQ_01 mask over a terrain grid.

    Parameters
    ----------
    lat : ndarray (n_lat,)
    lon : ndarray (n_lon,)

    Returns
    -------
    mask : ndarray (n_lat, n_lon)
        True  -> inside 50 km
        False -> outside 50 km
    """
    LAT2D = lat[:, None]      # (n_lat, 1)
    LON2D = lon[None, :]      # (1, n_lon)

    dist_km = haversine_km(LAT2D, LON2D, NICE_LAT, NICE_LON)
    mask = dist_km <= REQ01_RADIUS_KM

    return mask


def save_50km_mask(
    terrain_npz_path: str,
    output_npz_path: str = "mask_req01_50km.npz"
):
    """
    Load terrain (.npz), build REQ01 mask and save it.

    Output file contains:
      - mask_req01 (uint8 0/1)
      - lat
      - lon
    """
    data = np.load(terrain_npz_path)
    lat = data["lat"]
    lon = data["lon"]

    mask = build_50km_mask(lat, lon)

    np.savez_compressed(
        output_npz_path,
        mask_req01=mask.astype(np.uint8),
        lat=lat,
        lon=lon
    )

    print("PROJECT_REQ_01 mask saved to:", output_npz_path)
    print("Allowed cells:", int(mask.sum()), "/", mask.size)
