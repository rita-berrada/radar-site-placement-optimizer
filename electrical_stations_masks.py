"""
Electrical Stations Masks Module

This module implements boolean geographical masks for electrical infrastructure constraint.
Adapted to work with the centralized ENU metric coordinate system.
"""

import numpy as np
import json
from typing import List, Dict

# We need these constants to convert Station GPS points into Local Meters
from geo_utils_earth_curvature import REF_LAT, REF_LON, EARTH_RADIUS_M


def load_stations_and_convert_to_enu(json_file: str) -> List[Dict[str, float]]:
    """
    Load electrical stations from JSON AND convert to ENU (Meters).

    Parameters:
    -----------
    json_file : str
        Path to the page1.json file containing Enedis stations data.

    Returns:
    --------
    List[Dict[str, float]]
        List of stations with 'x_m', 'y_m' keys (Metric coordinates).
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Pre-calculate conversion factors (Same logic as geo_utils)
    lat_ref_rad = np.radians(REF_LAT)
    meters_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
    meters_per_deg_lon = (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(lat_ref_rad)

    stations_enu = []

    for result in data['results']:
        geopoint = result['_geopoint']
        if isinstance(geopoint, str):
            lat, lon = map(float, geopoint.split(','))
        else:
            lat = geopoint['lat']
            lon = geopoint['lon']

        # Convert to Meters relative to Reference
        y_m = (lat - REF_LAT) * meters_per_deg_lat
        x_m = (lon - REF_LON) * meters_per_deg_lon

        stations_enu.append({
            'x_m': x_m,
            'y_m': y_m
        })

    return stations_enu


def mask_electrical_proximity_fast(X_grid: np.ndarray, Y_grid: np.ndarray, 
                                   stations_metric: List[Dict[str, float]], 
                                   radius_m: float = 500.0) -> np.ndarray:
    """
    Create a boolean mask for locations within specified radius of any station.
    Now works purely in METERS (Euclidean Distance).

    Parameters:
    -----------
    X_grid : np.ndarray (2D)
        X coordinates of the terrain grid in meters.
    Y_grid : np.ndarray (2D)
        Y coordinates of the terrain grid in meters.
    stations_metric : List[Dict]
        List of stations with 'x_m' and 'y_m' coordinates.
    radius_m : float
        Maximum distance in meters (default: 500.0).

    Returns:
    --------
    mask : np.ndarray (bool)
        True = within radius (Admissible).
        False = outside radius (Too far).
    """
    # Initialize mask as False (everything is outside by default)
    mask = np.zeros(X_grid.shape, dtype=bool)

    print(f"   Processing {len(stations_metric)} electrical stations (Metric System)...")

    radius_sq = radius_m**2

    for i, station in enumerate(stations_metric):
        if (i + 1) % 50 == 0:
            print(f"   ... station {i+1}/{len(stations_metric)}")

        sx = station['x_m']
        sy = station['y_m']

        # 1. Bounding Box Filter (in Meters)
        x_min, x_max = sx - radius_m, sx + radius_m
        y_min, y_max = sy - radius_m, sy + radius_m

        # Find grid points in bounding box
        in_box = ((X_grid >= x_min) & (X_grid <= x_max) & 
                  (Y_grid >= y_min) & (Y_grid <= y_max))

        if not np.any(in_box):
            continue

        # 2. Exact Euclidean Distance: (x-sx)^2 + (y-sy)^2
        # Only compute for points inside the box
        dist_sq = (X_grid[in_box] - sx)**2 + (Y_grid[in_box] - sy)**2

        # Check against radius squared
        within_radius = dist_sq <= radius_sq

        # 3. Update mask - points within radius become True
        # We use logical_or to accumulate results from all stations
        mask[in_box] = np.logical_or(mask[in_box], within_radius)

    return mask


def mask_electrical_from_json(X_grid: np.ndarray, Y_grid: np.ndarray,
                              json_file: str = 'geographical_data/page1.json',
                              radius_m: float = 500.0) -> np.ndarray:
    """
    Entry point. Loads JSON, converts to meters, computes mask.

    Parameters:
    -----------
    X_grid : np.ndarray (2D)
        X coordinates of the terrain grid in meters.
    Y_grid : np.ndarray (2D)
        Y coordinates of the terrain grid in meters.
    json_file : str
        Path to electrical stations JSON file (default: "geographical_data/page1.json").
    radius_m : float
        Maximum distance in meters (default: 500.0).

    Returns:
    --------
    mask : np.ndarray (bool)
        True = within radius of at least one electrical station (Admissible).
        False = too far from all stations.
    """
    # 1. Load and Convert to Meters
    print(f"   Loading electrical stations from {json_file}...")
    stations_m = load_stations_and_convert_to_enu(json_file)
    print(f"   ✅ Loaded {len(stations_m)} electrical stations")

    # 2. Compute Mask using Metric Grid
    return mask_electrical_proximity_fast(X_grid, Y_grid, stations_m, radius_m)