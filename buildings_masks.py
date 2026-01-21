"""
Buildings Masks Module

This module implements boolean geographical masks for building proximity constraint.
Adapted to work with the centralized ENU metric coordinate system.
"""

import numpy as np
import json
from typing import List, Dict

# Constants for coordinate conversion
from geo_utils_earth_curvature import REF_LAT, REF_LON, EARTH_RADIUS_M

def load_buildings_and_convert_to_enu(geojson_file: str) -> List[Dict[str, float]]:
    """
    Load buildings from GeoJSON AND convert to ENU (Meters).

    Parameters:
    -----------
    geojson_file : str
        Path to buildings.geojson

    Returns:
    --------
    List[Dict[str, float]]
        List of buildings with 'x_m', 'y_m' keys (Metric coordinates).
    """
    with open(geojson_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Pre-calculate conversion factors
    lat_ref_rad = np.radians(REF_LAT)
    meters_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
    meters_per_deg_lon = (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(lat_ref_rad)

    buildings_enu = []
    
    # Handle FeatureCollection
    features = data.get("features", [])
    
    for feat in features:
        geom = feat.get("geometry", {})
        # We focus on Points for buildings
        if geom.get("type") != "Point":
            continue
            
        coords = geom.get("coordinates", None)
        if not coords or len(coords) < 2:
            continue

        lon = float(coords[0])
        lat = float(coords[1])

        # Convert to Meters relative to Reference
        y_m = (lat - REF_LAT) * meters_per_deg_lat
        x_m = (lon - REF_LON) * meters_per_deg_lon

        buildings_enu.append({"x_m": x_m, "y_m": y_m})

    return buildings_enu


def mask_buildings_exclusion_fast(X_grid: np.ndarray, Y_grid: np.ndarray,
                                  buildings_metric: List[Dict[str, float]],
                                  radius_m: float = 1000.0) -> np.ndarray:
    """
    Create a boolean mask EXCLUDING locations within radius_m of any building.
    Now works purely in METERS (Euclidean Distance).

    Parameters:
    -----------
    X_grid : np.ndarray (2D)
        X coordinates of the terrain grid in meters.
    Y_grid : np.ndarray (2D)
        Y coordinates of the terrain grid in meters.
    buildings_metric : List[Dict]
        List of buildings with 'x_m' and 'y_m' coordinates.
    radius_m : float
        Exclusion radius in meters (default: 1000.0).

    Returns:
    --------
    mask : np.ndarray (bool)
        True = admissible (FAR from buildings)
        False = excluded (TOO CLOSE to a building)
    """
    # Initialize mask as True (Everything is allowed by default)
    mask = np.ones(X_grid.shape, dtype=bool)

    print(f"   Processing {len(buildings_metric)} buildings (Metric System)...")

    radius_sq = radius_m**2

    for i, b in enumerate(buildings_metric):
        if (i + 1) % 1000 == 0:
            print(f"   ... building {i+1}/{len(buildings_metric)}")

        bx = b["x_m"]
        by = b["y_m"]

        # 1. Bounding Box Filter (in Meters)
        x_min, x_max = bx - radius_m, bx + radius_m
        y_min, y_max = by - radius_m, by + radius_m

        # Find grid points in bounding box
        in_box = ((X_grid >= x_min) & (X_grid <= x_max) &
                  (Y_grid >= y_min) & (Y_grid <= y_max))

        if not np.any(in_box):
            continue

        # 2. Exact Euclidean Distance Squared
        # Only compute for points inside the box
        dist_sq = (X_grid[in_box] - bx)**2 + (Y_grid[in_box] - by)**2

        # Check against radius squared
        too_close = dist_sq <= radius_sq

        # 3. Update Mask (Exclusion Logic)
        # We only modify points inside the box.
        # If a point is 'too_close', it becomes False.
        # Logical AND with NOT too_close:
        # True & ~True  -> False (Excluded)
        # True & ~False -> True  (Kept)
        mask[in_box] = np.logical_and(mask[in_box], ~too_close)
        
        # Optimization: If the whole map is already false, we can stop
        # (Unlikely for buildings, but good practice)
        # if not np.any(mask):
        #     break

    return mask


def mask_buildings_from_geojson(X_grid: np.ndarray, Y_grid: np.ndarray,
                                geojson_file: str = "buildings.geojson",
                                radius_m: float = 1000.0) -> np.ndarray:
    """
    Entry point. Loads GeoJSON, converts to meters, computes exclusion mask.
    """
    # 1. Load and Convert to Meters
    buildings_m = load_buildings_and_convert_to_enu(geojson_file)

    # 2. Compute Mask using Metric Grid
    return mask_buildings_exclusion_fast(X_grid, Y_grid, buildings_m, radius_m)