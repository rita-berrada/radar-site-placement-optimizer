"""
Buildings Masks Module

This module implements boolean geographical masks for building proximity constraint.
Adapted to work with the centralized ENU metric coordinate system.

UPDATE: The buffer is now SUBTRACTED from the exclusion radius.
This makes the constraint MORE PERMISSIVE.
Exclusion Radius = 1000m - 100m = 900m.
(Sites at 900m+ are now accepted).
"""

import numpy as np
import json
from typing import List, Dict

# Constants for coordinate conversion
from geo_utils_earth_curvature import REF_LAT, REF_LON, EARTH_RADIUS_M

def load_buildings_and_convert_to_enu(geojson_file: str) -> List[Dict[str, float]]:
    """
    Load buildings from GeoJSON AND convert to ENU (Meters).
    """
    try:
        with open(geojson_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"   [Error] Could not load buildings file {geojson_file}: {e}")
        return []

    # Pre-calculate conversion factors
    lat_ref_rad = np.radians(REF_LAT)
    meters_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
    meters_per_deg_lon = (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(lat_ref_rad)

    buildings_enu = []
    
    # Handle FeatureCollection
    features = data.get("features", [])
    
    for feat in features:
        geom = feat.get("geometry", {})
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
    
    Parameters:
    -----------
    radius_m : float
        The effective exclusion radius (after buffer adjustment).
    """
    # Initialize mask as True (Everything is allowed by default)
    mask = np.ones(X_grid.shape, dtype=bool)

    if not buildings_metric:
        print("   [Warn] No buildings loaded. Mask will be all True (Admissible).")
        return mask

    print(f"   Processing {len(buildings_metric)} buildings (Metric System, Exclusion Radius={radius_m:.0f}m)...")

    radius_sq = radius_m**2
    total_buildings = len(buildings_metric)

    for i, b in enumerate(buildings_metric):
        if total_buildings > 1000 and (i + 1) % 1000 == 0:
            print(f"   ... building {i+1}/{total_buildings}")

        bx = b["x_m"]
        by = b["y_m"]

        # 1. Bounding Box Filter
        x_min, x_max = bx - radius_m, bx + radius_m
        y_min, y_max = by - radius_m, by + radius_m

        in_box = ((X_grid >= x_min) & (X_grid <= x_max) &
                  (Y_grid >= y_min) & (Y_grid <= y_max))

        if not np.any(in_box):
            continue

        # 2. Exact Euclidean Distance
        dist_sq = (X_grid[in_box] - bx)**2 + (Y_grid[in_box] - by)**2

        # Check against radius squared
        too_close = dist_sq <= radius_sq

        # 3. Update Mask (Exclude points that are too close)
        mask[in_box] = np.logical_and(mask[in_box], ~too_close)

    return mask


def mask_buildings_from_geojson(X_grid: np.ndarray, Y_grid: np.ndarray,
                                geojson_file: str = "geographical_data/buildings.geojson",
                                radius_m: float = 1000.0,
                                buffer_m: float = 100.0) -> np.ndarray:
    """
    Entry point.
    
    Parameters:
    -----------
    radius_m : float
        Strict constraint (default 1000m).
    buffer_m : float
        Tolerance buffer (default 100m).
    
    Calculation:
    ------------
    Effective Exclusion = radius_m - buffer_m = 900m.
    This means we tolerate proximity up to 900m instead of 1000m.
    """
    # 1. Load and Convert to Meters
    print(f"   Loading buildings from {geojson_file}...")
    buildings_m = load_buildings_and_convert_to_enu(geojson_file)
    print(f"   ✅ Loaded {len(buildings_m)} buildings")

    # 2. Compute Effective Radius (SUBTRACTION for Exclusion)
    # We allow getting closer by 'buffer_m' meters.
    effective_radius = max(0.0, radius_m - buffer_m)
    
    print(f"   [Constraint] Building Exclusion: Base {radius_m}m - Tolerance {buffer_m}m = {effective_radius}m (Effective Exclusion)")

    # 3. Compute Mask using Metric Grid
    return mask_buildings_exclusion_fast(X_grid, Y_grid, buildings_m, effective_radius)