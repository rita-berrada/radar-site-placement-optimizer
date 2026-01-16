"""
Buildings Masks Module

This module implements boolean geographical masks for building proximity constraint.
These masks identify admissible areas based on distance to building points
(e.g., REQ_xx: no radar within 1 km of any dwelling/building).

Masks are designed to be:
- Reusable and combinable with other geographical masks
- Independent from radar logic
- Compatible with the DTED terrain grid structure
"""

import numpy as np
import json
from typing import List, Dict


def load_buildings_from_geojson(geojson_file: str) -> List[Dict[str, float]]:
    """
    Load buildings data from GeoJSON file (Point geometries).

    Parameters:
    -----------
    geojson_file : str
        Path to buildings.geojson

    Returns:
    --------
    List[Dict[str, float]]
        List of buildings with 'lat', 'lon' keys
    """
    with open(geojson_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    buildings = []
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates", None)
        if not coords or len(coords) < 2:
            continue

        lon = float(coords[0])
        lat = float(coords[1])

        buildings.append({"lat": lat, "lon": lon})

    return buildings


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.

    Parameters:
    -----------
    lat1, lon1 : float
        Latitude and longitude of first point (degrees)
    lat2, lon2 : float
        Latitude and longitude of second point (degrees)

    Returns:
    --------
    float
        Distance in kilometers
    """
    R = 6371.0  # Earth radius in kilometers

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def mask_buildings_exclusion(lats: np.ndarray, lons: np.ndarray,
                            buildings: List[Dict[str, float]],
                            radius_m: float = 1000.0) -> np.ndarray:
    """
    Create a boolean mask excluding locations within radius_m of any building.

    OPTIMIZED VERSION: Uses bounding boxes for faster calculation.

    Mask definition:
    - True  = admissible (farther than radius from all buildings)
    - False = excluded (within radius of at least one building)

    Parameters:
    -----------
    lats : np.ndarray
        1D array of latitude values (degrees)
    lons : np.ndarray
        1D array of longitude values (degrees)
    buildings : List[Dict[str, float]]
        List of buildings with 'lat' and 'lon' keys
    radius_m : float, optional
        Exclusion radius in meters (default: 1000.0)

    Returns:
    --------
    np.ndarray
        Boolean array of shape (len(lats), len(lons))
        True = admissible, False = excluded
    """
    # Create meshgrid for all lat/lon combinations
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Initialize mask as True (everything admissible), then exclude around buildings
    mask = np.ones((len(lats), len(lons)), dtype=bool)

    # Convert radius from meters to kilometers
    radius_km = radius_m / 1000.0

    # Approximate radius in degrees for a bounding box
    # 1 deg lat ≈ 111 km ; 1 deg lon ≈ 111*cos(lat) km
    # Use a conservative latitude near Nice (~43.6°) like the electrical module
    radius_deg_lat = radius_km / 111.0
    radius_deg_lon = radius_km / (111.0 * np.cos(np.radians(43.6)))
    buffer_deg = max(radius_deg_lat, radius_deg_lon) * 1.5  # safety margin

    # Earth radius (km)
    R = 6371.0

    print(f"   Processing {len(buildings)} buildings (optimized)...")

    for i, b in enumerate(buildings):
        if (i + 1) % 500 == 0:
            print(f"   ... building {i+1}/{len(buildings)}")

        b_lat = b["lat"]
        b_lon = b["lon"]

        # Bounding box filter
        lat_min = b_lat - buffer_deg
        lat_max = b_lat + buffer_deg
        lon_min = b_lon - buffer_deg
        lon_max = b_lon + buffer_deg

        in_box = ((lat_grid >= lat_min) & (lat_grid <= lat_max) &
                  (lon_grid >= lon_min) & (lon_grid <= lon_max))

        if not np.any(in_box):
            continue

        # Exact distances only for points in bounding box
        lat1_rad = np.radians(b_lat)
        lon1_rad = np.radians(b_lon)
        lat2_rad = np.radians(lat_grid[in_box])
        lon2_rad = np.radians(lon_grid[in_box])

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        distances_km = R * c
        too_close = distances_km <= radius_km

        # Exclude points too close
        mask[in_box] = np.logical_and(mask[in_box], ~too_close)

        # Micro-optim: if everything excluded, stop early
        if not mask.any():
            break

    return mask


def mask_buildings_from_geojson(lats: np.ndarray, lons: np.ndarray,
                               geojson_file: str = "buildings.geojson",
                               radius_m: float = 1000.0) -> np.ndarray:
    """
    Create buildings exclusion mask directly from GeoJSON file.

    Convenience function that loads buildings and creates mask in one call.

    Parameters:
    -----------
    lats : np.ndarray
        1D array of latitude values (degrees)
    lons : np.ndarray
        1D array of longitude values (degrees)
    geojson_file : str, optional
        Path to buildings GeoJSON (default: "buildings.geojson")
    radius_m : float, optional
        Exclusion radius in meters (default: 1000.0)

    Returns:
    --------
    np.ndarray
        Boolean array (True=admissible, False=excluded)
    """
    buildings = load_buildings_from_geojson(geojson_file)
    return mask_buildings_exclusion(lats, lons, buildings, radius_m)
