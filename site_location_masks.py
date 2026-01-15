"""
Site Location Masks Module

This module implements boolean geographical masks for Lot 2 - Radar site location study.
These masks identify admissible areas for radar installation based on static geographical
constraints without modifying terrain data.

The masks are designed to be:
- Reusable and combinable
- Independent from any radar logic
- Compatible with the DTED terrain grid structure
"""

import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth using the haversine formula.
    
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
    # Earth radius in kilometers
    R = 6371.0
    
    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance_km = R * c
    return distance_km


def mask_land(lats: np.ndarray, lons: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Create a boolean mask for onshore areas (land).
    
    The mask is True where elevation > 0 meters (land) and False where elevation <= 0 (sea/offshore).
    This mask is based solely on terrain elevation from the DTED grid.
    
    Parameters:
    -----------
    lats : np.ndarray
        1D array of latitude values (degrees)
    lons : np.ndarray
        1D array of longitude values (degrees)
    Z : np.ndarray
        2D array of terrain elevation (meters above sea level), shape (len(lats), len(lons))
    
    Returns:
    --------
    np.ndarray
        Boolean array of shape (len(lats), len(lons))
        True = land (admissible), False = sea/offshore (excluded)
    """
    if Z.shape != (len(lats), len(lons)):
        raise ValueError(f"Terrain shape mismatch: Z{Z.shape} vs ({len(lats)}, {len(lons)})")
    
    # True where elevation > 0 (land), False where elevation <= 0 (sea)
    mask = Z > 0.0
    return mask


def mask_50km(lats: np.ndarray, lons: np.ndarray, 
              center_lat: float, center_lon: float, 
              radius_km: float = 50.0) -> np.ndarray:
    """
    Create a boolean mask for locations within a specified radius from a center point.
    
    The mask is True where distance from center <= radius_km, False otherwise.
    Uses haversine distance calculation for accurate great-circle distances.
    
    Parameters:
    -----------
    lats : np.ndarray
        1D array of latitude values (degrees)
    lons : np.ndarray
        1D array of longitude values (degrees)
    center_lat : float
        Latitude of center point (degrees)
    center_lon : float
        Longitude of center point (degrees)
    radius_km : float, optional
        Maximum distance in kilometers (default: 50.0)
    
    Returns:
    --------
    np.ndarray
        Boolean array of shape (len(lats), len(lons))
        True = within radius (admissible), False = outside radius (excluded)
    """
    # Create meshgrid for all lat/lon combinations
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Calculate distance from center to each grid point
    # Vectorized haversine calculation
    R = 6371.0  # Earth radius in kilometers
    
    lat1_rad = np.radians(center_lat)
    lon1_rad = np.radians(center_lon)
    lat2_rad = np.radians(lat_grid)
    lon2_rad = np.radians(lon_grid)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distances_km = R * c
    
    # True where distance <= radius_km
    mask = distances_km <= radius_km
    return mask


def combine_masks(*masks: np.ndarray) -> np.ndarray:
    """
    Combine multiple boolean masks using logical AND.
    
    The result is True only where all input masks are True.
    All masks must have the same shape.
    
    Parameters:
    -----------
    *masks : np.ndarray
        Variable number of boolean arrays to combine
    
    Returns:
    --------
    np.ndarray
        Combined boolean array (logical AND of all masks)
    """
    if len(masks) == 0:
        raise ValueError("At least one mask must be provided")
    
    # Check all masks have the same shape
    shape = masks[0].shape
    for i, mask in enumerate(masks[1:], 1):
        if mask.shape != shape:
            raise ValueError(f"Mask {i} has shape {mask.shape}, expected {shape}")
    
    # Combine using logical AND
    result = masks[0].copy()
    for mask in masks[1:]:
        result = np.logical_and(result, mask)
    
    return result


def mask_french_territory(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Create a boolean mask for French territory only.
    
    Excludes Monaco and Italy. The mask is True where the location is in France,
    False for Monaco, Italy, or other non-French territories.
    
    Parameters:
    -----------
    lats : np.ndarray
        1D array of latitude values (degrees)
    lons : np.ndarray
        1D array of longitude values (degrees)
    
    Returns:
    --------
    np.ndarray
        Boolean array of shape (len(lats), len(lons))
        True = French territory (admissible), False = Monaco/Italy/excluded
    """
    # Create meshgrid for all lat/lon combinations
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Initialize mask: all points are French by default
    mask = np.ones((len(lats), len(lons)), dtype=bool)
    
    # Exclude Monaco (approximate bounding box)
    # Monaco is roughly: 43.72-43.75°N, 7.40-7.44°E
    monaco_lat_min = 43.72
    monaco_lat_max = 43.75
    monaco_lon_min = 7.40
    monaco_lon_max = 7.44
    
    monaco_mask = ((lat_grid >= monaco_lat_min) & (lat_grid <= monaco_lat_max) &
                   (lon_grid >= monaco_lon_min) & (lon_grid <= monaco_lon_max))
    mask[monaco_mask] = False
    
    # Exclude Italy (east of French-Italian border)
    # The border in this region is approximately at longitude 7.5-7.6°E
    # Using a conservative boundary: exclude everything east of 7.5°E
    italy_mask = lon_grid > 7.5
    mask[italy_mask] = False
    
    return mask
