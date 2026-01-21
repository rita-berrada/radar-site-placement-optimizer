"""
Site Location Masks Module

This module implements boolean geographical masks for Lot 2.
Adapted to work with the centralized ENU metric coordinate system.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

# We need constants to convert the political borders (Monaco/Italy) into Meters
from geo_utils_earth_curvature import REF_LAT, REF_LON, EARTH_RADIUS_M

def mask_land(Z_raw: np.ndarray) -> np.ndarray:
    """
    Create a boolean mask for onshore areas (land).
    
    IMPORTANT: This should use the RAW Terrain Z (from the file), 
    NOT the curvature-corrected Z. Sea level is always 0 in raw data.
    
    Parameters:
    -----------
    Z_raw : np.ndarray (2D)
        Raw terrain elevation in meters.
    
    Returns:
    --------
    mask : np.ndarray (bool)
        True = land (> 0), False = sea (<= 0).
    """
    # Simple check: Land is where elevation is positive
    return Z_raw > 0.0


def mask_coastline_buffer(X_grid: np.ndarray, Y_grid: np.ndarray, Z_raw: np.ndarray, buffer_m: float = 100.0) -> np.ndarray:
    """
    Create a mask excluding the seaside buffer zone.
    
    Parameters:
    -----------
    X_grid, Y_grid : np.ndarray
        Metric grids (used to calculate pixel resolution).
    Z_raw : np.ndarray
        Raw elevation (to identify the sea).
    buffer_m : float
        Buffer distance in meters.
        
    Returns:
    --------
    mask : np.ndarray (bool)
        True = Admissible (Inland), False = Too close to sea.
    """
    # 1. Base Land Mask
    land_mask = Z_raw > 0
    
    # 2. Compute Grid Resolution automatically
    # We look at the step size between two adjacent pixels
    if X_grid.ndim == 2:
        dx = np.abs(X_grid[0, 1] - X_grid[0, 0])
        dy = np.abs(Y_grid[1, 0] - Y_grid[0, 0])
    else:
        # If 1D axes are passed
        dx = np.abs(X_grid[1] - X_grid[0])
        dy = np.abs(Y_grid[1] - Y_grid[0])
        
    mean_resolution = (dx + dy) / 2.0
    
    # 3. Compute distance from Sea (Distance Transform)
    # edt computes distance to the nearest zero (Sea) for each Land pixel
    dist_in_pixels = distance_transform_edt(land_mask)
    dist_in_meters = dist_in_pixels * mean_resolution
    
    # 4. Apply Buffer
    return dist_in_meters > buffer_m


def mask_50km(X_grid: np.ndarray, Y_grid: np.ndarray, radius_km: float = 50.0) -> np.ndarray:
    """
    Create a boolean mask for locations within radius.
    
    Since coordinates are relative to Nice Airport (0,0), 
    calculation is purely Euclidean.
    
    Parameters:
    -----------
    X_grid, Y_grid : np.ndarray
        Metric coordinates.
    radius_km : float
        Radius in km.
    
    Returns:
    --------
    mask : np.ndarray (bool)
        True = within radius.
    """
    # Convert radius to meters
    radius_m = radius_km * 1000.0
    
    # Simple Pythagoras: Dist^2 = X^2 + Y^2
    dist_sq = X_grid**2 + Y_grid**2
    
    # Compare with radius squared (faster)
    return dist_sq <= (radius_m**2)


def mask_french_territory(X_grid: np.ndarray, Y_grid: np.ndarray) -> np.ndarray:
    """
    Create a boolean mask for French territory only.
    Converts Lat/Lon borders of Monaco/Italy into local Metric borders.
    
    Parameters:
    -----------
    X_grid, Y_grid : np.ndarray
        Metric coordinates.
    
    Returns:
    --------
    mask : np.ndarray (bool)
        True = French territory.
    """
    # Initialize mask: all points are True by default
    mask = np.ones(X_grid.shape, dtype=bool)
    
    # --- Conversion Constants ---
    lat_ref_rad = np.radians(REF_LAT)
    m_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
    m_per_deg_lon = (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(lat_ref_rad)
    
    # Helper to convert Lat/Lon to Y/X
    def to_y(lat): return (lat - REF_LAT) * m_per_deg_lat
    def to_x(lon): return (lon - REF_LON) * m_per_deg_lon

    # 1. Exclude Monaco (Bounding Box)
    # Original: 43.72-43.75°N, 7.40-7.44°E
    monaco_y_min = to_y(43.72)
    monaco_y_max = to_y(43.75)
    monaco_x_min = to_x(7.40)
    monaco_x_max = to_x(7.44)
    
    monaco_mask = ((Y_grid >= monaco_y_min) & (Y_grid <= monaco_y_max) &
                   (X_grid >= monaco_x_min) & (X_grid <= monaco_x_max))
    
    mask[monaco_mask] = False
    
    # 2. Exclude Italy (East of border)
    # Original: East of 7.5°E
    italy_border_x = to_x(7.5)
    
    italy_mask = X_grid > italy_border_x
    mask[italy_mask] = False
    
    return mask


def combine_masks(*masks: np.ndarray) -> np.ndarray:
    """
    Combine multiple boolean masks using logical AND.
    Unchanged logic.
    """
    if len(masks) == 0:
        raise ValueError("At least one mask must be provided")
    
    result = masks[0].copy()
    for mask in masks[1:]:
        result = np.logical_and(result, mask)
    
    return result