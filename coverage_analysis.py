"""
Coverage Analysis Module

This module provides functions to compute radar coverage maps for different flight levels.
Each coverage map is a 2D boolean array indicating visibility (True) or blockage (False)
at a specific altitude.
"""

import numpy as np
from typing import Dict, List, Optional
from LOS import los_visible, fl_to_m


def compute_coverage_map(
    radar_lat: float,
    radar_lon: float,
    radar_height_agl_m: float,
    flight_level: float,
    lats: np.ndarray,
    lons: np.ndarray,
    Z: np.ndarray,
    n_samples: int = 400,
    margin_m: float = 0.0,
    progress_callback: Optional[callable] = None,
    point_progress_callback: Optional[callable] = None
) -> np.ndarray:
    """
    Compute coverage map for a single flight level.
    
    Parameters:
    -----------
    radar_lat : float
        Radar latitude (degrees)
    radar_lon : float
        Radar longitude (degrees)
    radar_height_agl_m : float
        Radar height above ground level (meters)
    flight_level : float
        Flight level (e.g., 5, 10, 20, 50, 100, 200, 300, 400)
    lats : np.ndarray
        1D array of latitude values
    lons : np.ndarray
        1D array of longitude values
    Z : np.ndarray
        2D terrain elevation array with shape (len(lats), len(lons))
    n_samples : int, optional
        Number of samples along LOS path (default: 400)
    margin_m : float, optional
        Safety margin in meters (default: 0.0)
    progress_callback : callable, optional
        (Deprecated - use point_progress_callback instead)
    point_progress_callback : callable, optional
        Callback function for progress updates: callback(current, total, percentage)
    
    Returns:
    --------
    np.ndarray
        2D boolean array with shape (len(lats), len(lons))
        True = visible, False = blocked
    """
    # Convert flight level to altitude in meters
    target_alt_m_msl = fl_to_m(flight_level)
    
    # Initialize coverage map
    coverage_map = np.zeros((len(lats), len(lons)), dtype=bool)
    
    # Total number of grid points
    total_points = len(lats) * len(lons)
    current_point = 0
    
    # Progress reporting interval (print every N points)
    progress_interval = max(1, total_points // 50)  # Report ~50 times
    
    # Loop over all grid points
    for i in range(len(lats)):
        for j in range(len(lons)):
            target_lat = lats[i]
            target_lon = lons[j]
            
            # Check LOS visibility
            is_visible = los_visible(
                radar_lat, radar_lon, radar_height_agl_m,
                target_lat, target_lon, target_alt_m_msl,
                lats, lons, Z,
                n_samples=n_samples, margin_m=margin_m
            )
            
            coverage_map[i, j] = is_visible
            
            # Progress callback (simplified - just report periodically)
            current_point += 1
            if point_progress_callback and (current_point % progress_interval == 0 or current_point == total_points):
                pct = (current_point / total_points) * 100
                point_progress_callback(current_point, total_points, pct)
    
    return coverage_map


def compute_all_coverage_maps(
    radar_lat: float,
    radar_lon: float,
    radar_height_agl_m: float,
    flight_levels: List[float],
    lats: np.ndarray,
    lons: np.ndarray,
    Z: np.ndarray,
    n_samples: int = 400,
    margin_m: float = 0.0,
    progress_callback: Optional[callable] = None
) -> Dict[float, np.ndarray]:
    """
    Compute coverage maps for multiple flight levels.
    
    Parameters:
    -----------
    radar_lat : float
        Radar latitude (degrees)
    radar_lon : float
        Radar longitude (degrees)
    radar_height_agl_m : float
        Radar height above ground level (meters)
    flight_levels : List[float]
        List of flight levels (e.g., [5, 10, 20, 50, 100, 200, 300, 400])
    lats : np.ndarray
        1D array of latitude values
    lons : np.ndarray
        1D array of longitude values
    Z : np.ndarray
        2D terrain elevation array with shape (len(lats), len(lons))
    n_samples : int, optional
        Number of samples along LOS path (default: 400)
    margin_m : float, optional
        Safety margin in meters (default: 0.0)
    progress_callback : callable, optional
        Callback function for progress updates: callback(flight_level, current_fl, total_fl)
    
    Returns:
    --------
    Dict[float, np.ndarray]
        Dictionary mapping flight level to coverage map array
        Keys are flight levels, values are 2D boolean arrays
    """
    coverage_maps = {}
    
    for idx, flight_level in enumerate(flight_levels):
        print(f"  → Starting FL{flight_level}...", end='', flush=True)
        
        coverage_map = compute_coverage_map(
            radar_lat, radar_lon, radar_height_agl_m,
            flight_level, lats, lons, Z,
            n_samples=n_samples, margin_m=margin_m
        )
        
        # Report completion
        if progress_callback:
            progress_callback(flight_level, idx + 1, len(flight_levels))
        else:
            print(f"  ✓ FL{flight_level:3.0f} complete ({idx + 1}/{len(flight_levels)})")
        
        coverage_maps[flight_level] = coverage_map
    
    return coverage_maps
