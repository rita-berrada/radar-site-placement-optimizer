"""
Coverage Analysis Module


This module provides functions to compute radar coverage maps for different flight levels.
Each coverage map is a 2D boolean array indicating visibility (True) or blockage (False)
at a specific altitude.
"""


import numpy as np
from typing import Dict, List, Optional, Callable
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
   progress_callback: Optional[Callable] = None,          # kept for compatibility
   point_progress_callback: Optional[Callable] = None     # callback(k, total, pct)
) -> np.ndarray:
   """
   Compute coverage map for a single flight level.


   This version removes the explicit double loop and iterates over a flattened
   meshgrid, then reshapes back to (len(lats), len(lons)).
   It still calls los_visible once per grid point (cannot be purely vectorized).


   Returns:
       coverage_map: (len(lats), len(lons)) boolean array
   """
   target_alt_m_msl = fl_to_m(flight_level)


   # Build the grid exactly matching original indexing:
   # coverage_map[i, j] corresponds to lats[i], lons[j]
   lon_grid, lat_grid = np.meshgrid(lons, lats)  # shape: (N, M)
   lat_flat = lat_grid.ravel()
   lon_flat = lon_grid.ravel()


   total = lat_flat.size
   progress_interval = max(1, total // 50)


   def gen_visibility():
       for k, (lat, lon) in enumerate(zip(lat_flat, lon_flat), start=1):
           v = los_visible(
               radar_lat, radar_lon, radar_height_agl_m,
               float(lat), float(lon), float(target_alt_m_msl),
               lats, lons, Z,
               n_samples=n_samples,
               margin_m=margin_m
           )


           if point_progress_callback and (k % progress_interval == 0 or k == total):
               pct = 100.0 * k / total
               point_progress_callback(k, total, pct)


           yield v


   coverage_flat = np.fromiter(gen_visibility(), dtype=np.bool_, count=total)
   coverage_map = coverage_flat.reshape(len(lats), len(lons))
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


