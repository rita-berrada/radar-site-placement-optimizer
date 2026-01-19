"""
geo_utils.py

Centralized module for coordinate system management.
Handles the conversion from Geographic coordinates (Lat/Lon) 
to a Local Tangent Plane (ENU - East, North, Up) in meters.

This ensures all physical calculations (Slope, Distance, Coverage) 
share the same reference system.
"""

import numpy as np

# --- Project Constants ---
# Reference Point: Nice Côte d'Azur Airport
# This serves as the origin (0,0) for the local metric system.
REF_LAT = 43.6584
REF_LON = 7.2159

# Earth Radius (standard approximation for this latitude)
EARTH_RADIUS_M = 6371000.0

def load_and_convert_to_enu(npz_file_path):
    """
    Loads terrain data and performs the coordinate conversion.
    
    Applies a Local Flat Earth approximation suitable for the 
    50km project radius.
    
    Parameters:
    -----------
    npz_file_path : str
        Path to the .npz file containing 'lat', 'lon', and 'ter'.
        
    Returns:
    --------
    X_m : np.ndarray (1D or 2D)
        Distance in meters along the East-West axis (relative to Ref).
    Y_m : np.ndarray (1D or 2D)
        Distance in meters along the North-South axis (relative to Ref).
    Z : np.ndarray (2D)
        Altitude in meters (Elevation).
    lats : np.ndarray
        Original Latitudes (degrees).
    lons : np.ndarray
        Original Longitudes (degrees).
    """
    # 1. Load raw data
    data = np.load(npz_file_path)
    lats = data['lat']
    lons = data['lon']
    Z = data['ter']

    # 2. Compute conversion factors (Meters per Degree)
    # Convert reference latitude to radians for cosine calculation
    lat_ref_rad = np.radians(REF_LAT)
    
    # Latitude degrees are roughly constant (~111km)
    meters_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
    
    # Longitude degrees vary with latitude (getting smaller towards poles)
    # We use the cosine of the reference latitude for the local projection
    meters_per_deg_lon = (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(lat_ref_rad)

    # 3. Generate Metric Grids (ENU System)
    # Calculate distance from the reference point
    Y_m = (lats - REF_LAT) * meters_per_deg_lat  # North distance
    X_m = (lons - REF_LON) * meters_per_deg_lon  # East distance
    
    return X_m, Y_m, Z, lats, lons