"""
protected_areas_masks.py

Handles exclusion zones based on GeoJSON polygons (National Parks, Forests, etc.).
Adapted to work with the centralized ENU metric coordinate system.
"""

import json
import numpy as np
from matplotlib.path import Path

# Constants for coordinate conversion
from geo_utils_earth_curvature import REF_LAT, REF_LON, EARTH_RADIUS_M

def mask_protected_areas_from_geojson(X_grid: np.ndarray, Y_grid: np.ndarray, geojson_file: str) -> np.ndarray:
    """
    Creates a boolean mask identifying areas OUTSIDE protected zones.
    
    Parameters:
    -----------
    X_grid, Y_grid : np.ndarray (2D)
        Metric coordinates of the terrain grid.
    geojson_file : str
        Path to the .geojson file containing polygons.
    
    Returns:
    --------
    mask : np.ndarray (bool)
        True = Admissible (Outside protected areas)
        False = Forbidden (Inside protected areas)
    """
    # 1. Initialize mask (All True by default)
    # Working on flattened array for Path efficiency
    mask_flat = np.ones(X_grid.size, dtype=bool)
    
    # Prepare grid points for vectorized check: list of (x, y)
    points = np.vstack((X_grid.flatten(), Y_grid.flatten())).T
    
    # Grid limits for optimization (Bounding Box in Meters)
    min_x, max_x = np.min(X_grid), np.max(X_grid)
    min_y, max_y = np.min(Y_grid), np.max(Y_grid)
    
    # 2. Compute Conversion Factors (Degrees -> Meters)
    lat_ref_rad = np.radians(REF_LAT)
    m_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
    m_per_deg_lon = (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(lat_ref_rad)
    
    # 3. Load GeoJSON
    try:
        with open(geojson_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"   [Error] {geojson_file} not found. Skipping protected areas.")
        return mask_flat.reshape(X_grid.shape)

    print(f"   ... Processing protected areas from {geojson_file} (Metric System) ...")
    count_polygons = 0
    
    features = data.get('features', [])
    
    # 4. Iterate over features
    for feature in features:
        geom = feature.get('geometry', {})
        geom_type = geom.get('type')
        
        # Extract polygon coordinates
        polygons_coords = []
        if geom_type == 'Polygon':
            polygons_coords.append(geom['coordinates'][0])
        elif geom_type == 'MultiPolygon':
            for poly in geom['coordinates']:
                polygons_coords.append(poly[0])
        else:
            continue
            
        # 5. Apply exclusion
        for poly_coord in polygons_coords:
            # Convert Polygon vertices to Meters
            poly_arr_deg = np.array(poly_coord)
            lons = poly_arr_deg[:, 0]
            lats = poly_arr_deg[:, 1]
            
            # Conversion to ENU
            xs = (lons - REF_LON) * m_per_deg_lon
            ys = (lats - REF_LAT) * m_per_deg_lat
            
            # Stack into (N, 2) array for Path
            poly_arr_m = np.column_stack((xs, ys))
            
            # Optimization: Bounding box check (in Meters)
            if (np.max(xs) < min_x or np.min(xs) > max_x or
                np.max(ys) < min_y or np.min(ys) > max_y):
                continue

            # Check points inside polygon
            path_obj = Path(poly_arr_m)
            is_inside = path_obj.contains_points(points)
            
            # Update mask: Inside = False (Forbidden)
            mask_flat[is_inside] = False
            count_polygons += 1

    print(f"   -> Applied exclusion for {count_polygons} protected zones.")
    
    # Return reshaped 2D mask
    return mask_flat.reshape(X_grid.shape)