"""
mask_residential.py

Handles exclusion of residential areas based on GeoJSON polygons.
Adapted to work with the centralized ENU metric coordinate system.
"""

import json
import numpy as np
from matplotlib.path import Path

# We need constants to convert the Polygon vertices into Meters
from geo_utils_earth_curvature import REF_LAT, REF_LON, EARTH_RADIUS_M

def mask_residential_from_geojson(X_grid: np.ndarray, Y_grid: np.ndarray, geojson_file: str) -> np.ndarray:
    """
    Creates a boolean mask identifying areas OUTSIDE residential zones.
    
    Parameters:
    -----------
    X_grid, Y_grid : np.ndarray (2D)
        Metric coordinates of the terrain.
    geojson_file : str
        Path to the GeoJSON file containing residential polygons.
        
    Returns:
    --------
    mask : np.ndarray (bool)
        True = Admissible (Outside residential areas)
        False = Forbidden (Inside residential areas)
    """
    # 1. Initialize mask (All True by default)
    # We work on the flattened array for efficiency with matplotlib.Path
    mask_flat = np.ones(X_grid.size, dtype=bool)
    
    # Prepare grid points: list of (x, y) pairs
    # We flatten the grids to 1D arrays
    points = np.vstack((X_grid.flatten(), Y_grid.flatten())).T
    
    # Grid limits for optimization (Bounding Box of the map)
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
        print(f"   [Error] {geojson_file} not found. Returning empty mask.")
        return mask_flat.reshape(X_grid.shape)

    print(f"   ... Processing residential areas from {geojson_file} (Metric Conversion)...")
    
    features = data.get('features', [])
    total_features = len(features)
    count_excluded = 0
    
    # 4. Iterate over features
    for i, feature in enumerate(features):
        geom = feature.get('geometry', {})
        if not geom: continue
        
        geom_type = geom.get('type')
        polygons_coords = []
        
        # Extract coordinates based on type (Polygon or MultiPolygon)
        # Note: GeoJSON coords are [lon, lat]
        if geom_type == 'Polygon':
            polygons_coords.append(geom['coordinates'][0])
        elif geom_type == 'MultiPolygon':
            for poly in geom['coordinates']:
                polygons_coords.append(poly[0])
        else:
            continue
            
        # 5. Apply exclusion
        for poly_coord in polygons_coords:
            # Convert this polygon to Meters
            # poly_coord is a list of [lon, lat]
            poly_arr_deg = np.array(poly_coord)
            lons = poly_arr_deg[:, 0]
            lats = poly_arr_deg[:, 1]
            
            # Conversion to ENU
            xs = (lons - REF_LON) * m_per_deg_lon
            ys = (lats - REF_LAT) * m_per_deg_lat
            
            # Stack into (N, 2) array for Path
            poly_arr_m = np.column_stack((xs, ys))
            
            # Optimization: Skip if polygon is completely outside our 50km map
            # Use the Metric bounding box
            if (np.max(xs) < min_x or np.min(xs) > max_x or
                np.max(ys) < min_y or np.min(ys) > max_y):
                continue

            # Check points inside polygon (using Matplotlib Path)
            path_obj = Path(poly_arr_m)
            
            # contains_points returns True if point is INSIDE
            is_inside = path_obj.contains_points(points)
            
            # Update mask: Inside = False (Forbidden)
            # Logical AND: Keep existing False, turn new Insides to False
            # equivalent to: mask_flat = mask_flat & (~is_inside)
            # But modifying via index is faster:
            mask_flat[is_inside] = False
            
            count_excluded += 1
            
        # Progress log
        if total_features > 1000 and (i+1) % 1000 == 0:
             print(f"       Processed {i+1}/{total_features} zones...")

    print(f"   -> Applied exclusion for {count_excluded} residential polygons.")
    
    # Reshape back to 2D Grid
    return mask_flat.reshape(X_grid.shape)