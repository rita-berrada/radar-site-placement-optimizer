"""
mask_residential.py

Handles exclusion of residential areas based on GeoJSON polygons.
Uses matplotlib.path to determine if grid points fall inside residential zones.
"""

import json
import numpy as np
from matplotlib.path import Path

def mask_residential_from_geojson(lats: np.ndarray, lons: np.ndarray, geojson_file: str) -> np.ndarray:
    """
    Creates a boolean mask identifying areas OUTSIDE residential zones.
    
    Returns:
    --------
    mask : np.ndarray (bool)
        True = Admissible (Outside residential areas)
        False = Forbidden (Inside residential areas)
    """
    # 1. Initialize mask (All True by default)
    mask = np.ones((len(lats), len(lons)), dtype=bool)
    
    # Prepare grid points for vectorized check
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.vstack((lon_grid.flatten(), lat_grid.flatten())).T
    
    # Limits for optimization
    min_lat, max_lat = np.min(lats), np.max(lats)
    min_lon, max_lon = np.min(lons), np.max(lons)

    # 2. Load GeoJSON
    try:
        with open(geojson_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {geojson_file} not found.")
        return mask

    print(f"   ... Processing residential areas from {geojson_file} ...")
    
    features = data.get('features', [])
    total_features = len(features)
    count_excluded = 0
    
    # 3. Iterate over features
    for i, feature in enumerate(features):
        geom = feature.get('geometry', {})
        if not geom: continue
        
        geom_type = geom.get('type')
        polygons_coords = []
        
        # Extract coordinates based on type
        if geom_type == 'Polygon':
            polygons_coords.append(geom['coordinates'][0])
        elif geom_type == 'MultiPolygon':
            for poly in geom['coordinates']:
                polygons_coords.append(poly[0])
        else:
            continue
            
        # 4. Apply exclusion
        for poly_coord in polygons_coords:
            poly_arr = np.array(poly_coord)
            
            # Optimization: Skip if polygon is completely outside our map
            if (np.max(poly_arr[:,0]) < min_lon or np.min(poly_arr[:,0]) > max_lon or
                np.max(poly_arr[:,1]) < min_lat or np.min(poly_arr[:,1]) > max_lat):
                continue

            # Check points inside polygon
            path_obj = Path(poly_coord)
            is_inside = path_obj.contains_points(points)
            
            # Update mask: Inside = False (Forbidden)
            # We use logical AND to accumulate constraints
            current_mask_flat = mask.flatten()
            current_mask_flat[is_inside] = False
            mask = current_mask_flat.reshape(mask.shape)
            count_excluded += 1
            
        # Progress log for large files
        if total_features > 1000 and (i+1) % 1000 == 0:
             print(f"       Processed {i+1}/{total_features} zones...")

    print(f"   -> Applied exclusion for {count_excluded} residential polygons.")
    return mask