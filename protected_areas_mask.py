"""
protected_areas_masks.py

Handles exclusion zones based on GeoJSON polygons (National Parks, Forests, etc.).
Uses matplotlib.path to determine if grid points fall inside forbidden polygons.
"""

import json
import numpy as np
from matplotlib.path import Path

def mask_protected_areas_from_geojson(lats: np.ndarray, lons: np.ndarray, geojson_file: str) -> np.ndarray:
    """
    Creates a boolean mask identifying areas OUTSIDE protected zones.
    
    Parameters:
    -----------
    lats, lons : 1D arrays of coordinates.
    geojson_file : Path to the .geojson file containing polygons.
    
    Returns:
    --------
    mask : np.ndarray (bool)
        True = Admissible (Outside protected areas)
        False = Forbidden (Inside protected areas)
    """
    # 1. Initialize mask (All True by default)
    mask = np.ones((len(lats), len(lons)), dtype=bool)
    
    # Prepare grid points for vectorized check
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.vstack((lon_grid.flatten(), lat_grid.flatten())).T
    
    # 2. Load GeoJSON
    try:
        with open(geojson_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {geojson_file} not found.")
        return mask

    print(f"   ... Processing protected areas from {geojson_file} ...")
    count_polygons = 0
    
    # 3. Iterate over features
    for feature in data.get('features', []):
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
            
        # 4. Apply exclusion
        for poly_coord in polygons_coords:
            # Create Path object
            path_obj = Path(poly_coord)
            
            # Optimization: Bounding box check
            poly_arr = np.array(poly_coord)
            if (np.max(poly_arr[:,0]) < np.min(lons) or np.min(poly_arr[:,0]) > np.max(lons) or
                np.max(poly_arr[:,1]) < np.min(lats) or np.min(poly_arr[:,1]) > np.max(lats)):
                continue

            # Check points inside polygon
            is_inside = path_obj.contains_points(points)
            
            # Update mask: Inside = False (Forbidden)
            mask[is_inside.reshape(mask.shape)] = False
            count_polygons += 1

    print(f"   -> Applied exclusion for {count_polygons} zones.")
    return mask