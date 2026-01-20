"""
buildings_masks.py (Optimized for Merged Zones)

Handles exclusion of buildings or residential zones with a safety buffer.
Uses Shapely to 'inflate' the polygon by the safety radius, 
then checks if points fall inside this inflated zone.
"""

import json
import numpy as np
from shapely.geometry import shape, Point
from shapely.ops import unary_union
import matplotlib.path as mpath

def mask_buildings_from_geojson(lats: np.ndarray, lons: np.ndarray, geojson_file: str, radius_m: float = 500.0) -> np.ndarray:
    """
    Creates a mask excluding areas inside polygons + a safety buffer (radius).
    
    Parameters:
    -----------
    radius_m : float
        Safety distance in meters. 
        Note: Since coordinates are in degrees, we perform an approximate conversion.
    """
    print(f"   ... Processing exclusion zones from {geojson_file} with {radius_m}m buffer ...")

    # 1. Initialize mask (True = Allowed)
    mask = np.ones((len(lats), len(lons)), dtype=bool)
    
    # 2. Convert meters to degrees (Approximate for Nice region)
    # 1 deg lat ~= 111 km. 1 deg lon at 43°N ~= 81 km.
    # We take a safe approximation: 1 degree ~ 100,000 meters
    buffer_deg = radius_m / 100000.0
    
    # 3. Load and Buffer Geometry
    try:
        with open(geojson_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {geojson_file} not found.")
        return mask

    # Convert features to Shapely polygons and apply buffer
    polys = []
    for feature in data.get('features', []):
        if feature.get('geometry'):
            geom = shape(feature['geometry'])
            # Buffer (inflate) the geometry
            buffered_geom = geom.buffer(buffer_deg)
            polys.append(buffered_geom)

    if not polys:
        print("   [!] No valid polygons found.")
        return mask

    # Merge all buffered zones into one (optimization)
    print(f"   -> Merging and buffering {len(polys)} zones...")
    combined_area = unary_union(polys)

    # 4. Create Grid Points
    # Optimization: Check bounding box first
    min_lon, min_lat, max_lon, max_lat = combined_area.bounds
    
    # Grid generation
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.vstack((lon_grid.flatten(), lat_grid.flatten())).T

    # 5. Fast Point-in-Polygon Check
    # We convert the complex Shapely shape back to a Matplotlib path for vectorization speed
    if combined_area.geom_type == 'Polygon':
        zones = [combined_area]
    elif combined_area.geom_type == 'MultiPolygon':
        zones = list(combined_area.geoms)
    else:
        zones = []

    count_excluded = 0
    print(f"   -> Applying mask on grid...")
    
    for poly in zones:
        # Check bounds to skip useless checks
        p_minx, p_miny, p_maxx, p_maxy = poly.bounds
        if (p_maxx < np.min(lons) or p_minx > np.max(lons) or 
            p_maxy < np.min(lats) or p_miny > np.max(lats)):
            continue

        # Extract exterior coordinates
        exterior_coords = np.array(poly.exterior.coords)
        path = mpath.Path(exterior_coords)
        
        # Check points
        is_inside = path.contains_points(points)
        
        if np.any(is_inside):
            mask.flat[is_inside] = False
            count_excluded += 1

    print(f"   -> Exclusion applied. Safety buffer of ~{radius_m}m included.")
    return mask