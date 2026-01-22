"""
roads_masks.py

Implements PROJECT_REQ_05: Road proximity constraint.
Adapted to work with the centralized ENU metric coordinate system.

UPDATE: Added a safety buffer (default 100m) to the standard 500m requirement.
Total acceptance radius = 500m + 100m = 600m.
"""

import numpy as np
import json
from typing import List, Tuple

# We need these constants to convert the Road GPS points into Local Meters
from geo_utils_earth_curvature import REF_LAT, REF_LON, EARTH_RADIUS_M

def load_roads_and_convert_to_enu(geojson_file: str, major_roads_only: bool = True) -> List[List[Tuple[float, float]]]:
    """
    Load road network data from GeoJSON AND convert it to ENU (Meters).
    """
    try:
        with open(geojson_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"   [Error] Could not load roads file {geojson_file}: {e}")
        return []
    
    roads_enu = []
    
    # Pre-calculate conversion factors (Same logic as geo_utils)
    lat_ref_rad = np.radians(REF_LAT)
    meters_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
    meters_per_deg_lon = (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(lat_ref_rad)
    
    major_road_types = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary'}
    
    # Helper to process a list of [lon, lat] and convert to [x, y]
    def process_coords(coords_list):
        road_metric = []
        for pt in coords_list:
            if len(pt) < 2: continue
            lon, lat = pt[0], pt[1]
            # Convert to Meters relative to Reference
            y_m = (lat - REF_LAT) * meters_per_deg_lat
            x_m = (lon - REF_LON) * meters_per_deg_lon
            road_metric.append((x_m, y_m))
        return road_metric

    # Handle GeoJSON FeatureCollection
    if 'features' in data:
        for feature in data['features']:
            geom = feature.get('geometry', {})
            if not geom: continue
            
            if geom['type'] == 'LineString':
                if major_roads_only:
                    props = feature.get('properties', {})
                    road_type = props.get('highway', '').lower()
                    if road_type not in major_road_types:
                        continue
                
                # Convert raw coords to Meters
                raw_coords = geom['coordinates']
                converted = process_coords(raw_coords)
                if converted:
                    roads_enu.append(converted)
                    
            elif geom['type'] == 'MultiLineString':
                # Handle MultiLineString just in case
                if major_roads_only:
                    props = feature.get('properties', {})
                    road_type = props.get('highway', '').lower()
                    if road_type not in major_road_types:
                        continue
                
                for line_coords in geom['coordinates']:
                    converted = process_coords(line_coords)
                    if converted:
                        roads_enu.append(converted)
    
    # Handle OSM JSON format (fallback)
    elif 'elements' in data:
        for element in data['elements']:
            if element.get('type') == 'way' and 'geometry' in element:
                if major_roads_only:
                    tags = element.get('tags', {})
                    road_type = tags.get('highway', '').lower()
                    if road_type not in major_road_types:
                        continue
                
                # Extract lon/lat and convert
                raw_coords = [[n['lon'], n['lat']] for n in element['geometry']]
                converted = process_coords(raw_coords)
                if converted:
                    roads_enu.append(converted)
    
    return roads_enu


def mask_roads_proximity_fast(X_grid: np.ndarray, Y_grid: np.ndarray,
                               roads_metric: List[List[Tuple[float, float]]],
                               max_distance_m: float = 500.0) -> np.ndarray:
    """
    Create a boolean mask for locations within specified distance of any road.
    Now works purely in METERS (Euclidean Distance).
    
    Parameters:
    -----------
    X_grid, Y_grid : np.ndarray (2D)
        Metric coordinate grids.
    roads_metric : List
        List of roads in meter coordinates [(x,y), ...].
    max_distance_m : float
        Maximum distance in meters (Constraint + Buffer).
    
    Returns:
    --------
    mask : np.ndarray (bool)
        True = within distance (Admissible).
    """
    # Initialize mask (False = Too far)
    mask = np.zeros(X_grid.shape, dtype=bool)
    
    if not roads_metric:
        print("   [Warn] No roads loaded. Mask will be all False.")
        return mask

    print(f"   Processing {len(roads_metric)} roads (Metric System, Limit={max_distance_m:.0f}m)...")
    
    # Optimization: Process roads in batches or just inform user
    road_count = 0
    total_roads = len(roads_metric)
    
    # Pre-calculate squared distance for speed
    dist_sq_limit = max_distance_m**2
    
    for road in roads_metric:
        road_count += 1
        if total_roads > 1000 and road_count % 1000 == 0:
            print(f"   ... processed {road_count}/{total_roads} roads")
        
        if len(road) < 2:
            continue
        
        # 1. Bounding Box Filter (in Meters) - Very fast pre-check
        # Create a bounding box around the road + max_distance
        road_xs = [pt[0] for pt in road]
        road_ys = [pt[1] for pt in road]
        
        x_min, x_max = min(road_xs) - max_distance_m, max(road_xs) + max_distance_m
        y_min, y_max = min(road_ys) - max_distance_m, max(road_ys) + max_distance_m
        
        # Identify grid indices inside this bounding box
        # We assume X_grid and Y_grid are meshgrids or regular grids
        in_road_box = ((X_grid >= x_min) & (X_grid <= x_max) & 
                       (Y_grid >= y_min) & (Y_grid <= y_max))
        
        # Skip if no grid points are in the box
        if not np.any(in_road_box):
            continue 
        
        # 2. Detailed Distance Check
        # We iterate over road segments or points
        # Sampling points is faster than checking every meter of road
        # Adaptive sampling: roughly every 50m of road length
        sample_step = 1 # Check every point by default
        
        for i in range(0, len(road), sample_step):
            rx, ry = road[i]
            
            # Further optimize: Only check points in the bounding box of this specific ROAD POINT
            sub_x_min, sub_x_max = rx - max_distance_m, rx + max_distance_m
            sub_y_min, sub_y_max = ry - max_distance_m, ry + max_distance_m
            
            # Refine the mask from the road-box to the point-box
            in_point_box = (in_road_box & 
                           (X_grid >= sub_x_min) & (X_grid <= sub_x_max) & 
                           (Y_grid >= sub_y_min) & (Y_grid <= sub_y_max))
            
            if not np.any(in_point_box):
                continue

            # Exact Euclidean Distance
            # Only compute where necessary
            dist_sq = (X_grid[in_point_box] - rx)**2 + (Y_grid[in_point_box] - ry)**2
            
            within_dist = dist_sq <= dist_sq_limit
            
            # Update the main mask (Logical OR)
            # We map the subset back to the full array
            full_indices = np.where(in_point_box)
            mask[full_indices] = np.logical_or(mask[full_indices], within_dist)
    
    return mask


def mask_roads_from_geojson(X_grid: np.ndarray, Y_grid: np.ndarray,
                            geojson_file: str = 'roads_nice_50km.geojson',
                            max_distance_m: float = 500.0,
                            buffer_m: float = 100.0,
                            major_roads_only: bool = True) -> np.ndarray:
    """
    Entry point. Loads GeoJSON, converts to meters, computes mask.
    
    Parameters:
    -----------
    max_distance_m : float
        The base constraint requirement (default 500m).
    buffer_m : float
        Additional tolerance buffer (default 100m).
        Total acceptance distance = max_distance_m + buffer_m.
    """
    # 1. Load and Convert to Meters
    roads_m = load_roads_and_convert_to_enu(geojson_file, major_roads_only)
    
    # 2. Compute Total Allowed Distance
    total_allowed_dist = max_distance_m + buffer_m
    print(f"   [Constraint] Road Proximity: Base {max_distance_m}m + Buffer {buffer_m}m = {total_allowed_dist}m")
    
    # 3. Compute Mask using Metric Grid
    return mask_roads_proximity_fast(X_grid, Y_grid, roads_m, total_allowed_dist)