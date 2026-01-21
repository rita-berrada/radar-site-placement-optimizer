"""
roads_masks.py

Implements PROJECT_REQ_05: Road proximity constraint (< 500m).
Adapted to work with the centralized ENU metric coordinate system.
"""

import numpy as np
import json
from typing import List, Tuple

# We need these constants to convert the Road GPS points into Local Meters
from geo_utils_earth_curvature import REF_LAT, REF_LON, EARTH_RADIUS_M

def load_roads_and_convert_to_enu(geojson_file: str, major_roads_only: bool = True) -> List[List[Tuple[float, float]]]:
    """
    Load road network data from GeoJSON AND convert it to ENU (Meters).
    
    Parameters:
    -----------
    geojson_file : str
        Path to the GeoJSON file.
    major_roads_only : bool
        If True, only load major roads.
    
    Returns:
    --------
    List[List[Tuple[float, float]]]
        List of roads, where each road is a list of (X_m, Y_m) coordinates.
    """
    with open(geojson_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
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
            lon, lat = pt[0], pt[1]
            # Convert to Meters relative to Reference
            y_m = (lat - REF_LAT) * meters_per_deg_lat
            x_m = (lon - REF_LON) * meters_per_deg_lon
            road_metric.append((x_m, y_m))
        return road_metric

    # Handle GeoJSON FeatureCollection
    if 'features' in data:
        for feature in data['features']:
            if feature['geometry']['type'] == 'LineString':
                if major_roads_only:
                    props = feature.get('properties', {})
                    road_type = props.get('highway', '').lower()
                    if road_type not in major_road_types:
                        continue
                
                # Convert raw coords to Meters
                raw_coords = feature['geometry']['coordinates']
                roads_enu.append(process_coords(raw_coords))
    
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
                roads_enu.append(process_coords(raw_coords))
    
    return roads_enu


def mask_roads_proximity_fast(X_grid: np.ndarray, Y_grid: np.ndarray,
                               roads_metric: List[List[Tuple[float, float]]],
                               max_distance_m: float = 500.0) -> np.ndarray:
    """
    Create a boolean mask for locations within specified distance of any road.
    Now works purely in METERS (Euclidean Distance).
    
    Parameters:
    -----------
    X_grid : np.ndarray (2D)
        X coordinates of the terrain grid in meters.
    Y_grid : np.ndarray (2D)
        Y coordinates of the terrain grid in meters.
    roads_metric : List
        List of roads in meter coordinates [(x,y), ...].
    max_distance_m : float
        Maximum distance in meters.
    
    Returns:
    --------
    mask : np.ndarray (bool)
        True = within distance (Admissible).
    """
    # Initialize mask
    mask = np.zeros(X_grid.shape, dtype=bool)
    
    print(f"   Processing {len(roads_metric)} roads (Metric System)...")
    
    road_count = 0
    for road in roads_metric:
        road_count += 1
        if road_count % 1000 == 0:
            print(f"   ... road {road_count}/{len(roads_metric)}")
        
        if len(road) < 2:
            continue
        
        # 1. Bounding Box Filter (in Meters) - Very fast
        road_xs = [pt[0] for pt in road]
        road_ys = [pt[1] for pt in road]
        
        # Add buffer to bounding box
        x_min, x_max = min(road_xs) - max_distance_m, max(road_xs) + max_distance_m
        y_min, y_max = min(road_ys) - max_distance_m, max(road_ys) + max_distance_m
        
        # Check which grid points are inside this box
        in_road_box = ((X_grid >= x_min) & (X_grid <= x_max) & 
                       (Y_grid >= y_min) & (Y_grid <= y_max))
        
        if not np.any(in_road_box):
            continue 
        
        # 2. Detailed Distance Check (Pythagoras)
        # Sample points to speed up (keep 1 point every ~10)
        sample_step = max(1, len(road) // 10)
        
        for i in range(0, len(road), sample_step):
            rx, ry = road[i]
            
            # Sub-box for this specific point (optimization)
            sub_x_min, sub_x_max = rx - max_distance_m, rx + max_distance_m
            sub_y_min, sub_y_max = ry - max_distance_m, ry + max_distance_m
            
            in_point_box = (in_road_box & 
                           (X_grid >= sub_x_min) & (X_grid <= sub_x_max) & 
                           (Y_grid >= sub_y_min) & (Y_grid <= sub_y_max))
            
            if not np.any(in_point_box):
                continue

            # Exact Euclidean Distance: sqrt(dx^2 + dy^2)
            # We only compute this on the small subset of points
            dist_sq = (X_grid[in_point_box] - rx)**2 + (Y_grid[in_point_box] - ry)**2
            
            # Compare squared distance (faster than taking sqrt)
            within_dist = dist_sq <= (max_distance_m**2)
            
            # Update the main mask
            # We need to map the 'within_dist' result back to the full mask
            current_indices = np.where(in_point_box)
            mask[current_indices] = np.logical_or(mask[current_indices], within_dist)
    
    return mask


def mask_roads_from_geojson(X_grid: np.ndarray, Y_grid: np.ndarray,
                            geojson_file: str = 'roads_nice_50km.geojson',
                            max_distance_m: float = 500.0,
                            major_roads_only: bool = True) -> np.ndarray:
    """
    Entry point. Loads GeoJSON, converts to meters, computes mask.
    """
    # 1. Load and Convert to Meters
    roads_m = load_roads_and_convert_to_enu(geojson_file, major_roads_only)
    
    # 2. Compute Mask using Metric Grid
    return mask_roads_proximity_fast(X_grid, Y_grid, roads_m, max_distance_m)