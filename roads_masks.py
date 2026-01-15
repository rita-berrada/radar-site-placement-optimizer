"""
Roads Masks Module - ULTRA FAST VERSION

This module implements boolean geographical masks for road infrastructure constraint.
OPTIMIZED: Only processes major roads (motorways, trunk, primary, secondary roads)
to dramatically reduce computation time.
"""

import numpy as np
import json
from typing import List, Tuple, Dict, Optional


def load_roads_from_geojson(geojson_file: str, major_roads_only: bool = True) -> List[List[Tuple[float, float]]]:
    """
    Load road network data from GeoJSON file.
    
    Parameters:
    -----------
    geojson_file : str
        Path to the GeoJSON file containing road network data
    major_roads_only : bool, optional
        If True, only load major roads (motorway, trunk, primary, secondary)
        This dramatically speeds up computation (default: True)
    
    Returns:
    --------
    List[List[Tuple[float, float]]]
        List of roads, where each road is a list of (lon, lat) coordinate tuples
    """
    with open(geojson_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    roads = []
    
    # Major road types to keep (for faster computation)
    major_road_types = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary'}
    
    # Handle GeoJSON FeatureCollection format
    if 'features' in data:
        for feature in data['features']:
            if feature['geometry']['type'] == 'LineString':
                # Filter by road type if requested
                if major_roads_only:
                    props = feature.get('properties', {})
                    road_type = props.get('highway', '').lower()
                    if road_type not in major_road_types:
                        continue  # Skip minor roads
                
                coords = feature['geometry']['coordinates']
                road = [(c[0], c[1]) for c in coords]
                roads.append(road)
    
    # Handle OSM JSON format
    elif 'elements' in data:
        for element in data['elements']:
            if element.get('type') == 'way' and 'geometry' in element:
                # Filter by road type if requested
                if major_roads_only:
                    tags = element.get('tags', {})
                    road_type = tags.get('highway', '').lower()
                    if road_type not in major_road_types:
                        continue  # Skip minor roads
                
                coords = [(node['lon'], node['lat']) for node in element['geometry']]
                roads.append(coords)
    
    return roads


def mask_roads_proximity_fast(lats: np.ndarray, lons: np.ndarray,
                               roads: List[List[Tuple[float, float]]],
                               max_distance_m: float = 2000.0) -> np.ndarray:
    """
    Create a boolean mask for locations within specified distance of any road.
    
    ULTRA FAST VERSION: 
    - Works best with major roads only
    - Uses aggressive bounding box filtering
    - Samples fewer points per road
    
    Parameters:
    -----------
    lats : np.ndarray
        1D array of latitude values (degrees)
    lons : np.ndarray
        1D array of longitude values (degrees)
    roads : List[List[Tuple[float, float]]]
        List of roads, each road is list of (lon, lat) tuples
    max_distance_m : float, optional
        Maximum distance in meters (default: 2000.0 for major roads)
    
    Returns:
    --------
    np.ndarray
        Boolean array of shape (len(lats), len(lons))
        True = within distance of road (admissible)
        False = no road within distance (excluded)
    """
    # Create meshgrid for all lat/lon combinations
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Initialize mask as False (no roads nearby)
    mask = np.zeros((len(lats), len(lons)), dtype=bool)
    
    # Convert distance from meters to kilometers
    max_distance_km = max_distance_m / 1000.0
    
    # Approximate distance in degrees (for bounding box)
    buffer_deg = max_distance_km / 111.0 * 1.5  # Safety margin
    
    # Earth radius in kilometers
    R = 6371.0
    
    print(f"   Processing {len(roads)} roads (ULTRA FAST mode)...")
    
    # Process roads in batches
    road_count = 0
    for road in roads:
        road_count += 1
        if road_count % 1000 == 0:  # Progress every 1000 roads
            print(f"   ... road {road_count}/{len(roads)}")
        
        if len(road) < 2:
            continue
        
        # Get bounding box for entire road
        road_lons = [pt[0] for pt in road]
        road_lats = [pt[1] for pt in road]
        
        road_lon_min = min(road_lons) - buffer_deg
        road_lon_max = max(road_lons) + buffer_deg
        road_lat_min = min(road_lats) - buffer_deg
        road_lat_max = max(road_lats) + buffer_deg
        
        # Find grid points in road bounding box
        in_road_box = ((lon_grid >= road_lon_min) & (lon_grid <= road_lon_max) & 
                       (lat_grid >= road_lat_min) & (lat_grid <= road_lat_max))
        
        if not np.any(in_road_box):
            continue  # Skip if no points near this road
        
        # ULTRA FAST: Sample even fewer points (max 10 per road)
        sample_step = max(1, len(road) // 10)
        
        for i in range(0, len(road), sample_step):
            lon_pt, lat_pt = road[i]
            
            # Small bounding box around this road point
            lat_min = lat_pt - buffer_deg
            lat_max = lat_pt + buffer_deg
            lon_min = lon_pt - buffer_deg
            lon_max = lon_pt + buffer_deg
            
            # Find grid points in this small box
            in_box = ((lat_grid >= lat_min) & (lat_grid <= lat_max) & 
                     (lon_grid >= lon_min) & (lon_grid <= lon_max))
            
            if not np.any(in_box):
                continue
            
            # Calculate exact distance only for points in box
            lat1_rad = np.radians(lat_pt)
            lon1_rad = np.radians(lon_pt)
            lat2_rad = np.radians(lat_grid[in_box])
            lon2_rad = np.radians(lon_grid[in_box])
            
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            
            a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            
            distances_km = R * c
            
            # Mark points within distance
            within_distance = distances_km <= max_distance_km
            
            # Update mask
            mask[in_box] = np.logical_or(mask[in_box], within_distance)
    
    return mask


def mask_roads_from_geojson(lats: np.ndarray, lons: np.ndarray,
                            geojson_file: str = 'roads_nice_50km.geojson',
                            max_distance_m: float = 2000.0,
                            major_roads_only: bool = True) -> np.ndarray:
    """
    Create road proximity mask directly from GeoJSON file.
    
    ULTRA FAST VERSION with major roads filtering.
    
    Parameters:
    -----------
    lats : np.ndarray
        1D array of latitude values (degrees)
    lons : np.ndarray
        1D array of longitude values (degrees)
    geojson_file : str, optional
        Path to GeoJSON file (default: 'roads_nice_50km.geojson')
    max_distance_m : float, optional
        Maximum distance in meters (default: 2000.0)
    major_roads_only : bool, optional
        If True, only use major roads (MUCH faster) (default: True)
    
    Returns:
    --------
    np.ndarray
        Boolean array of shape (len(lats), len(lons))
        True = within distance of road (admissible)
        False = no road within distance (excluded)
    """
    roads = load_roads_from_geojson(geojson_file, major_roads_only=major_roads_only)
    print(f"   Loaded {len(roads)} roads (major_roads_only={major_roads_only})")
    return mask_roads_proximity_fast(lats, lons, roads, max_distance_m)