"""
Electrical Stations Masks Module

This module implements boolean geographical masks for electrical infrastructure constraint.
These masks identify admissible areas based on proximity to Enedis electrical stations
(REQ_06: Electrical access < 500m).

The masks are designed to be:
- Reusable and combinable with other geographical masks
- Independent from radar logic
- Compatible with the DTED terrain grid structure
"""

import numpy as np
import json
from typing import List, Dict, Optional


def load_stations_from_json(json_file: str) -> List[Dict[str, float]]:
    """
    Load electrical stations data from Enedis JSON file.
    
    Parameters:
    -----------
    json_file : str
        Path to the page1.json file containing Enedis stations data
    
    Returns:
    --------
    List[Dict[str, float]]
        List of stations with 'lat', 'lon', 'distance_m' keys
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stations = []
    for result in data['results']:
        geopoint = result['_geopoint']
        if isinstance(geopoint, str):
            lat, lon = map(float, geopoint.split(','))
        else:
            lat = geopoint['lat']
            lon = geopoint['lon']
        
        stations.append({
            'lat': lat,
            'lon': lon,
            'distance_m': result['_geo_distance']
        })
    
    return stations


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    
    Parameters:
    -----------
    lat1, lon1 : float
        Latitude and longitude of first point (degrees)
    lat2, lon2 : float
        Latitude and longitude of second point (degrees)
    
    Returns:
    --------
    float
        Distance in kilometers
    """
    R = 6371.0  # Earth radius in kilometers
    
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance_km = R * c
    return distance_km


def mask_electrical_500m(lats: np.ndarray, lons: np.ndarray, 
                         stations: List[Dict[str, float]], 
                         radius_m: float = 500.0) -> np.ndarray:
    """
    Create a boolean mask for locations within specified radius of any electrical station.
    
    OPTIMIZED VERSION: Uses bounding boxes for faster calculation.
    
    The mask is True where at least one electrical station is within radius_m meters,
    False otherwise. This implements REQ_06: Electrical access < 500m.
    
    Parameters:
    -----------
    lats : np.ndarray
        1D array of latitude values (degrees)
    lons : np.ndarray
        1D array of longitude values (degrees)
    stations : List[Dict[str, float]]
        List of electrical stations with 'lat' and 'lon' keys
    radius_m : float, optional
        Maximum distance in meters (default: 500.0)
    
    Returns:
    --------
    np.ndarray
        Boolean array of shape (len(lats), len(lons))
        True = within radius of at least one station (admissible)
        False = no station within radius (excluded)
    """
    # Create meshgrid for all lat/lon combinations
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Initialize mask as False (no stations nearby)
    mask = np.zeros((len(lats), len(lons)), dtype=bool)
    
    # Convert radius from meters to kilometers
    radius_km = radius_m / 1000.0
    
    # Approximate radius in degrees (for bounding box)
    # At 43°N latitude, 1 degree ≈ 111 km for latitude, ~78 km for longitude
    radius_deg_lat = radius_km / 111.0
    radius_deg_lon = radius_km / (111.0 * np.cos(np.radians(43.6)))
    buffer_deg = max(radius_deg_lat, radius_deg_lon) * 1.5  # Add safety margin
    
    # Earth radius in kilometers
    R = 6371.0
    
    print(f"   Processing {len(stations)} electrical stations (optimized)...")
    
    # For each station, mark all grid points within radius
    for i, station in enumerate(stations):
        if (i + 1) % 50 == 0:  # Progress indicator
            print(f"   ... station {i+1}/{len(stations)}")
        
        station_lat = station['lat']
        station_lon = station['lon']
        
        # OPTIMIZATION: Use bounding box to filter grid points first
        lat_min = station_lat - buffer_deg
        lat_max = station_lat + buffer_deg
        lon_min = station_lon - buffer_deg
        lon_max = station_lon + buffer_deg
        
        # Find grid points in bounding box
        in_box = ((lat_grid >= lat_min) & (lat_grid <= lat_max) & 
                 (lon_grid >= lon_min) & (lon_grid <= lon_max))
        
        if not np.any(in_box):
            continue  # Skip if no points in bounding box
        
        # Calculate exact distance only for points in bounding box
        lat1_rad = np.radians(station_lat)
        lon1_rad = np.radians(station_lon)
        lat2_rad = np.radians(lat_grid[in_box])
        lon2_rad = np.radians(lon_grid[in_box])
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        distances_km = R * c
        
        # Mark points within radius
        within_radius = distances_km <= radius_km
        
        # Update mask (only for points in bounding box)
        mask[in_box] = np.logical_or(mask[in_box], within_radius)
    
    return mask


def mask_electrical_from_json(lats: np.ndarray, lons: np.ndarray,
                               json_file: str = 'page1.json',
                               radius_m: float = 500.0) -> np.ndarray:
    """
    Create electrical station mask directly from JSON file.
    
    Convenience function that loads stations and creates mask in one call.
    
    Parameters:
    -----------
    lats : np.ndarray
        1D array of latitude values (degrees)
    lons : np.ndarray
        1D array of longitude values (degrees)
    json_file : str, optional
        Path to Enedis JSON file (default: 'page1.json')
    radius_m : float, optional
        Maximum distance in meters (default: 500.0)
    
    Returns:
    --------
    np.ndarray
        Boolean array of shape (len(lats), len(lons))
        True = within radius of electrical station (admissible)
        False = no station within radius (excluded)
    """
    stations = load_stations_from_json(json_file)
    return mask_electrical_500m(lats, lons, stations, radius_m)