"""
KML/KMZ Export Module

This module provides functions to export coverage maps to KML/KMZ format
for visualization in Google Earth.
"""

import numpy as np
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, Optional
from pathlib import Path


def create_visibility_map_kml(
    coverage_map: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    flight_level: float,
    radar_lat: Optional[float] = None,
    radar_lon: Optional[float] = None,
    visible_color: str = "7f00ff00",  # Green with 50% opacity (AABBGGRR format)
    blocked_color: str = "7f0000ff"   # Red with 50% opacity (AABBGGRR format)
) -> ET.Element:
    """
    Create KML structure for a single coverage map.
    
    Parameters:
    -----------
    coverage_map : np.ndarray
        2D boolean array (True=visible, False=blocked)
    lats : np.ndarray
        1D array of latitude values
    lons : np.ndarray
        1D array of longitude values
    flight_level : float
        Flight level for naming
    radar_lat : float, optional
        Radar latitude
    radar_lon : float, optional
        Radar longitude
    visible_color : str
        Color for visible areas (AABBGGRR hex format, default: green)
    blocked_color : str
        Color for blocked areas (AABBGGRR hex format, default: red)
    
    Returns:
    --------
    ET.Element
        KML Document element
    """
    # Create KML document
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = ET.SubElement(kml, "Document")
    
    # Document name
    ET.SubElement(document, "name").text = f"Radar Coverage - FL{flight_level}"
    
    # Create styles
    # Style for visible areas
    visible_style = ET.SubElement(document, "Style", id="visible_style")
    poly_style = ET.SubElement(visible_style, "PolyStyle")
    ET.SubElement(poly_style, "color").text = visible_color
    ET.SubElement(poly_style, "fill").text = "1"
    ET.SubElement(poly_style, "outline").text = "0"
    
    # Style for blocked areas
    blocked_style = ET.SubElement(document, "Style", id="blocked_style")
    poly_style = ET.SubElement(blocked_style, "PolyStyle")
    ET.SubElement(poly_style, "color").text = blocked_color
    ET.SubElement(poly_style, "fill").text = "1"
    ET.SubElement(poly_style, "outline").text = "0"
    
    # Create folder for coverage polygons
    folder = ET.SubElement(document, "Folder")
    ET.SubElement(folder, "name").text = f"Coverage Map FL{flight_level}"
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Create polygons for each grid cell
    # For efficiency, we'll create larger polygons by grouping adjacent cells with same status
    # Simple approach: create polygon for each cell
    for i in range(len(lats) - 1):
        for j in range(len(lons) - 1):
            # Get cell corners
            lat0, lat1 = lats[i], lats[i + 1]
            lon0, lon1 = lons[j], lons[j + 1]
            
            # Get coverage status (use center of cell)
            status = coverage_map[i, j]
            
            # Create polygon
            placemark = ET.SubElement(folder, "Placemark")
            ET.SubElement(placemark, "name").text = f"Cell ({i},{j})"
            ET.SubElement(placemark, "styleUrl").text = "#visible_style" if status else "#blocked_style"
            
            polygon = ET.SubElement(placemark, "Polygon")
            outer_boundary = ET.SubElement(polygon, "outerBoundaryIs")
            linear_ring = ET.SubElement(outer_boundary, "LinearRing")
            coordinates = ET.SubElement(linear_ring, "coordinates")
            
            # Define polygon corners (rectangle)
            coord_str = f"{lon0},{lat0},0 {lon1},{lat0},0 {lon1},{lat1},0 {lon0},{lat1},0 {lon0},{lat0},0"
            coordinates.text = coord_str
    
    # Add radar position and reference points
    reference_folder = ET.SubElement(document, "Folder")
    ET.SubElement(reference_folder, "name").text = "Reference Points"
    
    # Add radar position if provided
    if radar_lat is not None and radar_lon is not None:
        radar_placemark = ET.SubElement(reference_folder, "Placemark")
        ET.SubElement(radar_placemark, "name").text = "Radar"
        ET.SubElement(radar_placemark, "description").text = f"Radar position at ({radar_lat:.6f}°N, {radar_lon:.6f}°E)"
        
        # Add blue icon style
        radar_style = ET.SubElement(radar_placemark, "Style")
        radar_icon_style = ET.SubElement(radar_style, "IconStyle")
        ET.SubElement(radar_icon_style, "color").text = "ffff0000"  # Blue (AABBGGRR format)
        
        radar_point = ET.SubElement(radar_placemark, "Point")
        radar_coords = ET.SubElement(radar_point, "coordinates")
        radar_coords.text = f"{radar_lon},{radar_lat},0"
    
    # Add Nice Airport as reference point
    nice_lat = 43.6584
    nice_lon = 7.2159
    nice_placemark = ET.SubElement(reference_folder, "Placemark")
    ET.SubElement(nice_placemark, "name").text = "Nice Airport (LFMN)"
    ET.SubElement(nice_placemark, "description").text = f"Nice Côte d'Azur Airport at ({nice_lat:.6f}°N, {nice_lon:.6f}°E)"
    
    # Add blue icon style
    nice_style = ET.SubElement(nice_placemark, "Style")
    nice_icon_style = ET.SubElement(nice_style, "IconStyle")
    ET.SubElement(nice_icon_style, "color").text = "ffff0000"  # Blue (AABBGGRR format)
    
    nice_point = ET.SubElement(nice_placemark, "Point")
    nice_coords = ET.SubElement(nice_point, "coordinates")
    nice_coords.text = f"{nice_lon},{nice_lat},0"
    
    return kml


def export_coverage_to_kml(
    coverage_map: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    flight_level: float,
    output_path: str,
    radar_lat: Optional[float] = None,
    radar_lon: Optional[float] = None
) -> None:
    """
    Export a single coverage map to KML file.
    
    Parameters:
    -----------
    coverage_map : np.ndarray
        2D boolean array (True=visible, False=blocked)
    lats : np.ndarray
        1D array of latitude values
    lons : np.ndarray
        1D array of longitude values
    flight_level : float
        Flight level
    output_path : str
        Output KML file path
    radar_lat : float, optional
        Radar latitude
    radar_lon : float, optional
        Radar longitude
    """
    kml = create_visibility_map_kml(
        coverage_map, lats, lons, flight_level, radar_lat, radar_lon
    )
    
    # Write to file
    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding='utf-8', xml_declaration=True)


def export_all_coverage_to_kmz(
    coverage_maps: Dict[float, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    radar_lat: Optional[float] = None,
    radar_lon: Optional[float] = None,
    output_path: str = "radar_coverage.kmz"
) -> None:
    """
    Export all coverage maps to a single KMZ file.
    
    Parameters:
    -----------
    coverage_maps : Dict[float, np.ndarray]
        Dictionary mapping flight level to coverage map
    lats : np.ndarray
        1D array of latitude values
    lons : np.ndarray
        1D array of longitude values
    radar_lat : float, optional
        Radar latitude
    radar_lon : float, optional
        Radar longitude
    output_path : str
        Output KMZ file path
    """
    # Create main KML document
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = ET.SubElement(kml, "Document")
    ET.SubElement(document, "name").text = "Radar Coverage Analysis"
    
    # Create shared styles
    visible_style = ET.SubElement(document, "Style", id="visible_style")
    poly_style = ET.SubElement(visible_style, "PolyStyle")
    ET.SubElement(poly_style, "color").text = "7f00ff00"  # Green
    ET.SubElement(poly_style, "fill").text = "1"
    ET.SubElement(poly_style, "outline").text = "0"
    
    blocked_style = ET.SubElement(document, "Style", id="blocked_style")
    poly_style = ET.SubElement(blocked_style, "PolyStyle")
    ET.SubElement(poly_style, "color").text = "7f0000ff"  # Red
    ET.SubElement(poly_style, "fill").text = "1"
    ET.SubElement(poly_style, "outline").text = "0"
    
    # Create folder for each flight level
    flight_levels = sorted(coverage_maps.keys())
    
    for fl in flight_levels:
        folder = ET.SubElement(document, "Folder")
        ET.SubElement(folder, "name").text = f"FL{fl}"
        ET.SubElement(folder, "description").text = f"Coverage map for Flight Level {fl}"
        
        coverage_map = coverage_maps[fl]
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Create polygons for coverage map
        # For large grids, we'll sample or simplify to avoid too many polygons
        # Use every Nth point to reduce polygon count
        step = max(1, min(len(lats) // 100, len(lons) // 100))  # Adaptive sampling
        
        for i in range(0, len(lats) - 1, step):
            for j in range(0, len(lons) - 1, step):
                lat0, lat1 = lats[i], lats[min(i + step, len(lats) - 1)]
                lon0, lon1 = lons[j], lons[min(j + step, len(lons) - 1)]
                
                status = coverage_map[i, j]
                
                placemark = ET.SubElement(folder, "Placemark")
                polygon = ET.SubElement(placemark, "Polygon")
                outer_boundary = ET.SubElement(polygon, "outerBoundaryIs")
                linear_ring = ET.SubElement(outer_boundary, "LinearRing")
                coordinates = ET.SubElement(linear_ring, "coordinates")
                
                coord_str = f"{lon0},{lat0},0 {lon1},{lat0},0 {lon1},{lat1},0 {lon0},{lat1},0 {lon0},{lat0},0"
                coordinates.text = coord_str
                
                ET.SubElement(placemark, "styleUrl").text = "#visible_style" if status else "#blocked_style"
    
    # Add radar position and reference points
    reference_folder = ET.SubElement(document, "Folder")
    ET.SubElement(reference_folder, "name").text = "Reference Points"
    
    # Add radar position if provided
    if radar_lat is not None and radar_lon is not None:
        radar_placemark = ET.SubElement(reference_folder, "Placemark")
        ET.SubElement(radar_placemark, "name").text = "Radar"
        ET.SubElement(radar_placemark, "description").text = f"Radar position at ({radar_lat:.6f}°N, {radar_lon:.6f}°E)"
        
        # Add blue icon style
        radar_style = ET.SubElement(radar_placemark, "Style")
        radar_icon_style = ET.SubElement(radar_style, "IconStyle")
        ET.SubElement(radar_icon_style, "color").text = "ffff0000"  # Blue (AABBGGRR format)
        
        radar_point = ET.SubElement(radar_placemark, "Point")
        radar_coords = ET.SubElement(radar_point, "coordinates")
        radar_coords.text = f"{radar_lon},{radar_lat},0"
    
    # Add Nice Airport as reference point
    nice_lat = 43.6584
    nice_lon = 7.2159
    nice_placemark = ET.SubElement(reference_folder, "Placemark")
    ET.SubElement(nice_placemark, "name").text = "Nice Airport (LFMN)"
    ET.SubElement(nice_placemark, "description").text = f"Nice Côte d'Azur Airport at ({nice_lat:.6f}°N, {nice_lon:.6f}°E)"
    
    # Add blue icon style
    nice_style = ET.SubElement(nice_placemark, "Style")
    nice_icon_style = ET.SubElement(nice_style, "IconStyle")
    ET.SubElement(nice_icon_style, "color").text = "ffff0000"  # Blue (AABBGGRR format)
    
    nice_point = ET.SubElement(nice_placemark, "Point")
    nice_coords = ET.SubElement(nice_point, "coordinates")
    nice_coords.text = f"{nice_lon},{nice_lat},0"
    
    # Create KMZ file (ZIP archive)
    output_path = Path(output_path)
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as kmz:
        # Write main KML file
        tree = ET.ElementTree(kml)
        ET.indent(tree, space="  ")
        
        # Write to string first, then to ZIP
        from io import BytesIO
        kml_str = ET.tostring(kml, encoding='utf-8', xml_declaration=True)
        kmz.writestr("doc.kml", kml_str)
    
    print(f"KMZ file created: {output_path}")
