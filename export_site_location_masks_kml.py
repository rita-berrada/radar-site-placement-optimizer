"""
Site Location Masks KML/KMZ Export Module

This module provides functions to export geographical masks to KML/KMZ format
for visualization in Google Earth.

Masks are exported with:
- Admissible areas: transparent (not shown, allowing Google Earth base map to show)
- Excluded areas: grey with partial opacity overlaying the map
"""

import numpy as np
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, Optional
from pathlib import Path


def _create_grouped_polygons(folder: ET.Element, mask: np.ndarray, 
                             lats: np.ndarray, lons: np.ndarray, 
                             style_url: str) -> None:
    """
    Create grouped polygons for excluded areas to reduce pixelation.
    
    Groups adjacent excluded cells into larger rectangular polygons for smoother
    boundaries in Google Earth visualization.
    
    Parameters:
    -----------
    folder : ET.Element
        KML Folder element to add polygons to
    mask : np.ndarray
        Boolean mask (True=admissible, False=excluded)
    lats : np.ndarray
        1D array of latitude values
    lons : np.ndarray
        1D array of longitude values
    style_url : str
        Style URL for the polygons
    """
    excluded = ~mask  # True where excluded
    
    # Use a simple grouping algorithm: merge horizontally adjacent cells
    # This reduces the number of polygons and creates smoother boundaries
    visited = np.zeros_like(excluded, dtype=bool)
    polygon_count = 0
    
    for i in range(len(lats) - 1):
        for j in range(len(lons) - 1):
            if excluded[i, j] and not visited[i, j]:
                # Find the extent of this excluded region (horizontal grouping)
                # Start from current cell
                start_j = j
                end_j = j
                
                # Extend horizontally as long as cells are excluded
                while end_j + 1 < len(lons) - 1 and excluded[i, end_j + 1] and not visited[i, end_j + 1]:
                    end_j += 1
                
                # Try to extend vertically to form a larger rectangle
                end_i = i
                can_extend = True
                while can_extend and end_i + 1 < len(lats) - 1:
                    # Check if next row has excluded cells in the same column range
                    if np.all(excluded[end_i + 1, start_j:end_j + 1]) and \
                       np.all(~visited[end_i + 1, start_j:end_j + 1]):
                        end_i += 1
                    else:
                        can_extend = False
                
                # Mark all cells in this group as visited
                visited[i:end_i + 1, start_j:end_j + 1] = True
                
                # Create polygon for this group
                lat0, lat1 = lats[i], lats[end_i + 1]
                lon0, lon1 = lons[start_j], lons[end_j + 1]
                
                placemark = ET.SubElement(folder, "Placemark")
                ET.SubElement(placemark, "name").text = f"Excluded Region {polygon_count}"
                ET.SubElement(placemark, "styleUrl").text = style_url
                
                polygon = ET.SubElement(placemark, "Polygon")
                outer_boundary = ET.SubElement(polygon, "outerBoundaryIs")
                linear_ring = ET.SubElement(outer_boundary, "LinearRing")
                coordinates = ET.SubElement(linear_ring, "coordinates")
                
                # Define polygon corners (rectangle)
                coord_str = f"{lon0},{lat0},0 {lon1},{lat0},0 {lon1},{lat1},0 {lon0},{lat1},0 {lon0},{lat0},0"
                coordinates.text = coord_str
                
                polygon_count += 1


def create_mask_kml(
    mask: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    mask_name: str = "Site Location Mask",
    nice_lat: Optional[float] = None,
    nice_lon: Optional[float] = None,
    excluded_color: str = "CC808080"  # Grey with 80% opacity (AABBGGRR format)
) -> ET.Element:
    """
    Create KML structure for a single mask.
    
    Parameters:
    -----------
    mask : np.ndarray
        2D boolean array (True=admissible, False=excluded)
    lats : np.ndarray
        1D array of latitude values
    lons : np.ndarray
        1D array of longitude values
    mask_name : str
        Name for the mask
    nice_lat : float, optional
        Nice airport latitude
    nice_lon : float, optional
        Nice airport longitude
    excluded_color : str
        Color for excluded areas (AABBGGRR hex format, default: grey with 50% opacity)
    
    Returns:
    --------
    ET.Element
        KML Document element
    """
    # Create KML document
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = ET.SubElement(kml, "Document")
    
    # Document name
    ET.SubElement(document, "name").text = mask_name
    
    # Create style for excluded areas
    excluded_style = ET.SubElement(document, "Style", id="excluded_style")
    poly_style = ET.SubElement(excluded_style, "PolyStyle")
    ET.SubElement(poly_style, "color").text = excluded_color
    ET.SubElement(poly_style, "fill").text = "1"
    ET.SubElement(poly_style, "outline").text = "0"
    
    # Create folder for excluded area polygons
    folder = ET.SubElement(document, "Folder")
    ET.SubElement(folder, "name").text = "Excluded Areas"
    ET.SubElement(folder, "description").text = "Areas excluded from radar site location (grey overlay)"
    
    # Create polygons only for excluded areas (admissible areas are transparent/not shown)
    # Group adjacent excluded cells into larger polygons for smoother boundaries
    _create_grouped_polygons(folder, mask, lats, lons, "#excluded_style")
    
    # Add reference points folder
    reference_folder = ET.SubElement(document, "Folder")
    ET.SubElement(reference_folder, "name").text = "Reference Points"
    
    # Add Nice Airport as reference point
    if nice_lat is not None and nice_lon is not None:
        nice_placemark = ET.SubElement(reference_folder, "Placemark")
        ET.SubElement(nice_placemark, "name").text = "Nice Airport (LFMN)"
        ET.SubElement(nice_placemark, "description").text = (
            f"Nice Côte d'Azur Airport at ({nice_lat:.6f}°N, {nice_lon:.6f}°E) - "
            "Reference point for distance constraint"
        )
        
        # Add icon style
        nice_style = ET.SubElement(nice_placemark, "Style")
        nice_icon_style = ET.SubElement(nice_style, "IconStyle")
        ET.SubElement(nice_icon_style, "color").text = "ffff0000"  # Blue (AABBGGRR format)
        ET.SubElement(nice_icon_style, "scale").text = "1.2"
        
        nice_point = ET.SubElement(nice_placemark, "Point")
        nice_coords = ET.SubElement(nice_point, "coordinates")
        nice_coords.text = f"{nice_lon},{nice_lat},0"
    
    # Add statistics folder
    stats_folder = ET.SubElement(document, "Folder")
    ET.SubElement(stats_folder, "name").text = "Statistics"
    
    admissible_count = np.sum(mask)
    total_count = mask.size
    admissible_pct = admissible_count / total_count * 100
    
    stats_placemark = ET.SubElement(stats_folder, "Placemark")
    ET.SubElement(stats_placemark, "name").text = "Mask Statistics"
    ET.SubElement(stats_placemark, "description").text = (
        f"Total grid points: {total_count:,}\n"
        f"Admissible points: {admissible_count:,} ({admissible_pct:.1f}%)\n"
        f"Excluded points: {total_count - admissible_count:,} ({100 - admissible_pct:.1f}%)"
    )
    
    return kml


def export_mask_to_kml(
    mask: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    output_path: str,
    mask_name: str = "Site Location Mask",
    nice_lat: Optional[float] = None,
    nice_lon: Optional[float] = None
) -> None:
    """
    Export a single mask to KML file.
    
    Parameters:
    -----------
    mask : np.ndarray
        2D boolean array (True=admissible, False=excluded)
    lats : np.ndarray
        1D array of latitude values
    lons : np.ndarray
        1D array of longitude values
    output_path : str
        Output KML file path
    mask_name : str, optional
        Name for the mask
    nice_lat : float, optional
        Nice airport latitude
    nice_lon : float, optional
        Nice airport longitude
    """
    kml = create_mask_kml(mask, lats, lons, mask_name, nice_lat, nice_lon)
    
    # Write to file
    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"Exported mask to KML: {output_path}")


def export_masks_to_kmz(
    masks_dict: Dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    output_path: str = "site_location_masks.kmz",
    nice_lat: Optional[float] = None,
    nice_lon: Optional[float] = None
) -> None:
    """
    Export multiple masks to a single KMZ file.
    
    Parameters:
    -----------
    masks_dict : Dict[str, np.ndarray]
        Dictionary mapping mask names to boolean arrays
    lats : np.ndarray
        1D array of latitude values
    lons : np.ndarray
        1D array of longitude values
    output_path : str, optional
        Output KMZ file path (default: "site_location_masks.kmz")
    nice_lat : float, optional
        Nice airport latitude
    nice_lon : float, optional
        Nice airport longitude
    """
    # Create main KML document
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = ET.SubElement(kml, "Document")
    ET.SubElement(document, "name").text = "Site Location Masks"
    ET.SubElement(document, "description").text = (
        "Geographical masks for radar site location study. "
        "Grey areas are excluded, transparent areas are admissible."
    )
    
    # Create folder for each mask
    for mask_name, mask in masks_dict.items():
        # Verify mask shape
        if mask.shape != (len(lats), len(lons)):
            raise ValueError(f"Mask '{mask_name}' has shape {mask.shape}, expected ({len(lats)}, {len(lons)})")
        
        mask_folder = ET.SubElement(document, "Folder")
        ET.SubElement(mask_folder, "name").text = mask_name
        
        # Create style for excluded areas
        style_id = f"excluded_style_{mask_name.replace(' ', '_')}"
        excluded_style = ET.SubElement(document, "Style", id=style_id)
        poly_style = ET.SubElement(excluded_style, "PolyStyle")
        ET.SubElement(poly_style, "color").text = "CC808080"  # Grey with 80% opacity
        ET.SubElement(poly_style, "fill").text = "1"
        ET.SubElement(poly_style, "outline").text = "0"
        
        # Create subfolder for excluded areas
        excluded_folder = ET.SubElement(mask_folder, "Folder")
        ET.SubElement(excluded_folder, "name").text = "Excluded Areas"
        
        # Create grouped polygons for excluded cells (smoother boundaries)
        _create_grouped_polygons(excluded_folder, mask, lats, lons, f"#{style_id}")
    
    # Add reference points folder
    reference_folder = ET.SubElement(document, "Folder")
    ET.SubElement(reference_folder, "name").text = "Reference Points"
    
    if nice_lat is not None and nice_lon is not None:
        nice_placemark = ET.SubElement(reference_folder, "Placemark")
        ET.SubElement(nice_placemark, "name").text = "Nice Airport (LFMN)"
        ET.SubElement(nice_placemark, "description").text = (
            f"Nice Côte d'Azur Airport at ({nice_lat:.6f}°N, {nice_lon:.6f}°E)"
        )
        
        nice_style = ET.SubElement(nice_placemark, "Style")
        nice_icon_style = ET.SubElement(nice_style, "IconStyle")
        ET.SubElement(nice_icon_style, "color").text = "ffff0000"  # Blue
        ET.SubElement(nice_icon_style, "scale").text = "1.2"
        
        nice_point = ET.SubElement(nice_placemark, "Point")
        nice_coords = ET.SubElement(nice_point, "coordinates")
        nice_coords.text = f"{nice_lon},{nice_lat},0"
    
    # Create KMZ file
    kmz_path = Path(output_path)
    with zipfile.ZipFile(kmz_path, 'w', zipfile.ZIP_DEFLATED) as kmz:
        # Write KML to string
        tree = ET.ElementTree(kml)
        ET.indent(tree, space="  ")
        kml_str = ET.tostring(kml, encoding='utf-8', xml_declaration=True)
        
        # Write to KMZ
        kmz.writestr('doc.kml', kml_str)
    
    print(f"Exported {len(masks_dict)} mask(s) to KMZ: {output_path}")
