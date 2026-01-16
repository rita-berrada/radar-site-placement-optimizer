#!/usr/bin/env python3
"""
Visualize authorized radar candidate points in Google Earth (KMZ).

Input:
- authorized_points_01deg.npz  (contains array key "points" with shape (N,2): [lat, lon])

Output:
- authorized_points_01deg.kmz  (open with Google Earth)
"""

from pathlib import Path
import zipfile
import numpy as np
import xml.etree.ElementTree as ET


def _kml_color(aabbggrr: str) -> str:
    """
    KML color format is AABBGGRR (alpha, blue, green, red).
    Example: "ff0000ff" = opaque red.
    """
    if len(aabbggrr) != 8:
        raise ValueError("KML color must be 8 hex chars in AABBGGRR format, e.g. 'ff0000ff'")
    return aabbggrr.lower()


def export_points_to_kmz(
    points_latlon: np.ndarray,
    output_kmz_path: str = "authorized_points_01deg.kmz",
    doc_name: str = "Authorized candidate points (0.1°)",
    point_name_prefix: str = "Candidate",
    point_color: str = "ff00ffff",  # opaque yellow (AABBGGRR)
    point_scale: float = 0.8,
    nice_lat: float | None = 43.6584,
    nice_lon: float | None = 7.2159,
) -> None:
    """
    Export candidate points to a KMZ file for Google Earth.

    Args:
        points_latlon: np.ndarray shape (N,2) with columns [lat, lon]
        output_kmz_path: output .kmz filename
        doc_name: document name shown in Google Earth
        point_name_prefix: label prefix for points
        point_color: KML color in AABBGGRR
        point_scale: icon scale
        nice_lat/nice_lon: optional reference point (Nice Airport). Set to None to disable.

    Returns:
        None (writes KMZ on disk)
    """
    if points_latlon.ndim != 2 or points_latlon.shape[1] != 2:
        raise ValueError("points_latlon must be a (N, 2) array with [lat, lon].")

    # Root KML structure
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = doc_name

    # Style for candidate points
    style_id = "candidateStyle"
    style = ET.SubElement(doc, "Style", id=style_id)
    icon_style = ET.SubElement(style, "IconStyle")
    ET.SubElement(icon_style, "color").text = _kml_color(point_color)
    ET.SubElement(icon_style, "scale").text = str(point_scale)
    icon = ET.SubElement(icon_style, "Icon")
    # Simple default icon (Google Earth built-in circle icon)
    ET.SubElement(icon, "href").text = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"

    # Folder for candidates
    folder = ET.SubElement(doc, "Folder")
    ET.SubElement(folder, "name").text = "Authorized candidates"

    # Add each point
    for idx, (lat, lon) in enumerate(points_latlon, start=1):
        pm = ET.SubElement(folder, "Placemark")
        ET.SubElement(pm, "name").text = f"{point_name_prefix} {idx}"
        ET.SubElement(pm, "styleUrl").text = f"#{style_id}"

        pt = ET.SubElement(pm, "Point")
        # KML uses lon,lat,alt
        ET.SubElement(pt, "coordinates").text = f"{float(lon):.6f},{float(lat):.6f},0"

    # Optional: Nice reference point
    if nice_lat is not None and nice_lon is not None:
        ref_folder = ET.SubElement(doc, "Folder")
        ET.SubElement(ref_folder, "name").text = "Reference"
        pm = ET.SubElement(ref_folder, "Placemark")
        ET.SubElement(pm, "name").text = "Nice Airport (LFMN)"
        # Simple different style
        ref_style_id = "refStyle"
        ref_style = ET.SubElement(doc, "Style", id=ref_style_id)
        ref_icon_style = ET.SubElement(ref_style, "IconStyle")
        ET.SubElement(ref_icon_style, "color").text = _kml_color("ff0000ff")  # opaque red
        ET.SubElement(ref_icon_style, "scale").text = "1.1"
        ref_icon = ET.SubElement(ref_icon_style, "Icon")
        ET.SubElement(ref_icon, "href").text = "http://maps.google.com/mapfiles/kml/shapes/airports.png"
        ET.SubElement(pm, "styleUrl").text = f"#{ref_style_id}"

        pt = ET.SubElement(pm, "Point")
        ET.SubElement(pt, "coordinates").text = f"{float(nice_lon):.6f},{float(nice_lat):.6f},0"

    # Write KMZ (zip containing doc.kml)
    output_kmz_path = str(Path(output_kmz_path))
    kml_bytes = ET.tostring(kml, encoding="utf-8", xml_declaration=True)

    with zipfile.ZipFile(output_kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_bytes)

    print(f"✓ Exported {len(points_latlon):,} points to {output_kmz_path}")


def main():
    # Load points
    in_npz = "authorized_points_01deg.npz"
    data = np.load(in_npz)
    points = data["points"]  # (N,2) [lat, lon]

    # Safety: ensure float + 2 columns
    points = np.array(points, dtype=float).reshape(-1, 2)

    export_points_to_kmz(
        points_latlon=points,
        output_kmz_path="authorized_points_01deg.kmz",
        doc_name="Authorized radar candidates (rounded 0.1°)",
        point_name_prefix="Candidate",
        point_color="ff00ffff",  # yellow
        point_scale=0.8,
        nice_lat=43.6584,
        nice_lon=7.2159,
    )


if __name__ == "__main__":
    main()
