"""
export_protected_areas_mask_kml.py

Generates a Google Earth KMZ to visualize Protected Areas (National Parks).
Forbidden zones are rendered in semi-transparent GRAY.
"""

import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt

from geo_utils import load_and_convert_to_enu
from protected_areas_mask import mask_protected_areas_from_geojson

# Configuration
INPUT_NPZ = "terrain_mat.npz"
INPUT_GEOJSON = "protected_areas.geojson"
OUTPUT_KMZ = "protected_areas_gray.kmz"
GRAY_RGBA = [0.4, 0.4, 0.4, 0.6]  # Dark Gray, 60% opacity

def generate_kmz():
    print(f"Generating {OUTPUT_KMZ}...")

    if not os.path.exists(INPUT_GEOJSON):
        print(f"Error: {INPUT_GEOJSON} not found. Run generate_protected_geojson.py first.")
        return

    # 1. Load coordinates and Compute Mask
    # mask_protected returns True for Admissible (Outside), False for Inside
    _, _, _, lats, lons = load_and_convert_to_enu(INPUT_NPZ)
    is_admissible = mask_protected_areas_from_geojson(lats, lons, INPUT_GEOJSON)
    
    # We want to visualize the FORBIDDEN zones (Inside the park)
    is_protected_zone = ~is_admissible

    # 2. Enforce North-to-South orientation (Image standard)
    # Sort latitudes descending so row 0 = North
    sort_indices = np.argsort(lats)[::-1]
    lats_sorted = lats[sort_indices]
    
    # Reorder the mask rows to match
    mask_sorted = is_protected_zone[sort_indices, :]

    # 3. Create RGBA Overlay
    ny, nx = mask_sorted.shape
    img_rgba = np.zeros((ny, nx, 4), dtype=float)
    
    # Apply Gray color to protected areas
    img_rgba[mask_sorted] = GRAY_RGBA

    # 4. Save temporary PNG
    img_filename = "overlay_protected.png"
    plt.imsave(img_filename, img_rgba)

    # 5. Define bounding box
    north = lats_sorted[0]   # Max lat
    south = lats_sorted[-1]  # Min lat
    east = np.max(lons)
    west = np.min(lons)

    # 6. Write KML
    kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Protected Areas Exclusion</name>
    <GroundOverlay>
      <name>Protected Zones</name>
      <Icon><href>{img_filename}</href></Icon>
      <LatLonBox>
        <north>{north}</north><south>{south}</south>
        <east>{east}</east><west>{west}</west>
      </LatLonBox>
    </GroundOverlay>
  </Document>
</kml>"""

    with open("doc.kml", "w") as f:
        f.write(kml_content)

    # 7. Zip into KMZ
    with zipfile.ZipFile(OUTPUT_KMZ, "w", zipfile.ZIP_DEFLATED) as z:
        z.write("doc.kml")
        z.write(img_filename)

    # Cleanup
    os.remove("doc.kml")
    os.remove(img_filename)
    print(f"Success: {OUTPUT_KMZ} created.")

if __name__ == "__main__":
    generate_kmz()