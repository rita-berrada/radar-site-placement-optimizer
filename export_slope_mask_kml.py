"""
export_slope_mask.py

Exports a Google Earth KMZ file visualizing steep terrain.
Steep areas (> 15%) are rendered in semi-transparent gray.
"""

import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt

from geo_utils import load_and_convert_to_enu
from mask_slope import mask_slope

# Configuration
INPUT_FILE = "terrain_mat.npz"
OUTPUT_KMZ = "slope_mask_gray.kmz"
THRESHOLD_PCT = 15.0
GRAY_RGBA = [0.5, 0.5, 0.5, 0.6]

def generate_kmz():
    # 1. Compute logic and load coordinates
    is_flat = mask_slope(INPUT_FILE, max_slope_percent=THRESHOLD_PCT)
    _, _, _, lats, lons = load_and_convert_to_enu(INPUT_FILE)

    # 2. Enforce North-to-South orientation
    # Sort latitudes descending so row 0 corresponds to the North (Top of image)
    sort_indices = np.argsort(lats)[::-1]
    lats_sorted = lats[sort_indices]
    
    # Reorder the mask rows to match latitude sorting
    is_flat_sorted = is_flat[sort_indices, :]
    is_steep_sorted = ~is_flat_sorted

    # 3. Create RGBA Overlay
    ny, nx = is_flat_sorted.shape
    img_rgba = np.zeros((ny, nx, 4), dtype=float)
    img_rgba[is_steep_sorted] = GRAY_RGBA

    # 4. Save temporary PNG
    img_filename = "overlay.png"
    plt.imsave(img_filename, img_rgba)

    # 5. Define bounding box for Google Earth
    north = lats_sorted[0]   # Max lat (Top)
    south = lats_sorted[-1]  # Min lat (Bottom)
    east = np.max(lons)
    west = np.min(lons)

    # 6. Write KML
    kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Steep Areas (> {THRESHOLD_PCT}%)</name>
    <GroundOverlay>
      <name>Slope Mask</name>
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
    print(f"Successfully created: {OUTPUT_KMZ}")

if __name__ == "__main__":
    generate_kmz()