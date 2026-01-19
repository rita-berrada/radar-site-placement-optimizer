"""
export_authorized_points.py

Exports the final candidate sites from 'authorized_points_all_masks.npz'
to a Google Earth KMZ file as green placemarks.
"""

import numpy as np
import zipfile
import os

# Configuration
INPUT_FILE = "authorized_points_all_masks.npz"
OUTPUT_KMZ = "final_candidates.kmz"
MAX_POINTS_DISPLAY = 5000  # Safety limit to prevent Google Earth crash

def export_candidates():
    print(f"Exporting candidates from {INPUT_FILE}...")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run the masks script first.")
        return

    # 1. Load Data
    data = np.load(INPUT_FILE)
    lats = data['lat']
    lons = data['lon']
    # Elevation might be needed for 'absolute' altitude mode, 
    # but 'clampToGround' is safer for visualization.
    
    num_points = len(lats)
    print(f"   Total authorized points found: {num_points:,}")

    if num_points == 0:
        print("   No candidates to export.")
        return

    # 2. Subsampling (Safety check)
    step = 1
    if num_points > MAX_POINTS_DISPLAY:
        step = int(np.ceil(num_points / MAX_POINTS_DISPLAY))
        print(f"   [!] Too many points for Google Earth. Displaying 1 out of {step} points.")
    
    lats_viz = lats[::step]
    lons_viz = lons[::step]

    # 3. Generate KML Content
    # We use a simple Green Pushpin style
    kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Authorized Radar Sites</name>
    <Style id="greenPin">
      <IconStyle>
        <color>ff00ff00</color> <scale>1.1</scale>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/pushpin/grn-pushpin.png</href>
        </Icon>
      </IconStyle>
    </Style>
    <Folder>
      <name>Candidates</name>
"""
    
    kml_footer = """    </Folder>
  </Document>
</kml>"""

    kml_body = ""
    for i in range(len(lats_viz)):
        lat = lats_viz[i]
        lon = lons_viz[i]
        
        # Create a placemark for each point
        kml_body += f"""
      <Placemark>
        <styleUrl>#greenPin</styleUrl>
        <Point>
          <altitudeMode>clampToGround</altitudeMode>
          <coordinates>{lon},{lat},0</coordinates>
        </Point>
      </Placemark>"""

    # 4. Write and Zip
    with open("doc.kml", "w") as f:
        f.write(kml_header + kml_body + kml_footer)

    with zipfile.ZipFile(OUTPUT_KMZ, "w", zipfile.ZIP_DEFLATED) as z:
        z.write("doc.kml")

    os.remove("doc.kml")
    print(f"Success: {OUTPUT_KMZ} created with {len(lats_viz)} points.")

if __name__ == "__main__":
    export_candidates()