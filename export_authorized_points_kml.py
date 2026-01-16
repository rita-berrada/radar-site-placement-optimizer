#!/usr/bin/env python3
"""
Export authorized radar site points (from NPZ) to a Google Earth KMZ file.

- No external dependencies.
- Writes a KMZ (ZIP) containing a 'doc.kml' file, which Google Earth opens directly.
- Includes optional sub-sampling to avoid Google Earth lag with too many points.

Input:
  authorized_points_all_masks.npz  (expects arrays: lat, lon, and optionally z)

Output:
  authorized_points_all_masks_POINTS.kmz
"""

import io
import zipfile
import numpy as np


def export_points_kmz(
    npz_file: str,
    kmz_file: str,
    max_points: int = 10000,
    name: str = "Authorized points (ALL masks)",
    icon_href: str = "http://maps.google.com/mapfiles/kml/paddle/red-circle.png",
    icon_scale: float = 0.6,
):
    data = np.load(npz_file)
    lat = data["lat"]
    lon = data["lon"]
    z = data["z"] if "z" in data.files else None

    n = len(lat)
    if n == 0:
        raise RuntimeError(f"No points found in {npz_file} (lat/lon arrays are empty).")

    # Subsample to keep Google Earth responsive
    step = max(1, n // max_points) if max_points and max_points > 0 else 1
    lat_s = lat[::step]
    lon_s = lon[::step]
    z_s = z[::step] if z is not None else None

    # Build KML content in memory
    kml = io.StringIO()
    kml.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    kml.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
    kml.write("  <Document>\n")
    kml.write(f"    <name>{name}</name>\n")

    # A simple style for points
    kml.write('    <Style id="pt">\n')
    kml.write("      <IconStyle>\n")
    kml.write(f"        <scale>{float(icon_scale):.2f}</scale>\n")
    kml.write("        <Icon>\n")
    kml.write(f"          <href>{icon_href}</href>\n")
    kml.write("        </Icon>\n")
    kml.write("      </IconStyle>\n")
    kml.write("    </Style>\n")

    # Write points
    for i in range(len(lat_s)):
        kml.write("    <Placemark>\n")
        kml.write("      <styleUrl>#pt</styleUrl>\n")
        kml.write(f"      <name>Pt {i+1}</name>\n")
        if z_s is not None:
            kml.write(
                "      <Point><coordinates>{:.8f},{:.8f},{:.2f}</coordinates></Point>\n".format(
                    float(lon_s[i]), float(lat_s[i]), float(z_s[i])
                )
            )
        else:
            kml.write(
                "      <Point><coordinates>{:.8f},{:.8f},0</coordinates></Point>\n".format(
                    float(lon_s[i]), float(lat_s[i])
                )
            )
        kml.write("    </Placemark>\n")

    kml.write("  </Document>\n")
    kml.write("</kml>\n")

    # Write KMZ (ZIP) with doc.kml inside
    with zipfile.ZipFile(kmz_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml.getvalue())

    print(f"✓ Loaded {n:,} points from {npz_file}")
    print(f"✓ Exported {len(lat_s):,} points to {kmz_file} (subsample step={step})")


def main():
    export_points_kmz(
        npz_file="authorized_points_all_masks.npz",
        kmz_file="authorized_points_all_masks_POINTS.kmz",
        max_points=5000,  # Try 5000 if Google Earth still lags
    )


if __name__ == "__main__":
    main()
