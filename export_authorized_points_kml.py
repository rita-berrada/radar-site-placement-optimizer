import numpy as np

def export_points_kml(npz_file: str, kml_file: str, max_points: int = 50000):
    data = np.load(npz_file)
    lat = data["lat"]
    lon = data["lon"]
    z = data["z"] if "z" in data else None

    n = len(lat)
    if n == 0:
        raise RuntimeError("No authorized points in NPZ.")

    # Subsample if too many points (Google Earth will choke above ~50k-200k)
    step = max(1, n // max_points)
    lat = lat[::step]
    lon = lon[::step]
    if z is not None:
        z = z[::step]

    with open(kml_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
        f.write("  <Document>\n")
        f.write("    <name>Authorized points (ALL masks)</name>\n")

        # Style: visible icon (red dot) and decent scale
        f.write("    <Style id=\"pt\">\n")
        f.write("      <IconStyle>\n")
        f.write("        <scale>0.8</scale>\n")
        f.write("        <Icon>\n")
        f.write("          <href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href>\n")
        f.write("        </Icon>\n")
        f.write("      </IconStyle>\n")
        f.write("    </Style>\n")

        for i in range(len(lat)):
            f.write("    <Placemark>\n")
            f.write("      <styleUrl>#pt</styleUrl>\n")
            f.write(f"      <name>Pt {i+1}</name>\n")
            if z is not None:
                f.write("      <Point><coordinates>{:.8f},{:.8f},{:.2f}</coordinates></Point>\n".format(
                    float(lon[i]), float(lat[i]), float(z[i])
                ))
            else:
                f.write("      <Point><coordinates>{:.8f},{:.8f},0</coordinates></Point>\n".format(
                    float(lon[i]), float(lat[i])
                ))
            f.write("    </Placemark>\n")

        f.write("  </Document>\n")
        f.write("</kml>\n")

    print(f"✓ Wrote {kml_file} with {len(lat)} points (subsample step={step})")


def main():
    export_points_kml(
        npz_file="authorized_points_all_masks.npz",
        kml_file="authorized_points_all_masks_POINTS.kml",
        max_points=50000  # ajuste si tu veux plus/moins
    )

if __name__ == "__main__":
    main()
