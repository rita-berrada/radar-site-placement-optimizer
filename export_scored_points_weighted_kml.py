import numpy as np


def _format_cov(cov_dict):
    """Return a compact string like: FL5:92.1%, FL10:88.0%, ..."""
    if not isinstance(cov_dict, dict):
        return None

    # cov_dict keys may be float FLs
    fls = sorted([float(k) for k in cov_dict.keys()])
    parts = []
    for fl in fls:
        try:
            parts.append(f"FL{int(fl)}: {float(cov_dict[fl]):.1f}%")
        except Exception:
            continue
    return ", ".join(parts) if parts else None


def export_kml(
    npz_path="scored_candidates_01deg_weighted.npz",
    out_kml_path="scored_candidates_01deg_weighted.kml",
    top_k=5000,
    highlight_top=10
):
    """
    Export scored candidate points to a KML file for Google Earth.

    Expected NPZ structure (from your run_scoring weighted script):
      - scores_latlon: array shape (N,3) with columns [score, lat, lon]
      - cov_by_fl: optional object array of dicts (same order as scores_latlon)
    """
    data = np.load(npz_path, allow_pickle=True)

    scores_latlon = data["scores_latlon"]  # (N,3): score, lat, lon
    cov_by_fl = data["cov_by_fl"] if "cov_by_fl" in data.files else None

    # Sort by score descending
    order = np.argsort(scores_latlon[:, 0])[::-1]
    scores_latlon = scores_latlon[order]
    if cov_by_fl is not None:
        cov_by_fl = cov_by_fl[order]

    N = len(scores_latlon)
    K = min(N, int(top_k))

    kml = []
    kml.append('<?xml version="1.0" encoding="UTF-8"?>')
    kml.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    kml.append("<Document>")
    kml.append("<name>Weighted scored candidates</name>")

    # Styles
    kml.append("""
    <Style id="topStyle">
      <IconStyle>
        <scale>1.1</scale>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href>
        </Icon>
      </IconStyle>
      <LabelStyle><scale>0.8</scale></LabelStyle>
    </Style>

    <Style id="defaultStyle">
      <IconStyle>
        <scale>0.6</scale>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/paddle/blu-circle.png</href>
        </Icon>
      </IconStyle>
      <LabelStyle><scale>0.0</scale></LabelStyle>
    </Style>
    """)

    # Folder for all points
    kml.append("<Folder>")
    kml.append("<name>Candidate sites (weighted)</name>")
    kml.append(f"<description>Exported {K} / {N} points (top_k={top_k}).</description>")

    for i in range(K):
        score, lat, lon = scores_latlon[i]
        style = "#topStyle" if i < highlight_top else "#defaultStyle"

        cov_str = None
        if cov_by_fl is not None and i < len(cov_by_fl):
            cov_str = _format_cov(cov_by_fl[i])

        # Popup content
        desc_lines = [
            f"<b>Rank</b>: {i+1}",
            f"<b>Weighted score</b>: {score:.6f}",
            f"<b>Latitude</b>: {lat:.6f}",
            f"<b>Longitude</b>: {lon:.6f}",
        ]
        if cov_str:
            desc_lines.append(f"<b>Coverage</b>: {cov_str}")

        desc_html = "<br>".join(desc_lines)

        kml.append("<Placemark>")
        kml.append(f"<name>Rank {i+1} — {score:.4f}</name>")
        kml.append(f"<styleUrl>{style}</styleUrl>")
        kml.append("<description><![CDATA[")
        kml.append(desc_html)
        kml.append("]]></description>")
        kml.append("<Point>")
        # KML coordinates are lon,lat,alt
        kml.append(f"<coordinates>{lon:.6f},{lat:.6f},0</coordinates>")
        kml.append("</Point>")
        kml.append("</Placemark>")

    kml.append("</Folder>")
    kml.append("</Document>")
    kml.append("</kml>")

    with open(out_kml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(kml))

    print(f"✅ KML saved: {out_kml_path}")
    print("Open it with Google Earth (File > Open).")


if __name__ == "__main__":
    export_kml(
        npz_path="scored_candidates_01deg_weighted.npz",
        out_kml_path="scored_candidates_01deg_weighted.kml",
        top_k=5000,       # adjust (e.g., 1000 / 5000 / 20000)
        highlight_top=10  # top points in red
    )
