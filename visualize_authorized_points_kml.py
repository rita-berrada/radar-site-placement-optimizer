import numpy as np
from export_site_location_masks_kml import export_masks_to_kmz
from example_infrastructure_masks import load_terrain_npz


def main():
    print("Visualizing authorized points (ALL masks) in Google Earth")

    # 1. Load terrain (same grid as masks)
    terrain_file = "terrain_req01_50km.npz"
    lats, lons, Z = load_terrain_npz(terrain_file)

    # 2. Load authorized points
    data = np.load("authorized_points_all_masks.npz")
    auth_lat = data["lat"]
    auth_lon = data["lon"]

    # 3. Rebuild mask
    authorized_mask = np.zeros((len(lats), len(lons)), dtype=bool)

    lat_to_idx = {lat: i for i, lat in enumerate(lats)}
    lon_to_idx = {lon: j for j, lon in enumerate(lons)}

    for lat, lon in zip(auth_lat, auth_lon):
        i = lat_to_idx[lat]
        j = lon_to_idx[lon]
        authorized_mask[i, j] = True

    # 4. Export using EXISTING exporter
    export_masks_to_kmz(
        {"Authorized points (ALL masks)": authorized_mask},
        lats,
        lons,
        "authorized_points_all_masks.kmz"
    )

    print("✓ KMZ exported: authorized_points_all_masks.kmz")


if __name__ == "__main__":
    main()
