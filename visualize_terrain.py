import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_terrain_npz(npz_path: str):

    d = np.load(npz_path)
    lats = d["lat"].astype(float)
    lons = d["lon"].astype(float)
    Z = d["ter"].astype(float)

    if Z.shape != (len(lats), len(lons)):
        raise ValueError(f"Incohérence: Z{Z.shape} vs ({len(lats)}, {len(lons)})")

    return lats, lons, Z

# Load data from terrain_mat.npz
lats, lons, terrain = load_terrain_npz('terrain_mat.npz')

# Create meshgrid
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Nice Airport coordinates
nice_lat = 43.6584
nice_lon = 7.2159

# Find nearest grid point for Nice Airport elevation (faster than interpolation)
lat_idx = np.argmin(np.abs(lats - nice_lat))
lon_idx = np.argmin(np.abs(lons - nice_lon))
nice_elevation = terrain[lat_idx, lon_idx]

# 3D surface plot
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(lon_grid, lat_grid, terrain, cmap='terrain', alpha=0.8)
ax1.scatter([nice_lon], [nice_lat], [nice_elevation], c='red', s=100, label='Nice Airport')
ax1.text(nice_lon, nice_lat, nice_elevation, 'Nice Airport', fontsize=9, ha='left', va='bottom')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.set_zlabel('Elevation (m)')
ax1.set_title('3D Terrain Surface')
ax1.legend()

# 2D map with airport
ax2 = fig.add_subplot(122)
im = ax2.contourf(lon_grid, lat_grid, terrain, levels=20, cmap='terrain')
ax2.plot(nice_lon, nice_lat, 'ro', markersize=10, label='Nice Airport')
ax2.text(nice_lon, nice_lat, 'Nice Airport', fontsize=9, ha='left', va='bottom')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.set_title('2D Terrain Map')
plt.colorbar(im, ax=ax2, label='Elevation (m)')
ax2.legend()

plt.tight_layout()
plt.show()
