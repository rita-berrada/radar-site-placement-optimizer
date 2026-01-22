#!/usr/bin/env python3
"""
BUILDINGS (HABITATIONS) VISUALIZATION - UNIVERSAL VERSION
For the entire team (Mac, Windows, Linux)

INSTRUCTIONS FOR EVERYONE:
=========================

1. Create a project folder (e.g., "radar_thales")

2. Put these files IN THIS FOLDER:
   - terrain_req01_50km.npz   (preferred DTED extract)  OR  terrain_mat.npz
   - geographical_data/buildings.geojson        (building points)
   - buildings.py             (this file)

3. Run:
   python buildings.py

OUTPUTS:
========
- buildings_map.png
- buildings_zoom_10km.png

Notes
=====
- buildings.geojson should contain Point geometries (one per building).
- This script plots:
  (1) Full terrain map + buildings
  (2) Zoom around Nice Airport (10 km)

- It also draws 1 km radius circles (REQ: radar must be > 1 km from habitations)
  - Full map: circles shown only for buildings within 15 km (avoid clutter)
  - Zoom map: circles shown for all buildings within 10 km
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Optional (faster geojson reading); script works without it
try:
    import geopandas as gpd
except Exception:
    gpd = None

# ============================================================================
# HEADER
# ============================================================================
print("\n" + "=" * 70)
print("🏠 BUILDINGS (HABITATIONS) VISUALIZATION")
print("=" * 70)
print(f"\n📂 Working directory: {os.getcwd()}")

# ============================================================================
# REQUIRED FILES CHECK
# ============================================================================
terrain_candidates = ["terrain_req01_50km.npz", "terrain_mat.npz"]
terrain_file = None
for cand in terrain_candidates:
    if os.path.exists(cand):
        terrain_file = cand
        break

# Check for buildings.geojson in geographical_data folder
buildings_path = 'geographical_data/buildings.geojson'

required_files = {
    buildings_path: 'Buildings data (GeoJSON Points)'
}

missing_files = []
for filename, description in required_files.items():
    if not os.path.exists(filename):
        missing_files.append(f"   ❌ {filename} ({description})")
    else:
        file_size = os.path.getsize(filename) / (1024*1024)  # MB
        print(f"   ✅ {filename} ({file_size:.1f} MB)")

if terrain_file is None:
    missing_files.append("   ❌ terrain_req01_50km.npz or terrain_mat.npz (Terrain file DTED)")

if missing_files:
    print("\n❌ Missing files:")
    for m in missing_files:
        print(m)
    print("\n✅ Fix:")
    print("   1. Put the missing files in this folder")
    print("   2. Make sure buildings.geojson is in geographical_data/ subfolder")
    print("   3. Re-run the script")
    sys.exit(1)

print(f"   ✅ {terrain_file} (Terrain file DTED)")

# ============================================================================
# CONSTANTS
# ============================================================================
# Nice Airport (LFMN)
NICE_AIRPORT_LAT = 43.6584
NICE_AIRPORT_LON = 7.2159

# Requirement radii
REQ_RADIUS_M = 1000  # 1 km from buildings
CIRCLES_MAX_DIST_M = 15000  # show circles on full map only for buildings < 15 km
ZOOM_MAX_DIST_M = 10000     # zoom view: buildings < 10 km

# ============================================================================
# TERRAIN LOADING
# ============================================================================
print("\n📂 Loading terrain...")
try:
    terrain = np.load(terrain_file)
    lats = terrain['lat'].astype(float)
    lons = terrain['lon'].astype(float)
    Z = terrain['ter'].astype(float)
    print(f"✅ Terrain loaded: {len(lats)} x {len(lons)} points")
    print(f"   Latitude: {lats.min():.4f}° to {lats.max():.4f}°")
    print(f"   Longitude: {lons.min():.4f}° to {lons.max():.4f}°")
except Exception as e:
    print(f"❌ Terrain loading error: {e}")
    sys.exit(1)

# ============================================================================
# BUILDINGS LOADING
# ============================================================================
print("\n📂 Loading buildings...")
try:
    buildings = []

    if gpd is not None:
        print("   Using geopandas for faster loading...")
        gdf = gpd.read_file(buildings_path)
        gdf = gdf[gdf.geometry.type == 'Point']
        xs = gdf.geometry.x.to_numpy(dtype=float)
        ys = gdf.geometry.y.to_numpy(dtype=float)
        for lat, lon in zip(ys, xs):
            buildings.append({'lat': float(lat), 'lon': float(lon)})
    else:
        print("   Using json for loading (slower but works without geopandas)...")
        with open(buildings_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for feat in data.get('features', []):
            geom = feat.get('geometry', {})
            if geom.get('type') != 'Point':
                continue
            coords = geom.get('coordinates', None)
            if not coords or len(coords) < 2:
                continue
            lon, lat = coords[0], coords[1]
            buildings.append({'lat': float(lat), 'lon': float(lon)})

    print(f"✅ {len(buildings)} buildings loaded")

except Exception as e:
    print(f"❌ Buildings reading error: {e}")
    print(f"   Make sure {buildings_path} exists and is valid GeoJSON")
    sys.exit(1)

# ============================================================================
# DISTANCE TO AIRPORT
# ============================================================================
print("\n📏 Computing distances to Nice Airport...")

# Vectorized haversine (meters)
lat_arr = np.array([b['lat'] for b in buildings], dtype=float)
lon_arr = np.array([b['lon'] for b in buildings], dtype=float)

R = 6371000.0
phi1 = np.radians(lat_arr)
phi2 = np.radians(NICE_AIRPORT_LAT)
dphi = np.radians(NICE_AIRPORT_LAT - lat_arr)
dlmb = np.radians(NICE_AIRPORT_LON - lon_arr)
a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
dist_m = 2 * R * np.arcsin(np.sqrt(a))

for b, d in zip(buildings, dist_m):
    b['distance_m'] = float(d)

print(f"   Min distance: {min(b['distance_m'] for b in buildings):.0f}m")
print(f"   Max distance: {max(b['distance_m'] for b in buildings)/1000:.1f}km")

# ============================================================================
# PREPARE GRID FOR PLOTTING
# ============================================================================
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Convert 1 km to degrees in longitude (same trick as elecstations.py)
# (good enough for visualization)
radius_deg_lon = REQ_RADIUS_M / (111000 * np.cos(np.radians(43.6)))

# ============================================================================
# FULL MAP PLOT
# ============================================================================
print("\n🗺️ Plotting full map...")

plt.figure(figsize=(16, 15))

# Terrain
contour = plt.contourf(lon_grid, lat_grid, Z, levels=40, cmap='terrain', alpha=0.9)
cbar = plt.colorbar(contour, pad=0.02)
cbar.set_label('Altitude (m)', fontsize=12, fontweight='bold')

# Airport
plt.scatter([NICE_AIRPORT_LON], [NICE_AIRPORT_LAT],
            marker='*', s=250, color='red', edgecolor='white',
            linewidth=1.5, label='Nice Airport', zorder=10)

# Buildings points
plt.scatter(lon_arr, lat_arr, s=6, color='yellow', alpha=0.35,
            label=f'Buildings ({len(buildings)})', zorder=9)

# Circles (limited to nearby buildings)
print("⭕ Adding 1 km radius circles...")
buildings_for_circles = [b for b in buildings if b['distance_m'] < CIRCLES_MAX_DIST_M]
print(f"   {len(buildings_for_circles)} circles displayed (buildings < {CIRCLES_MAX_DIST_M/1000:.0f} km)")

ax = plt.gca()
for i, b in enumerate(buildings_for_circles):
    if i == 0:
        circle = Circle((b['lon'], b['lat']),
                        radius_deg_lon,
                        fill=False, edgecolor='yellow', linestyle='--',
                        linewidth=1, alpha=0.25,
                        label='1 km Radius (buildings)', zorder=8)
    else:
        circle = Circle((b['lon'], b['lat']),
                        radius_deg_lon,
                        fill=False, edgecolor='yellow', linestyle='--',
                        linewidth=1, alpha=0.25, zorder=8)
    ax.add_patch(circle)

# Formatting
plt.xlabel('Longitude (°)', fontsize=14, fontweight='bold')
plt.ylabel('Latitude (°)', fontsize=14, fontweight='bold')
plt.title('Terrain Map + Buildings (Habitations)\n' +
          '1 km radius around nearby buildings (REQ)',
          fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='upper left', fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
output_full = 'buildings_map.png'
plt.savefig(output_full, dpi=200, bbox_inches='tight')
print(f"✅ Saved: {output_full}")
plt.close()

# ============================================================================
# ZOOM MAP (10 KM AROUND AIRPORT)
# ============================================================================
print("\n🔍 Plotting zoom map (10 km around airport)...")

nearby = [b for b in buildings if b['distance_m'] < ZOOM_MAX_DIST_M]
print(f"✅ {len(nearby)} buildings within 10 km")

if len(nearby) == 0:
    print("⚠️ No buildings found within 10 km — zoom plot will be empty.")
    # still create a zoomed view around airport bounds
    lat_min = NICE_AIRPORT_LAT - 0.09
    lat_max = NICE_AIRPORT_LAT + 0.09
    lon_min = NICE_AIRPORT_LON - 0.12
    lon_max = NICE_AIRPORT_LON + 0.12
else:
    lat_nearby = [b['lat'] for b in nearby]
    lon_nearby = [b['lon'] for b in nearby]
    lat_min, lat_max = min(lat_nearby) - 0.02, max(lat_nearby) + 0.02
    lon_min, lon_max = min(lon_nearby) - 0.02, max(lon_nearby) + 0.02

# Terrain filtering for zoom area
lat_mask = (lats >= lat_min) & (lats <= lat_max)
lon_mask = (lons >= lon_min) & (lons <= lon_max)

Z_zoom = Z[np.ix_(lat_mask, lon_mask)]
lats_zoom = lats[lat_mask]
lons_zoom = lons[lon_mask]
lon_grid_zoom, lat_grid_zoom = np.meshgrid(lons_zoom, lats_zoom)

plt.figure(figsize=(18, 16))

# Terrain zoom
contour = plt.contourf(lon_grid_zoom, lat_grid_zoom, Z_zoom, levels=40, cmap='terrain', alpha=0.9)
cbar = plt.colorbar(contour, pad=0.02)
cbar.set_label('Altitude (m)', fontsize=12, fontweight='bold')

# Airport
plt.scatter([NICE_AIRPORT_LON], [NICE_AIRPORT_LAT],
            marker='*', s=300, color='red', edgecolor='white',
            linewidth=1.5, label='Nice Airport', zorder=10)

# Buildings nearby
if len(nearby) > 0:
    near_lon = np.array([b['lon'] for b in nearby], dtype=float)
    near_lat = np.array([b['lat'] for b in nearby], dtype=float)
    plt.scatter(near_lon, near_lat, s=10, color='yellow', alpha=0.45,
                label=f'Buildings (<10 km) ({len(nearby)})', zorder=9)

    # Circles for all nearby buildings (zoom)
    print("⭕ Adding 1 km circles for zoom buildings...")
    ax = plt.gca()
    for i, b in enumerate(nearby):
        if i == 0:
            circle = Circle((b['lon'], b['lat']),
                            radius_deg_lon,
                            fill=False, edgecolor='yellow', linestyle='--',
                            linewidth=1, alpha=0.25,
                            label='1 km Radius (REQ_buildings)', zorder=8)
        else:
            circle = Circle((b['lon'], b['lat']),
                            radius_deg_lon,
                            fill=False, edgecolor='yellow', linestyle='--',
                            linewidth=1, alpha=0.25, zorder=8)
        ax.add_patch(circle)
else:
    ax = plt.gca()

# Formatting
plt.xlabel('Longitude (°)', fontsize=14, fontweight='bold')
plt.ylabel('Latitude (°)', fontsize=14, fontweight='bold')
plt.title('Zoom < 10 km around Nice Airport\n' +
          'Terrain + Buildings (1 km exclusion circles)',
          fontsize=16, fontweight='bold', pad=20)

# Stats box (same style as elecstations.py)
stats_text = f"""ZOOM < 10 KM
━━━━━━━━━━━━━
Buildings: {len(nearby)}
All with 1 km circle
━━━━━━━━━━━━━"""
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.95),
        family='monospace')

plt.legend(loc='upper left', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()

output_zoom = 'buildings_zoom_10km.png'
plt.savefig(output_zoom, dpi=200, bbox_inches='tight')
print(f"✅ Saved: {output_zoom}")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("✅ DONE")
print("=" * 70)

print(f"\n📁 Outputs created:")
print(f"   - {output_full}")
print(f"   - {output_zoom}")

print(f"\n📌 Summary:")
print(f"   ✅ Buildings loaded: {len(buildings)}")
print(f"   ✅ Min distance to airport: {min(b['distance_m'] for b in buildings):.0f}m")
print(f"   ✅ Max distance to airport: {max(b['distance_m'] for b in buildings)/1000:.1f}km")
print(f"      → Full view: {len(buildings_for_circles)} circles < {CIRCLES_MAX_DIST_M/1000:.0f} km")
print(f"      → Zoomed view: {len(nearby)} buildings < 10 km, all with 1 km circle")

print(f"\n💡 Project constraints:")
print(f"   ✅  Radar must be > 1 km from habitations (yellow circles)")
print(f"   ✅ {len(buildings)} building points identified")

print("\n🚀 To share with the team:")
print("   - Send this script (buildings.py)")
print("   - Send the terrain .npz file")
print("   - Send geographical_data/buildings.geojson")
print("   → Everyone can generate the maps on their computer!")

print("\n" + "=" * 70)