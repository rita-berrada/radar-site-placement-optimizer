#!/usr/bin/env python3
"""
ELECTRICAL STATIONS VISUALIZATION - UNIVERSAL VERSION
For the entire team (Mac, Windows, Linux)

INSTRUCTIONS FOR EVERYONE:
=========================

1. Create a project folder (e.g., "radar_thales")

2. Put these files IN THIS FOLDER:
   - terrain_req01_50km.npz
   - page1.json
   - run_stations.py (this file)

3. Open a terminal/PowerShell in this folder

4. Run:
   Mac/Linux:  python3 run_stations.py
   Windows:    python run_stations.py

5. Images will be created in the same folder!

"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Compatible with all OS
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import sys

print("=" * 70)
print("🗺️  ELECTRICAL STATIONS VISUALIZATION - ENEDIS")
print("   MSSR Radar Project - Nice Côte d'Azur")
print("=" * 70)

# ============================================================================
# FILE VERIFICATION
# ============================================================================

print(f"\n📂 Working directory: {os.getcwd()}")

# List of required files
required_files = {
    'terrain_req01_50km.npz': 'Terrain file (DTED)',
    'page1.json': 'Enedis electrical stations data'
}

missing_files = []
for filename, description in required_files.items():
    if not os.path.exists(filename):
        missing_files.append(f"   ❌ {filename} ({description})")
    else:
        file_size = os.path.getsize(filename) / (1024*1024)  # MB
        print(f"   ✅ {filename} ({file_size:.1f} MB)")

if missing_files:
    print("\n❌ MISSING FILES:")
    for msg in missing_files:
        print(msg)
    print("\n💡 SOLUTION:")
    print("   1. Put all files in the SAME FOLDER as this script")
    print("   2. Check that you're in the correct folder (see above)")
    print("   3. Re-run the script")
    sys.exit(1)

# ============================================================================
# TERRAIN LOADING
# ============================================================================

print("\n📂 Loading terrain...")
try:
    terrain = np.load('terrain_req01_50km.npz')
    lats = terrain['lat']
    lons = terrain['lon']
    Z = terrain['ter']
    print(f"✅ Terrain loaded: {len(lats)} x {len(lons)} points")
    print(f"   Latitude: {lats.min():.4f}° to {lats.max():.4f}°")
    print(f"   Longitude: {lons.min():.4f}° to {lons.max():.4f}°")
except Exception as e:
    print(f"❌ Terrain reading error: {e}")
    sys.exit(1)

# ============================================================================
# STATIONS LOADING
# ============================================================================

print("\n📂 Loading electrical stations...")
try:
    with open('page1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stations = []
    for result in data['results']:
        geopoint = result['_geopoint']
        if isinstance(geopoint, str):
            lat, lon = map(float, geopoint.split(','))
        else:
            lat = geopoint['lat']
            lon = geopoint['lon']
        
        stations.append({
            'lat': lat,
            'lon': lon,
            'distance_m': result['_geo_distance']
        })
    
    print(f"✅ {len(stations)} stations loaded")
    print(f"   Min distance: {min(s['distance_m'] for s in stations):.0f}m")
    print(f"   Max distance: {max(s['distance_m'] for s in stations)/1000:.1f}km")

except Exception as e:
    print(f"❌ Stations reading error: {e}")
    sys.exit(1)

# Nice Airport
NICE_AIRPORT = {'lat': 43.6584, 'lon': 7.2159}

# ============================================================================
# MAIN MAP CREATION
# ============================================================================

print("\n🎨 Creating main map...")

fig, ax = plt.subplots(figsize=(18, 16))

# Terrain
lon_grid, lat_grid = np.meshgrid(lons, lats)
contour = ax.contourf(lon_grid, lat_grid, Z, levels=30, cmap='terrain', alpha=0.7)
cbar = plt.colorbar(contour, ax=ax, label='Altitude (m)', pad=0.02)

# Airport
ax.plot(NICE_AIRPORT['lon'], NICE_AIRPORT['lat'], 
        'r*', markersize=25, label='Nice Airport', 
        markeredgewidth=2, markeredgecolor='white', zorder=10)

# Electrical stations
station_lats = [s['lat'] for s in stations]
station_lons = [s['lon'] for s in stations]

ax.scatter(station_lons, station_lats, 
           c='blue', s=30, marker='s', 
           label=f'Electrical Stations ({len(stations)})',
           edgecolors='white', linewidth=0.5, alpha=0.8, zorder=9)

# ============================================================================
# 500m CIRCLES (PROJECT_REQ_06)
# ============================================================================

print("⭕ Adding 500m radius circles...")

# Filter: circles for stations < 15 km (otherwise too many circles)
stations_for_circles = [s for s in stations if s['distance_m'] < 15000]
print(f"   {len(stations_for_circles)} circles displayed (stations < 15 km)")

# Convert 500m to degrees
radius_m = 500
radius_deg_lon = radius_m / (111000 * np.cos(np.radians(43.6)))

for i, station in enumerate(stations_for_circles):
    if i == 0:
        circle = Circle((station['lon'], station['lat']), 
                       radius_deg_lon,
                       fill=False, edgecolor='blue', linestyle='--', 
                       linewidth=1, alpha=0.4,
                       label='500m Radius (REQ_06)', zorder=8)
    else:
        circle = Circle((station['lon'], station['lat']), 
                       radius_deg_lon,
                       fill=False, edgecolor='blue', linestyle='--', 
                       linewidth=1, alpha=0.4, zorder=8)
    ax.add_patch(circle)

# Formatting
ax.set_xlabel('Longitude (°)', fontsize=14, fontweight='bold')
ax.set_ylabel('Latitude (°)', fontsize=14, fontweight='bold')
ax.set_title('Terrain Map + Enedis Electrical Stations\n' +
             '500m radius around each station',
             fontsize=16, fontweight='bold', pad=20)

ax.legend(loc='upper left', fontsize=12, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Statistics
stats_text = f"""STATISTICS
━━━━━━━━━━━━━━━━━
Stations: {len(stations)}
500m Circles: {len(stations_for_circles)}
Min distance: {min(s['distance_m'] for s in stations):.0f}m
Max distance: {max(s['distance_m'] for s in stations):.0f}m
━━━━━━━━━━━━━━━━━"""

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.95),
        family='monospace')

plt.tight_layout()

# Save
output = 'electrical_stations_map.png'
plt.savefig(output, dpi=200, bbox_inches='tight')
print(f"✅ Main map saved: {output}")
plt.close()

# ============================================================================
# ZOOMED MAP CREATION (< 10 km)
# ============================================================================

print("\n🔍 Creating zoomed map (stations < 10 km)...")

nearby = [s for s in stations if s['distance_m'] < 10000]
print(f"   {len(nearby)} stations to display")

fig, ax = plt.subplots(figsize=(18, 16))

# Zoom limits
lat_nearby = [s['lat'] for s in nearby]
lon_nearby = [s['lon'] for s in nearby]
lat_min, lat_max = min(lat_nearby) - 0.02, max(lat_nearby) + 0.02
lon_min, lon_max = min(lon_nearby) - 0.02, max(lon_nearby) + 0.02

# Terrain filtering for zoom area
lat_mask = (lats >= lat_min) & (lats <= lat_max)
lon_mask = (lons >= lon_min) & (lons <= lon_max)
Z_zoom = Z[np.ix_(lat_mask, lon_mask)]
lats_zoom = lats[lat_mask]
lons_zoom = lons[lon_mask]

lon_grid_zoom, lat_grid_zoom = np.meshgrid(lons_zoom, lats_zoom)

# Zoomed terrain
contour = ax.contourf(lon_grid_zoom, lat_grid_zoom, Z_zoom,
                      levels=30, cmap='terrain', alpha=0.7)
cbar = plt.colorbar(contour, ax=ax, label='Altitude (m)', pad=0.02)

# Airport
ax.plot(NICE_AIRPORT['lon'], NICE_AIRPORT['lat'], 
        'r*', markersize=30, label='Nice Airport', 
        markeredgewidth=3, markeredgecolor='white', zorder=10)

# Stations
ax.scatter(lon_nearby, lat_nearby, 
           c='blue', s=80, marker='s', 
           label=f'Electrical Stations ({len(nearby)})',
           edgecolors='white', linewidth=1.5, alpha=0.9, zorder=9)

# 500m circles for ALL nearby stations
for i, station in enumerate(nearby):
    if i == 0:
        circle = Circle((station['lon'], station['lat']), 
                       radius_deg_lon,
                       fill=False, edgecolor='blue', linestyle='--', 
                       linewidth=1.5, alpha=0.5,
                       label='500m Radius', zorder=8)
    else:
        circle = Circle((station['lon'], station['lat']), 
                       radius_deg_lon,
                       fill=False, edgecolor='blue', linestyle='--', 
                       linewidth=1.5, alpha=0.5, zorder=8)
    ax.add_patch(circle)

# Formatting
ax.set_xlabel('Longitude (°)', fontsize=14, fontweight='bold')
ax.set_ylabel('Latitude (°)', fontsize=14, fontweight='bold')
ax.set_title('Zoomed Map - Nearby Stations (< 10 km)\n' +
             'All 500m circles visible',
             fontsize=16, fontweight='bold', pad=20)

ax.legend(loc='upper left', fontsize=12, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')

stats_text = f"""ZOOM < 10 KM
━━━━━━━━━━━━━
Stations: {len(nearby)}
All with 500m circle
━━━━━━━━━━━━━"""

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.95),
        family='monospace')

plt.tight_layout()

output_zoom = 'electrical_stations_zoom_10km.png'
plt.savefig(output_zoom, dpi=200, bbox_inches='tight')
print(f"✅ Zoomed map saved: {output_zoom}")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("✅ PROCESSING COMPLETED!")
print("=" * 70)

print(f"\n📦 Files created in: {os.getcwd()}")
print(f"\n📊 Generated maps:")
print(f"   1. {output}")
print(f"      → Full view: {len(stations)} stations, {len(stations_for_circles)} circles")
print(f"\n   2. {output_zoom}")
print(f"      → Zoomed view: {len(nearby)} stations < 10 km, all with 500m circle")

print(f"\n💡 Project constraints:")
print(f"   ✅ REQ_06: Electrical access < 500m (blue circles)")
print(f"   ✅ {len(stations)} Enedis electrical stations identified")
print(f"   ✅ Max radius 50 km from airport")

print("\n🚀 To share with the team:")
print("   - Send this script (run_stations.py)")
print("   - Send terrain_req01_50km.npz")
print("   - Send page1.json")
print("   → Everyone can generate the maps on their computer!")

print("\n" + "=" * 70)
