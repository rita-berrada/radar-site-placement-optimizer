#!/usr/bin/env python3
"""
TERRAIN + ROADS ONLY
Clean map with terrain and road network only

INSTRUCTIONS:
1. Put these files in the same folder:
   - terrain_req01_50km.npz
   - roads_nice_50km.geojson
   - terrain_roads_only.py (this file)
2. Run: python3 terrain_roads_only.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

print("=" * 70)
print("🗺️  TERRAIN + ROADS ONLY")
print("=" * 70)

# ============================================================================
# TERRAIN LOADING
# ============================================================================

print("\n📂 Loading terrain...")
terrain = np.load('terrain_req01_50km.npz')
lats = terrain['lat']
lons = terrain['lon']
Z = terrain['ter']
print(f"✅ Terrain: {len(lats)} x {len(lons)} points")

# ============================================================================
# ROADS LOADING (GeoJSON)
# ============================================================================

print("\n📂 Loading roads from GeoJSON...")
roads_file = 'roads_nice_50km.geojson'

if not os.path.exists(roads_file):
    print(f"⚠️  WARNING: {roads_file} not found!")
    roads_data = None
else:
    with open(roads_file, 'r', encoding='utf-8') as f:
        roads_data = json.load(f)
    
    # Count roads
    if 'features' in roads_data:
        n_roads = len(roads_data['features'])
    elif 'elements' in roads_data:
        # Convert OSM JSON to GeoJSON
        print("   Converting OSM JSON to GeoJSON...")
        features = []
        for element in roads_data['elements']:
            if element.get('type') == 'way' and 'geometry' in element:
                feature = {
                    'type': 'Feature',
                    'properties': element.get('tags', {}),
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': [
                            [node['lon'], node['lat']] 
                            for node in element['geometry']
                        ]
                    }
                }
                features.append(feature)
        roads_data = {'type': 'FeatureCollection', 'features': features}
        n_roads = len(features)
    else:
        roads_data = None
        n_roads = 0
    
    if roads_data:
        print(f"✅ {n_roads} roads loaded")

# Nice Airport
NICE_AIRPORT = {'lat': 43.6584, 'lon': 7.2159}

# ============================================================================
# MAP CREATION - TERRAIN + ROADS ONLY
# ============================================================================

print("\n🎨 Creating map (terrain + roads)...")

fig, ax = plt.subplots(figsize=(20, 18))

# 1. TERRAIN (background)
lon_grid, lat_grid = np.meshgrid(lons, lats)
contour = ax.contourf(lon_grid, lat_grid, Z, levels=30, cmap='terrain', alpha=0.7)
cbar = plt.colorbar(contour, ax=ax, label='Altitude (m)', pad=0.02)

# 2. ROADS - RED COLOR
if roads_data:
    print("   → Adding roads...")
    road_count = 0
    for feature in roads_data['features']:
        if feature['geometry']['type'] == 'LineString':
            coords = feature['geometry']['coordinates']
            lons_road = [c[0] for c in coords]
            lats_road = [c[1] for c in coords]
            
            # Single red color for all roads
            ax.plot(lons_road, lats_road, 
                   color='red',
                   linewidth=1.5,
                   alpha=0.7,
                   zorder=5)
            road_count += 1
    print(f"   ✅ {road_count} roads plotted")

# 3. AIRPORT
ax.plot(NICE_AIRPORT['lon'], NICE_AIRPORT['lat'], 
        'r*', markersize=30, label='Nice Airport', 
        markeredgewidth=3, markeredgecolor='white', zorder=10)

# ============================================================================
# FORMATTING
# ============================================================================

ax.set_xlabel('Longitude (°)', fontsize=14, fontweight='bold')
ax.set_ylabel('Latitude (°)', fontsize=14, fontweight='bold')
ax.set_title('MSSR Radar Project - Nice Airport\n' +
             'Terrain Map + Road Network (50 km radius)',
             fontsize=16, fontweight='bold', pad=20)

# Simple legend
legend_elements = [
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='r', 
               markersize=15, label='Nice Airport'),
    plt.Line2D([0], [0], color='red', linewidth=2,
               label=f'Road Network ({n_roads if roads_data else 0} roads)')
]

ax.legend(handles=legend_elements, loc='upper left', 
         fontsize=12, framealpha=0.95, edgecolor='black')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()

# Save
output = 'terrain_roads_only.png'
plt.savefig(output, dpi=200, bbox_inches='tight')
print(f"\n✅ Map saved: {output}")
plt.close()

print("\n" + "=" * 70)
print("✅ DONE!")
print("=" * 70)
print(f"\n📊 Map includes:")
print(f"   ✅ Terrain elevation (50km radius)")
if roads_data:
    print(f"   ✅ Road network ({n_roads} roads)")
else:
    print(f"   ⚠️  Roads not included")
print(f"   ✅ Nice Airport marker")
print(f"\n💡 Files needed:")
print(f"   - terrain_req01_50km.npz")
print(f"   - roads_nice_50km.geojson")
print("\n" + "=" * 70)
