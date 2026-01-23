#!/usr/bin/env python3
"""
ROADS VISUALIZATION
Generates vibrant maps of the road network and proximity constraints.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Try importing geopandas for performance
try:
    import geopandas as gpd
except ImportError:
    gpd = None

# ============================================================================
# CONFIGURATION
# ============================================================================
NICE_AIRPORT_LAT = 43.6584
NICE_AIRPORT_LON = 7.2159
ROAD_BUFFER_M = 500.0 
MAJOR_ROAD_TYPES = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary'}

ROADS_PATH = 'geographical_data/roads_nice_50km.geojson'
TERRAIN_CANDIDATES = ["terrain_req01_50km.npz", "terrain_mat.npz"]

# ============================================================================
# FILE CHECK
# ============================================================================
terrain_file = None
for cand in TERRAIN_CANDIDATES:
    if os.path.exists(cand):
        terrain_file = cand
        break

if not terrain_file or not os.path.exists(ROADS_PATH):
    print("❌ Error: Missing files (terrain .npz or roads .geojson).")
    sys.exit(1)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("⏳ Loading terrain and roads...")

# Load Terrain
terrain = np.load(terrain_file)
lats, lons, Z = terrain['lat'], terrain['lon'], terrain['ter']

# Load Roads
roads_latlon = []
try:
    if gpd is not None:
        # Fast loading with Geopandas
        gdf = gpd.read_file(ROADS_PATH)
        if 'highway' in gdf.columns:
            gdf = gdf[gdf['highway'].isin(MAJOR_ROAD_TYPES)]
        
        for geom in gdf.geometry:
            if geom.geom_type == 'LineString':
                roads_latlon.append(list(geom.coords))
            elif geom.geom_type == 'MultiLineString':
                for part in geom.geoms:
                    roads_latlon.append(list(part.coords))
    else:
        # Fallback loading with JSON
        with open(ROADS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for feat in data.get('features', []):
            if feat.get('properties', {}).get('highway', '') in MAJOR_ROAD_TYPES:
                roads_latlon.append(feat['geometry']['coordinates'])

except Exception as e:
    print(f"❌ Error loading roads: {e}")
    sys.exit(1)

print(f"✅ Loaded {len(roads_latlon)} major road segments.")

# ============================================================================
# 2. PLOT FULL MAP
# ============================================================================
print("🎨 Generating full map...")
plt.figure(figsize=(16, 12))

# Terrain (Opaque/Vibrant)
plt.contourf(lons, lats, Z, levels=40, cmap='terrain', alpha=1.0)
plt.colorbar(label='Altitude (m)')

# Roads (Red)
lc = LineCollection(roads_latlon, colors='red', linewidths=0.8, alpha=1.0, label='Major Roads')
plt.gca().add_collection(lc)

# Airport
plt.scatter(NICE_AIRPORT_LON, NICE_AIRPORT_LAT, c='white', edgecolors='black', 
            marker='*', s=250, zorder=10, label='Nice Airport')

plt.xlim(lons.min(), lons.max())
plt.ylim(lats.min(), lats.max())
plt.title(f"Terrain & Road Network\n{len(roads_latlon)} segments", fontsize=15, fontweight='bold')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc='upper right', framealpha=0.9)

out_full = "roads_map.png"
plt.savefig(out_full, dpi=150, bbox_inches='tight')
print(f"💾 Saved: {out_full}")
plt.close()

# ============================================================================
# 3. PLOT ZOOM (10km AROUND AIRPORT)
# ============================================================================
print("🔍 Generating zoom map...")
lat_min, lat_max = NICE_AIRPORT_LAT - 0.09, NICE_AIRPORT_LAT + 0.09
lon_min, lon_max = NICE_AIRPORT_LON - 0.12, NICE_AIRPORT_LON + 0.12

# Filter visible roads
visible_roads = [r for r in roads_latlon if 
                 (min(p[1] for p in r) < lat_max and max(p[1] for p in r) > lat_min and
                  min(p[0] for p in r) < lon_max and max(p[0] for p in r) > lon_min)]

plt.figure(figsize=(16, 12))

# Zoomed Terrain
lat_idx = (lats >= lat_min) & (lats <= lat_max)
lon_idx = (lons >= lon_min) & (lons <= lon_max)
plt.contourf(lons[lon_idx], lats[lat_idx], Z[np.ix_(lat_idx, lon_idx)], 
             levels=30, cmap='terrain', alpha=1.0)

# Buffer Zone (Yellow, Transparent)
lc_buffer = LineCollection(visible_roads, colors='yellow', linewidths=12, alpha=0.4, label='Zone < 500m')
plt.gca().add_collection(lc_buffer)

# Road Centerlines (Red)
lc_center = LineCollection(visible_roads, colors='red', linewidths=1.5, alpha=1.0)
plt.gca().add_collection(lc_center)

plt.scatter(NICE_AIRPORT_LON, NICE_AIRPORT_LAT, c='white', edgecolors='black', 
            marker='*', s=300, zorder=10, label='Airport')

plt.xlim(lon_min, lon_max)
plt.ylim(lat_min, lat_max)
plt.title("Zoom: Roads (Red) & Exclusion Zone (Yellow)", fontsize=15, fontweight='bold')
plt.legend(loc='upper right')

out_zoom = "roads_zoom_10km.png"
plt.savefig(out_zoom, dpi=150, bbox_inches='tight')
print(f"💾 Saved: {out_zoom}")
plt.close()