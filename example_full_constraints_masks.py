"""
Example Usage: FULL CONSTRAINTS COMBINATION

This script generates the final 'authorized_points_all_masks.npz'.
It combines:
1. Geography (Land, 50km, France, Coastline Buffer > 100m)
2. Terrain (Slope <= 15%)
3. Infrastructure (Electricity < 500m, Roads < 500m, Buildings > 1000m)
4. Environment (Protected Areas exclusion)
"""

import numpy as np
import os

# Geographic & Terrain Modules
from site_location_masks import mask_land, mask_50km, mask_french_territory, combine_masks, mask_coastline_buffer
from mask_slope import mask_slope

# Infrastructure Modules
from electrical_stations_masks import mask_electrical_from_json
from roads_masks import mask_roads_from_geojson  
from buildings_masks import mask_buildings_from_geojson
from protected_areas_mask import mask_protected_areas_from_geojson

# Visualization Modules
from visualize_site_location_masks import plot_masks_overlay
from export_site_location_masks_kml import export_masks_to_kmz

def load_terrain_npz(npz_file: str):
    data = np.load(npz_file)
    return data['lat'], data['lon'], data['ter']

def main():
    print("="*70)
    print("FINAL SITE SELECTION - ALL CONSTRAINTS")
    print("="*70)
    
    # ---------------------------------------------------------
    # 1. SETUP & DATA LOADING
    # ---------------------------------------------------------
    # Verify which file you are using!
    terrain_file = 'terrain_mat.npz' 
    if not os.path.exists(terrain_file):
        # Fallback if you are using the other naming convention
        terrain_file = 'terrain_req01_50km.npz'
    
    print(f"\n1. Loading terrain from: {terrain_file}")
    try:
        lats, lons, Z = load_terrain_npz(terrain_file)
        print(f"   Grid size: {len(lats)} x {len(lons)}")
    except Exception as e:
        print(f"   Error: {e}")
        return

    nice_lat, nice_lon = 43.6584, 7.2159 # Nice Airport

    # ---------------------------------------------------------
    # 2. GEOGRAPHICAL MASKS
    # ---------------------------------------------------------
    print("\n2. Computing Geographical Masks...")
    
    # Basic Land & 50km & France
    m_50km = mask_50km(lats, lons, nice_lat, nice_lon, radius_km=50.0)
    m_france = mask_french_territory(lats, lons)
    
    # REQ_03: Seaside Buffer (Stricter than mask_land)
    print("   -> Coastline Buffer (> 100m)")
    m_seaside = mask_coastline_buffer(lats, lons, Z, buffer_m=100.0)

    # ---------------------------------------------------------
    # 3. TERRAIN MASKS (Slope)
    # ---------------------------------------------------------
    print("\n3. Computing Terrain Masks...")
    
    # REQ_10: Slope <= 15%
    print("   -> Slope Constraint (<= 15%)")
    m_slope = mask_slope(terrain_file, max_slope_percent=15.0)

    # ---------------------------------------------------------
    # 4. INFRASTRUCTURE MASKS
    # ---------------------------------------------------------
    print("\n4. Computing Infrastructure Masks...")

    # REQ_06: Electricity < 500m
    elec_file = 'page1.json'
    m_elec = None
    if os.path.exists(elec_file):
        print("   -> Electrical Grid Proximity (< 500m)")
        m_elec = mask_electrical_from_json(lats, lons, elec_file, radius_m=500.0)
    else:
        print(f"   [!] Missing {elec_file}, skipping electricity mask.")

    # REQ_05: Roads < 500m (Updated requirement)
    roads_file = 'roads_nice_50km.geojson'
    m_roads = None
    if os.path.exists(roads_file):
        print("   -> Road Network Proximity (< 500m)")
        m_roads = mask_roads_from_geojson(lats, lons, roads_file, max_distance_m=500.0, major_roads_only=True)
    else:
        print(f"   [!] Missing {roads_file}, skipping roads mask.")

    # REQ_02/03: Buildings > 1000m
    build_file = 'buildings.geojson'
    m_build = None
    if os.path.exists(build_file):
        print("   -> Buildings Exclusion (> 1000m)")
        m_build = mask_buildings_from_geojson(lats, lons, build_file, radius_m=1000.0)
    else:
        print(f"   [!] Missing {build_file}, skipping buildings mask.")

    # ---------------------------------------------------------
    # 5. ENVIRONMENTAL MASKS
    # ---------------------------------------------------------
    print("\n5. Computing Environmental Masks...")
    
    # Protected Areas (National Parks, etc.)
    prot_file = 'protected_areas.geojson'
    m_prot = None
    if os.path.exists(prot_file):
        print("   -> Protected Areas Exclusion (Mercantour, etc.)")
        m_prot = mask_protected_areas_from_geojson(lats, lons, prot_file)
    else:
        print(f"   [!] Missing {prot_file}, skipping protected areas.")

    # ---------------------------------------------------------
    # 6. COMBINATION & EXPORT
    # ---------------------------------------------------------
    print("\n6. Combining ALL Constraints...")
    
    # List of active masks
    masks_list = [m_50km, m_france, m_seaside, m_slope]
    
    # Dictionary for visualization
    masks_dict = {
        '50km Radius': m_50km,
        'French Territory': m_france,
        'Seaside Buffer (>100m)': m_seaside,
        'Slope (<=15%)': m_slope
    }

    # Add optional masks if they exist
    if m_elec is not None:
        masks_list.append(m_elec)
        masks_dict['Electricity (<500m)'] = m_elec
    
    if m_roads is not None:
        masks_list.append(m_roads)
        masks_dict['Roads (<500m)'] = m_roads
        
    if m_build is not None:
        masks_list.append(m_build)
        masks_dict['Buildings (>1000m)'] = m_build
        
    if m_prot is not None:
        masks_list.append(m_prot)
        masks_dict['Protected Areas'] = m_prot

    # Logical AND combination
    final_mask = combine_masks(*masks_list)
    masks_dict['FINAL CANDIDATES'] = final_mask
    
    total_points = final_mask.size
    valid_points = np.sum(final_mask)
    print(f"\n   >>> RESULT: {valid_points:,} valid points found out of {total_points:,} ({valid_points/total_points*100:.2f}%)")

    # Save to NPZ
    output_npz = "authorized_points_all_masks.npz"
    i, j = np.where(final_mask)
    np.savez(
        output_npz,
        lat=lats[i],
        lon=lons[j],
        z=Z[i, j],
        mask=final_mask
    )
    print(f"   [OK] Data saved to {output_npz}")

    # ---------------------------------------------------------
    # 7. VISUALIZATION
    # ---------------------------------------------------------
    print("\n7. Generating Visualizations...")
    
    # KMZ Export (Subsampled for performance)
    try:
        step = 10 # Adjust if too slow
        export_masks_to_kmz(
            {k: v[::step, ::step] for k, v in masks_dict.items()},
            lats[::step], lons[::step],
            "final_site_selection.kmz"
        )
        print("   [OK] KMZ exported: final_site_selection.kmz")
    except Exception as e:
        print(f"   [Warn] KMZ export failed: {e}")

if __name__ == "__main__":
    main()