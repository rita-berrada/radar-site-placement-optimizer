"""
FINAL SITE SELECTION MAIN SCRIPT

This script generates the final 'authorized_points_all_masks.npz'.
It combines:
1. Geography (Land, 50km, France, Coastline Buffer > 100m)
2. Terrain (Slope <= 15%)
3. Infrastructure (Electricity < 500m, Roads < 500m)
4. Habitation Exclusion (Optimized Zones + Safety Buffer) <--- MODIFIED
5. Environment (Protected Areas exclusion)
6. RADAR PERFORMANCE (Line of Sight to Nice Airport) <- FINAL FILTER
"""

import numpy as np
import os

# Geographic & Terrain Modules
from site_location_masks import mask_land, mask_50km, mask_french_territory, combine_masks, mask_coastline_buffer
from mask_slope import mask_slope

# Infrastructure Modules
from electrical_stations_masks import mask_electrical_from_json
from roads_masks import mask_roads_from_geojson  
# On utilise le masque 'buildings' car il gère le rayon de sécurité (buffer)
from buildings_masks import mask_buildings_from_geojson 

# Environmental Modules
from protected_areas_mask import mask_protected_areas_from_geojson

# Radar Logic Module (The Final Check)
from mask_see_airport import check_visibility_batch

# Visualization Modules
from visualize_site_location_masks import plot_masks_overlay
from export_site_location_masks_kml import export_masks_to_kmz

def load_terrain_npz(npz_file: str):
    data = np.load(npz_file)
    return data['lat'], data['lon'], data['ter']

def main():
    print("="*70)
    print("FINAL SITE SELECTION - OPTIMIZED ZONES + LOS")
    print("="*70)
    
    # ---------------------------------------------------------
    # 1. SETUP & DATA LOADING
    # ---------------------------------------------------------
    terrain_file = 'terrain_mat.npz' 
    if not os.path.exists(terrain_file):
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
    # 4. INFRASTRUCTURE & HABITATION (OPTIMIZED)
    # ---------------------------------------------------------
    print("\n4. Computing Infrastructure & Habitation Masks...")

    # REQ_06: Electricity < 500m
    elec_file = 'page1.json'
    m_elec = None
    if os.path.exists(elec_file):
        print("   -> Electrical Grid Proximity (< 500m)")
        m_elec = mask_electrical_from_json(lats, lons, elec_file, radius_m=500.0)
    else:
        print(f"   [!] Missing {elec_file}, skipping electricity mask.")

    # REQ_05: Roads < 500m
    roads_file = 'roads_nice_50km.geojson'
    m_roads = None
    if os.path.exists(roads_file):
        print("   -> Road Network Proximity (< 500m)")
        m_roads = mask_roads_from_geojson(lats, lons, roads_file, max_distance_m=500.0, major_roads_only=True)
    else:
        print(f"   [!] Missing {roads_file}, skipping roads mask.")

    # --- ZONES INTERDITES FUSIONNÉES (Résidentiel + Bâtiments) ---
    forbidden_file = 'forbidden_zones_optimized.geojson'
    m_forbidden = None
    
    if os.path.exists(forbidden_file):
        # PARAMÈTRE CRITIQUE : LE RAYON DE SÉCURITÉ
        # 300m est un bon compromis pour éviter les nuisances sans tout bloquer.
        # Si vous avez 0 candidat, baissez à 150m ou 200m.
        safety_radius = 300.0 
        
        print(f"   -> Forbidden Zones Exclusion (Source: {forbidden_file})")
        print(f"      Applying safety buffer of {safety_radius}m around zones...")
        
        m_forbidden = mask_buildings_from_geojson(
            lats, lons, 
            forbidden_file, 
            radius_m=safety_radius
        )
    else:
        print(f"   [!] Missing {forbidden_file}. PLEASE RUN optimize_zones.py FIRST!")

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
    # 6. COMBINATION (PRE-VISIBILITY)
    # ---------------------------------------------------------
    print("\n6. Combining Constraints (Pre-LOS)...")
    
    masks_list = [m_50km, m_france, m_seaside, m_slope]
    
    # Dictionary for visualization
    masks_dict = {
        '50km Radius': m_50km,
        'French Territory': m_france,
        'Seaside Buffer (>100m)': m_seaside,
        'Slope (<=15%)': m_slope
    }

    if m_elec is not None:
        masks_list.append(m_elec)
        masks_dict['Electricity (<500m)'] = m_elec
    
    if m_roads is not None:
        masks_list.append(m_roads)
        masks_dict['Roads (<500m)'] = m_roads
        
    if m_forbidden is not None:
        masks_list.append(m_forbidden)
        # On l'appelle Habitation pour la clarté dans Google Earth
        masks_dict[f'Habitation (>{int(safety_radius)}m)'] = m_forbidden 
        
    if m_prot is not None:
        masks_list.append(m_prot)
        masks_dict['Protected Areas'] = m_prot

    # Logical AND combination
    pre_los_mask = combine_masks(*masks_list)
    pre_los_count = np.sum(pre_los_mask)
    print(f"   -> {pre_los_count:,} candidates before Line of Sight check.")

    # ---------------------------------------------------------
    # 7. LINE OF SIGHT CHECK (The Final Filter)
    # ---------------------------------------------------------
    print("\n7. Checking Line of Sight to Airport (Final Filter)...")
    
    if pre_los_count == 0:
        print("   [!] No candidates left to check visibility.")
        final_mask = pre_los_mask
    else:
        # Get indices of remaining candidates
        cand_indices = np.where(pre_los_mask)
        
        # Run Ray Tracing only on these points
        # Radar Height = 20m, Target Height = 10m
        is_visible = check_visibility_batch(
            lats, lons, Z, 
            cand_indices, 
            nice_lat, nice_lon, 
            radar_height_m=20.0,
            target_height_m=10.0
        )
        
        # Create final mask
        final_mask = np.zeros_like(pre_los_mask)
        
        # Filter: Keep only points that were valid AND are visible
        valid_rows = cand_indices[0][is_visible]
        valid_cols = cand_indices[1][is_visible]
        
        final_mask[valid_rows, valid_cols] = True
        
        los_passed = np.sum(final_mask)
        print(f"   -> {los_passed:,} candidates have clear Line of Sight.")
        print(f"   -> Rejected {pre_los_count - los_passed:,} points due to obstruction.")

    # Update visualization dict
    masks_dict['FINAL CANDIDATES'] = final_mask
    
    # ---------------------------------------------------------
    # 8. EXPORT
    # ---------------------------------------------------------
    print("\n8. Saving Results...")

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

    # KMZ Export
    try:
        step = 10 
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