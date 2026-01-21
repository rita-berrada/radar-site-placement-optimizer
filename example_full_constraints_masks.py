"""
FINAL SITE SELECTION MAIN SCRIPT

This script generates the final 'authorized_points_all_masks.npz'.
It acts as the Coordinator:
1. Loads Terrain & Converts to Metric System (ENU) with Earth Curvature Drop.
2. Calculates all masks using X, Y (Meters) and Z_corrected (Curved Earth).
3. EXCEPT for Land/Sea masks which use Z_raw (Real Altitude).
4. Exports the final result in Latitude/Longitude for Google Earth.
"""

import numpy as np
import os

# --- 1. The New Geometry Engine ---
from geo_utils_earth_curvature import load_and_convert_to_enu, REF_LAT, REF_LON

# --- 2. The Updated Masks (Metric System Compatible) ---
from site_location_masks import mask_land, mask_50km, mask_french_territory, combine_masks, mask_coastline_buffer
from mask_slope import mask_slope
from electrical_stations_masks import mask_electrical_from_json
from roads_masks import mask_roads_from_geojson  
from buildings_masks import mask_buildings_from_geojson
from mask_residential import mask_residential_from_geojson 
from protected_areas_mask import mask_protected_areas_from_geojson

# --- 3. The Physics Module (Line of Sight) ---
from mask_see_airport import check_visibility_batch

# --- 4. Visualization Modules (Existing) ---
from visualize_site_location_masks import plot_masks_overlay
from export_site_location_masks_kml import export_masks_to_kmz

def main():
    print("="*80)
    print("FINAL SITE SELECTION - METRIC SYSTEM + EARTH CURVATURE CORRECTION")
    print("="*80)
    
    # ---------------------------------------------------------
    # 1. SETUP & DATA LOADING
    # ---------------------------------------------------------
    terrain_file = 'terrain_mat.npz' 
    if not os.path.exists(terrain_file):
        # Fallback if the standard file name isn't found
        if os.path.exists('terrain_req01_50km.npz'):
            terrain_file = 'terrain_req01_50km.npz'
        else:
            print(f"[Error] No terrain .npz file found (looked for {terrain_file}).")
            return

    print(f"\n1. Loading terrain and converting to ENU Metrics...")
    
    # A. Load Corrected Data for CALCULATIONS (Physics/Geometry)
    # X_m, Y_m are 1D axes (Vectors). Z_corrected is 2D matrix.
    # lats, lons are 1D axes (Original Degrees).
    X_m, Y_m, Z_corrected, lats, lons = load_and_convert_to_enu(terrain_file)
    
    # B. Load Raw Data for EXPORT and SEA DETECTION
    # We MUST use the real altitude for the "Land Mask", otherwise the sea 
    # at 50km would be at -200m (due to curvature correction) and considered as land.
    raw_data = np.load(terrain_file)
    Z_raw = raw_data['ter']

    print(f"   Grid dimensions: {len(Y_m)} rows x {len(X_m)} cols")
    print(f"   Center (Nice Airport) is at (0, 0) meters.")

    # C. Create 2D Meshgrids
    # Most masks (Roads, Buildings) need 2D matrices of coordinates to calculate distances.
    X_grid, Y_grid = np.meshgrid(X_m, Y_m)

    # ---------------------------------------------------------
    # 2. GEOGRAPHICAL MASKS
    # ---------------------------------------------------------
    print("\n2. Computing Geographical Masks (Metric System)...")
    
    # REQ_01: 50km Radius (Using Metric Grid)
    m_50km = mask_50km(X_grid, Y_grid, radius_km=50.0)
    
    # REQ_04: French Territory (Using Metric Grid conversion)
    m_france = mask_french_territory(X_grid, Y_grid)
    
    # Land Mask (Using Z_raw > 0)
    m_land = mask_land(Z_raw)
    
    # REQ_03: Seaside Buffer (Using Metric Grid + Z_raw)
    print("   -> Coastline Buffer (> 100m)")
    m_seaside = mask_coastline_buffer(X_grid, Y_grid, Z_raw, buffer_m=100.0)

    # ---------------------------------------------------------
    # 3. TERRAIN MASKS (Slope)
    # ---------------------------------------------------------
    print("\n3. Computing Terrain Masks...")
    
    # REQ_10: Slope <= 15%
    # Note: mask_slope uses 1D axes (X_m, Y_m) and Z_corrected
    print("   -> Slope Constraint (<= 15%)")
    m_slope = mask_slope(X_m, Y_m, Z_corrected, max_slope_percent=15.0)

    # ---------------------------------------------------------
    # 4. INFRASTRUCTURE & RESIDENTIAL MASKS
    # ---------------------------------------------------------
    print("\n4. Computing Infrastructure & Residential Masks...")

    # REQ_06: Electricity < 500m
    elec_file = 'page1.json'
    m_elec = None
    if os.path.exists(elec_file):
        print("   -> Electrical Grid Proximity (< 500m)")
        m_elec = mask_electrical_from_json(X_grid, Y_grid, elec_file, radius_m=500.0)
    else:
        print(f"   [!] Missing {elec_file}, skipping electricity mask.")

    # REQ_05: Roads < 500m
    roads_file = 'roads_nice_50km.geojson'
    m_roads = None
    if os.path.exists(roads_file):
        print("   -> Road Network Proximity (< 500m)")
        m_roads = mask_roads_from_geojson(X_grid, Y_grid, roads_file, max_distance_m=500.0, major_roads_only=True)
    else:
        print(f"   [!] Missing {roads_file}, skipping roads mask.")

    # REQ_02: Buildings > 1000m
    build_file = 'buildings.geojson'
    m_build = None
    if os.path.exists(build_file):
        print("   -> Buildings Exclusion (> 1000m)")
        m_build = mask_buildings_from_geojson(X_grid, Y_grid, build_file, radius_m=1000.0)
    else:
        print(f"   [!] Missing {build_file}, skipping buildings mask.")

    # Residential Areas (Exclusion)
    res_file = 'export.geojson'
    m_res = None
    if os.path.exists(res_file):
        print(f"   -> Residential Areas Exclusion ({res_file})")
        m_res = mask_residential_from_geojson(X_grid, Y_grid, res_file)
    else:
        print(f"   [!] Missing {res_file}, skipping residential mask.")

    # ---------------------------------------------------------
    # 5. ENVIRONMENTAL MASKS
    # ---------------------------------------------------------
    print("\n5. Computing Environmental Masks...")
    
    # Protected Areas (National Parks, etc.)
    prot_file = 'protected_areas.geojson'
    m_prot = None
    if os.path.exists(prot_file):
        print("   -> Protected Areas Exclusion")
        m_prot = mask_protected_areas_from_geojson(X_grid, Y_grid, prot_file)
    else:
        print(f"   [!] Missing {prot_file}, skipping protected areas.")

    # ---------------------------------------------------------
    # 6. COMBINATION (PRE-VISIBILITY)
    # ---------------------------------------------------------
    print("\n6. Combining Constraints (Pre-LOS)...")
    
    masks_list = [m_50km, m_france, m_seaside, m_slope]
    
    # Dictionary for visualization later
    masks_dict = {
        '50km Radius': m_50km,
        'French Territory': m_france,
        'Seaside Buffer': m_seaside,
        'Slope': m_slope
    }

    if m_elec is not None:
        masks_list.append(m_elec)
        masks_dict['Electricity'] = m_elec
    
    if m_roads is not None:
        masks_list.append(m_roads)
        masks_dict['Roads'] = m_roads
        
    if m_build is not None:
        masks_list.append(m_build)
        masks_dict['Buildings'] = m_build

    if m_res is not None:
        masks_list.append(m_res)
        masks_dict['Residential'] = m_res
        
    if m_prot is not None:
        masks_list.append(m_prot)
        masks_dict['Protected'] = m_prot

    # Logical AND combination
    pre_los_mask = combine_masks(*masks_list)
    pre_los_count = np.sum(pre_los_mask)
    print(f"   -> {pre_los_count:,} candidates before Line of Sight check.")

    # ---------------------------------------------------------
    # 7. LINE OF SIGHT CHECK (Final Physics Filter)
    # ---------------------------------------------------------
    print("\n7. Checking Line of Sight (Numba Accelerated + Earth Curvature)...")
    
    if pre_los_count == 0:
        print("   [!] No candidates left to check visibility.")
        final_mask = pre_los_mask
    else:
        # Get indices of remaining candidates
        cand_indices = np.where(pre_los_mask)
        
        # Run Ray Tracing on Z_corrected
        # The function now takes Metric Axes (X_m, Y_m) and Target Lat/Lon
        is_visible = check_visibility_batch(
            X_m, Y_m, Z_corrected,
            cand_indices, 
            REF_LAT, REF_LON,  # Nice Airport
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
    # 8. EXPORT (CRITICAL: BACK TO LAT/LON)
    # ---------------------------------------------------------
    print("\n8. Saving Results...")

    # Save to NPZ using LAT/LON for Google Earth compatibility
    output_npz = "authorized_points_all_masks.npz"
    
    # Get indices of final valid points
    i_rows, j_cols = np.where(final_mask)
    
    np.savez(
        output_npz,
        lat=lats[i_rows],      # Map Row index -> Latitude
        lon=lons[j_cols],      # Map Col index -> Longitude
        z=Z_raw[i_rows, j_cols], # Store REAL altitude (not corrected)
        mask=final_mask
    )
    print(f"   [OK] Data saved to {output_npz} (Lat/Lon format)")

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