
"""
Example Usage: Infrastructure Masks + COASTLINE BUFFER (REQ_03)

This script demonstrates how to use the electrical stations and roads masks
for radar site location study. It shows how to:
1. Load terrain data
2. Create electrical stations masks (REQ_06: 500m proximity)
3. Create roads proximity masks (REQ_05: 500m proximity)
4. Combine with other geographical masks (including REQ_03: Coastline > 100m)
5. Visualize results

This extends Lot 2 - Radar site location study with infrastructure constraints.
"""


import numpy as np
from site_location_masks import mask_land, mask_50km, mask_french_territory, combine_masks, mask_coastline_buffer
from electrical_stations_masks import mask_electrical_from_json
from roads_masks import mask_roads_from_geojson  
from visualize_site_location_masks import plot_masks_overlay
from export_site_location_masks_kml import export_masks_to_kmz
import os


def load_terrain_npz(npz_file: str):
    """Load terrain data from NPZ file."""
    terrain = np.load(npz_file)
    lats = terrain['lat']
    lons = terrain['lon']
    Z = terrain['ter']
    return lats, lons, Z


def main():
    """Example usage of electrical stations and roads masks."""
    
    print("="*70)

    print("Infrastructure Masks - Electrical Stations & Roads & Seaside Buffer")

    print("="*70)
    
    # ============================================================
    # 1. Load terrain data
    # ============================================================
    print("\n1. Loading terrain data...")
    terrain_file = 'terrain_req01_50km.npz' # ou 'terrain_mat.npz' selon votre projet
    
    try:
        lats, lons, Z = load_terrain_npz(terrain_file)
        print(f"   ✓ Terrain loaded: {len(lats)} x {len(lons)} grid points")
        print(f"   ✓ Latitude range: {lats.min():.6f} to {lats.max():.6f}")
        print(f"   ✓ Longitude range: {lons.min():.6f} to {lons.max():.6f}")
        print(f"   ✓ Elevation range: {Z.min():.1f} to {Z.max():.1f} meters")
    except FileNotFoundError:
        print(f"   ✗ Error: Terrain file '{terrain_file}' not found.")
        return
    except Exception as e:
        print(f"   ✗ Error loading terrain: {e}")
        return
    
    # ============================================================
    # 2. Define reference point (Nice airport)
    # ============================================================
    print("\n2. Defining reference point...")
    nice_lat = 43.6584  # Nice Airport latitude
    nice_lon = 7.2159   # Nice Airport longitude
    print(f"   ✓ Reference point: Nice Airport")
    print(f"   ✓ Coordinates: ({nice_lat:.4f}, {nice_lon:.4f})")
    
    # ============================================================
    # 3. Create basic geographical masks
    # ============================================================
    print("\n3. Creating basic geographical masks...")
    
    print("\n   a) Creating land mask (onshore constraint)...")
    mask_land_result = mask_land(lats, lons, Z)
    land_count = np.sum(mask_land_result)
    land_pct = land_count / mask_land_result.size * 100
    print(f"      ✓ Land mask: {land_count:,} admissible points ({land_pct:.1f}%)")
    
    print("\n   b) Creating 50km distance mask...")
    mask_50km_result = mask_50km(lats, lons, nice_lat, nice_lon, radius_km=50.0)
    within_50km = np.sum(mask_50km_result)
    within_pct = within_50km / mask_50km_result.size * 100
    print(f"      ✓ 50km mask: {within_50km:,} admissible points ({within_pct:.1f}%)")
    
    print("\n   c) Creating French territory mask...")
    mask_french_result = mask_french_territory(lats, lons)
    french_count = np.sum(mask_french_result)
    french_pct = french_count / mask_french_result.size * 100
    print(f"      ✓ French territory mask: {french_count:,} admissible points ({french_pct:.1f}%)")

    # ============================================================
    # 3b. REQ_03: Seaside Ban (Buffer 100m) - AJOUTÉ ICI
    # ============================================================
    print("\n   d) Creating Coastline Buffer mask (REQ_03: > 100m from sea)...")
    try:
        mask_seaside_result = mask_coastline_buffer(lats, lons, Z, buffer_m=100.0)
        seaside_count = np.sum(mask_seaside_result)
        seaside_pct = seaside_count / mask_seaside_result.size * 100
        print(f"      ✓ Seaside buffer mask: {seaside_count:,} admissible points ({seaside_pct:.1f}%)")
        print(f"      (Note: This is stricter than the basic Land mask)")
    except Exception as e:
        print(f"      ✗ Error creating coastline buffer: {e}")
        mask_seaside_result = None
    
    # ============================================================
    # 4. Create electrical stations mask (REQ_06)
    # ============================================================
    print("\n4. Creating electrical stations mask (REQ_06: 500m proximity)...")
    
    stations_file = 'page1.json'
    if not os.path.exists(stations_file):
        print(f"   ⚠ Warning: {stations_file} not found, skipping electrical mask")
        mask_electrical = None
    else:
        try:
            mask_electrical = mask_electrical_from_json(
                lats, lons, 
                json_file=stations_file, 
                radius_m=500.0
            )
            electrical_count = np.sum(mask_electrical)
            electrical_pct = electrical_count / mask_electrical.size * 100
            print(f"   ✓ Electrical stations mask: {electrical_count:,} admissible points ({electrical_pct:.1f}%)")
            print(f"   ✓ REQ_06: Electrical access within 500m")
        except Exception as e:
            print(f"   ✗ Error creating electrical mask: {e}")
            mask_electrical = None
    
    # ============================================================
    # 5. Create roads proximity mask
    # ============================================================

    print("\n5. Creating roads proximity mask (MAJOR ROADS ONLY)...")

    
    roads_file = 'roads_nice_50km.geojson'
    if not os.path.exists(roads_file):
        print(f"   ⚠ Warning: {roads_file} not found, skipping roads mask")
        mask_roads = None
    else:
        try:
            # MISE A JOUR REQ_05 : 500m au lieu de 2000m
            mask_roads = mask_roads_from_geojson(
                lats, lons, 
                geojson_file=roads_file, 

                max_distance_m=500.0,   # <-- CHANGÉ à 500m pour REQ_05
                major_roads_only=True   # Only motorways, trunk, primary, secondary

            )
            roads_count = np.sum(mask_roads)
            roads_pct = roads_count / mask_roads.size * 100
            print(f"   ✓ Roads proximity mask: {roads_count:,} admissible points ({roads_pct:.1f}%)")

            print(f"   ✓ REQ_05: Construction access within 500m of MAJOR roads")

        except Exception as e:
            print(f"   ✗ Error creating roads mask: {e}")
            mask_roads = None

    # ============================================================
    # 5b. Create buildings exclusion mask (OPTIONAL)
    # ============================================================
    print("\n5b. Creating buildings exclusion mask (OPTIONAL - >1000m from buildings)...")
    
    buildings_file = 'buildings.geojson'
    if not os.path.exists(buildings_file):
        print(f"   ⚠ Info: {buildings_file} not found, skipping buildings mask (optional)")
        mask_buildings = None
    else:
        try:
            # Import buildings mask module if it exists
            from buildings_masks import mask_buildings_from_geojson
            
            # True = admissible (farther than 1000m from any building)
            mask_buildings = mask_buildings_from_geojson(
                lats, lons,
                geojson_file=buildings_file,
                radius_m=1000.0
            )
            buildings_count = np.sum(mask_buildings)
            buildings_pct = buildings_count / mask_buildings.size * 100
            print(f"   ✓ Buildings exclusion mask: {buildings_count:,} admissible points ({buildings_pct:.1f}%)")

            print(f"   ✓ REQ_03: distance > 1000m from any dwelling")

        except Exception as e:
            print(f"   ⚠ Info: Could not create buildings mask: {e} (optional)")
            mask_buildings = None

    # ============================================================
    # 6. Combine all masks
    # ============================================================
    print("\n6. Combining all masks...")
    
    # Start with basic geographical masks
    # NOTE: On inclut mask_seaside_result qui est plus strict que mask_land
    masks_to_combine = [mask_land_result, mask_50km_result, mask_french_result]
    
    # Add optional masks
    if mask_seaside_result is not None:
        masks_to_combine.append(mask_seaside_result)
        print("   -> Adding Seaside Buffer constraint")

    if mask_electrical is not None:
        masks_to_combine.append(mask_electrical)
    if mask_roads is not None:
        masks_to_combine.append(mask_roads)
    if mask_buildings is not None:
        masks_to_combine.append(mask_buildings)
    
    mask_combined = combine_masks(*masks_to_combine)
    combined_count = np.sum(mask_combined)
    combined_pct = combined_count / mask_combined.size * 100
    print(f"   ✓ Combined mask: {combined_count:,} admissible points ({combined_pct:.1f}%)")

    # Save authorized (admissible) points to NPZ
    print("\n   Saving authorized points...")
    admissible_i, admissible_j = np.where(mask_combined)

    
    authorized_lat = lats[admissible_i]
    authorized_lon = lons[admissible_j]
    authorized_z = Z[admissible_i, admissible_j]

    np.savez(
        "authorized_points_all_masks.npz",
        lat=authorized_lat,
        lon=authorized_lon,
        z=authorized_z,
        mask=mask_combined
    )
    print(f"   ✓ Saved authorized points: authorized_points_all_masks.npz ({len(authorized_lat):,} points)")
    
    # ============================================================
    # 7. Display statistics
    # ============================================================
    print("\n7. Mask Statistics:")
    print("   " + "-"*60)
    print(f"   {'Mask':<35} {'Admissible Points':<20} {'Percentage':<15}")
    print("   " + "-"*60)
    print(f"   {'Land mask':<35} {land_count:>15,} {land_pct:>14.1f}%")
    print(f"   {'50km distance mask':<35} {within_50km:>15,} {within_pct:>14.1f}%")
    print(f"   {'French territory mask':<35} {french_count:>15,} {french_pct:>14.1f}%")
    
    if mask_seaside_result is not None:
        print(f"   {'Seaside Buffer (>100m)':<35} {seaside_count:>15,} {seaside_pct:>14.1f}%")

    if mask_electrical is not None:
        print(f"   {'Electrical 500m (REQ_06)':<35} {electrical_count:>15,} {electrical_pct:>14.1f}%")
    
    if mask_roads is not None:

        print(f"   {'Roads 500m (major roads)':<35} {roads_count:>15,} {roads_pct:>14.1f}%")

    if mask_buildings is not None:
        print(f"   {'Buildings 1000m exclusion':<35} {buildings_count:>15,} {buildings_pct:>14.1f}%")


    
    print(f"   {'Combined mask (ALL)':<35} {combined_count:>15,} {combined_pct:>14.1f}%")
    print("   " + "-"*60)
    
    # ============================================================
    # 8. Visualize masks
    # ============================================================
    print("\n8. Visualizing masks...")
    
    # Create dictionary of masks for visualization
    masks_dict = {
        'Land Mask (Onshore)': mask_land_result,
        '50km Distance Mask': mask_50km_result,
        'French Territory Mask': mask_french_result,
    }
    
    if mask_seaside_result is not None:
        masks_dict['Seaside Buffer >100m'] = mask_seaside_result

    if mask_electrical is not None:
        masks_dict['Electrical 500m (REQ_06)'] = mask_electrical
    
    if mask_roads is not None:
        masks_dict['Roads 500m (Major Roads)'] = mask_roads

    if mask_buildings is not None:
        masks_dict['Buildings >1000m'] = mask_buildings
    
    masks_dict['Combined Mask (ALL)'] = mask_combined
    
    try:
        # Use subset for faster visualization if needed
        if len(lats) * len(lons) > 500000:
            print("   (Using subset for faster visualization)")
            step = max(1, len(lats) // 300)
            lats_viz = lats[::step]
            lons_viz = lons[::step]
            Z_viz = Z[::step, ::step]
            masks_viz = {name: mask[::step, ::step] for name, mask in masks_dict.items()}
        else:
            lats_viz = lats
            lons_viz = lons
            Z_viz = Z
            masks_viz = masks_dict
        
        plot_masks_overlay(lats_viz, lons_viz, Z_viz, masks_viz, nice_lat, nice_lon,
                          save_path='infrastructure_masks_overlay.png')
        print("   ✓ PNG visualization saved: infrastructure_masks_overlay.png")
    except Exception as e:
        print(f"   ✗ Error in visualization: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================================
    # 9. Export to Google Earth (KMZ)
    # ============================================================
    print("\n9. Exporting masks to Google Earth (KMZ)...")
    
    try:
        # Use subset for faster export if needed
        if len(lats) * len(lons) > 500000:
            print("   (Using subset for faster export)")
            step = max(1, len(lats) // 200)
            lats_export = lats[::step]
            lons_export = lons[::step]
            masks_export = {name: mask[::step, ::step] for name, mask in masks_dict.items()}
        else:
            lats_export = lats
            lons_export = lons
            masks_export = masks_dict
        
        export_masks_to_kmz(masks_export, lats_export, lons_export,
                            'infrastructure_masks.kmz',
                            nice_lat=nice_lat, nice_lon=nice_lon)
        print("   ✓ KMZ file exported: infrastructure_masks.kmz (open in Google Earth)")
    except Exception as e:
        print(f"   ✗ Error in KMZ export: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================================
    # 10. Finding optimal candidate sites (Example)
    # ============================================================
    print("\n10. Finding optimal candidate sites...")
    
    admissible_indices = np.where(mask_combined)
    n_candidates = len(admissible_indices[0])
    
    print(f"   ✓ Found {n_candidates:,} admissible grid points")
    
    if n_candidates > 0:
        print("\n   Example optimal candidate locations (first 5):")
        print("   " + "-"*60)
        print(f"   {'Index':<10} {'Latitude':<15} {'Longitude':<15} {'Elevation (m)':<15}")
        print("   " + "-"*60)
        
        for idx in range(min(5, n_candidates)):
            i = admissible_indices[0][idx]
            j = admissible_indices[1][idx]
            lat = lats[i]
            lon = lons[j]
            elev = Z[i, j]
            print(f"   {idx+1:<10} {lat:>14.6f} {lon:>14.6f} {elev:>14.1f}")
        
        if n_candidates > 5:
            print(f"   ... and {n_candidates - 5:,} more candidates")
        print("   " + "-"*60)
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"✓ Terrain grid: {len(lats)} x {len(lons)} = {len(lats)*len(lons):,} points")

    print(f"✓ Combined mask (ALL): {combined_count:,} admissible points ({combined_pct:.1f}%)")
    
    print("\n✓ Constraints implemented:")
    print("  - REQ_01: 50km radius")
    print("  - REQ_03: Seaside buffer > 100m & Buildings > 1000m")
    print("  - REQ_05: Roads < 500m (updated from 2000m)")
    print("  - REQ_06: Electrical < 500m")
    print("  - REQ_07: French Territory")
    
    print("\nOutput files:")
    print("  - infrastructure_masks_overlay.png")
    print("  - infrastructure_masks.kmz")
    print("  - authorized_points_all_masks.npz")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()