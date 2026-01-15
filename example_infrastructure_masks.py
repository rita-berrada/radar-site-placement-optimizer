"""
Example Usage: Electrical Stations and Roads Masks

This script demonstrates how to use the electrical stations and roads masks
for radar site location study. It shows how to:
1. Load terrain data
2. Create electrical stations masks (REQ_06: 500m proximity)
3. Create roads proximity masks (construction access)
4. Combine with other geographical masks
5. Visualize results

This extends Lot 2 - Radar site location study with infrastructure constraints.
"""

import numpy as np
from site_location_masks import mask_land, mask_50km, mask_french_territory, combine_masks
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
    print("Infrastructure Masks - Electrical Stations & Roads")
    print("="*70)
    
    # ============================================================
    # 1. Load terrain data
    # ============================================================
    print("\n1. Loading terrain data...")
    terrain_file = 'terrain_req01_50km.npz'
    
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
    print("\n5. Creating roads proximity mask (MAJOR ROADS ONLY - ULTRA FAST)...")
    
    roads_file = 'roads_nice_50km.geojson'
    if not os.path.exists(roads_file):
        print(f"   ⚠ Warning: {roads_file} not found, skipping roads mask")
        mask_roads = None
    else:
        try:
            mask_roads = mask_roads_from_geojson(
                lats, lons, 
                geojson_file=roads_file, 
                max_distance_m=2000.0,  # 2 km from major road
                major_roads_only=True   # Only motorways, trunk, primary, secondary
            )
            roads_count = np.sum(mask_roads)
            roads_pct = roads_count / mask_roads.size * 100
            print(f"   ✓ Roads proximity mask: {roads_count:,} admissible points ({roads_pct:.1f}%)")
            print(f"   ✓ Construction access within 2000m of MAJOR roads")
        except Exception as e:
            print(f"   ✗ Error creating roads mask: {e}")
            mask_roads = None
    
    # ============================================================
    # 6. Combine all masks
    # ============================================================
    print("\n6. Combining all masks...")
    
    # Start with basic geographical masks
    masks_to_combine = [mask_land_result, mask_50km_result, mask_french_result]
    
    # Add infrastructure masks if available
    if mask_electrical is not None:
        masks_to_combine.append(mask_electrical)
    if mask_roads is not None:
        masks_to_combine.append(mask_roads)
    
    mask_combined = combine_masks(*masks_to_combine)
    combined_count = np.sum(mask_combined)
    combined_pct = combined_count / mask_combined.size * 100
    print(f"   ✓ Combined mask: {combined_count:,} admissible points ({combined_pct:.1f}%)")
    
    # ============================================================
    # 7. Display statistics
    # ============================================================
    print("\n7. Mask Statistics:")
    print("   " + "-"*60)
    print(f"   {'Mask':<30} {'Admissible Points':<20} {'Percentage':<15}")
    print("   " + "-"*60)
    print(f"   {'Land mask':<30} {land_count:>15,} {land_pct:>14.1f}%")
    print(f"   {'50km distance mask':<30} {within_50km:>15,} {within_pct:>14.1f}%")
    print(f"   {'French territory mask':<30} {french_count:>15,} {french_pct:>14.1f}%")
    
    if mask_electrical is not None:
        print(f"   {'Electrical 500m (REQ_06)':<30} {electrical_count:>15,} {electrical_pct:>14.1f}%")
    
    if mask_roads is not None:
        print(f"   {'Roads 2000m (major roads)':<30} {roads_count:>15,} {roads_pct:>14.1f}%")
    
    print(f"   {'Combined mask (ALL)':<30} {combined_count:>15,} {combined_pct:>14.1f}%")
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
    
    if mask_electrical is not None:
        masks_dict['Electrical 500m (REQ_06)'] = mask_electrical
    
    if mask_roads is not None:
        masks_dict['Roads 2000m (Major Roads)'] = mask_roads
    
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
    # 10. Example: Finding optimal candidate sites
    # ============================================================
    print("\n10. Finding optimal candidate sites...")
    
    # Find admissible grid points
    admissible_indices = np.where(mask_combined)
    n_candidates = len(admissible_indices[0])
    
    print(f"   ✓ Found {n_candidates:,} admissible grid points")
    
    if n_candidates > 0:
        # Show example candidate locations
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
    print(f"✓ Land mask: {land_count:,} admissible points ({land_pct:.1f}%)")
    print(f"✓ 50km mask: {within_50km:,} admissible points ({within_pct:.1f}%)")
    print(f"✓ French territory mask: {french_count:,} admissible points ({french_pct:.1f}%)")
    
    if mask_electrical is not None:
        print(f"✓ Electrical 500m (REQ_06): {electrical_count:,} admissible points ({electrical_pct:.1f}%)")
    
    if mask_roads is not None:
        print(f"✓ Roads 2000m (major roads): {roads_count:,} admissible points ({roads_pct:.1f}%)")
    
    print(f"✓ Combined mask (ALL): {combined_count:,} admissible points ({combined_pct:.1f}%)")
    
    print("\n✓ Infrastructure constraints implemented:")
    print("  - REQ_06: Electrical access < 500m from Enedis stations")
    print("  - Construction access < 2000m from MAJOR road network (motorway/trunk/primary/secondary)")
    print("  - Combined with geographical constraints (land, distance, territory)")
    
    print("\nNext steps:")
    print("  - Select candidate radar sites from admissible area")
    print("  - Run coverage analysis from each candidate")
    print("  - Evaluate cost-benefit for each site")
    
    print("\nOutput files:")
    print("  - infrastructure_masks_overlay.png: PNG visualization")
    print("  - infrastructure_masks.kmz: Google Earth visualization (3D interactive)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()