"""
Example Usage: Site Location Masks

This script demonstrates how to use the geographical masks for radar site location study.
It shows how to:
1. Load terrain data
2. Create individual masks (land/inland, distance, territory)
3. Combine masks
4. Visualize results (PNG)
5. Export results (KMZ for Google Earth)
6. Extract authorized candidate points, round them to 0.1°, and save to NPZ

This is for Lot 2 - Radar site location study, where masks define admissible
search areas based on static geographical constraints.
"""

import numpy as np
from scipy.ndimage import binary_dilation

from visualize_terrain import load_terrain_npz
from site_location_masks import mask_land, mask_50km, mask_french_territory, combine_masks
from visualize_site_location_masks import plot_masks_overlay
from export_site_location_masks_kml import export_masks_to_kmz


def main():
    """Example usage of site location masks."""

    print("=" * 70)
    print("Site Location Masks - Example Usage")
    print("=" * 70)

    # ============================================================
    # 1. Load terrain data
    # ============================================================
    print("\n1. Loading terrain data...")
    terrain_file = "terrain_mat.npz"

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
    # 3. Create individual masks
    # ============================================================
    print("\n3. Creating geographical masks...")

    # 3.a) Onshore constraint: Radar must be on land (Z > 0)
    print("\n   a) Creating land mask (onshore constraint)...")
    mask_land_result = mask_land(lats, lons, Z)
    land_count = int(np.sum(mask_land_result))
    land_pct = land_count / mask_land_result.size * 100.0
    print(f"      ✓ Land mask created: {land_count:,} admissible points ({land_pct:.1f}%)")
    print(f"      ✓ Shape: {mask_land_result.shape}")

    # 3.b) Distance constraint: Within 50km of Nice airport
    print("\n   b) Creating 50km distance mask...")
    radius_km = 50.0
    mask_50km_result = mask_50km(lats, lons, nice_lat, nice_lon, radius_km=radius_km)
    within_50km = int(np.sum(mask_50km_result))
    within_pct = within_50km / mask_50km_result.size * 100.0
    print(f"      ✓ 50km mask created: {within_50km:,} admissible points ({within_pct:.1f}%)")
    print(f"      ✓ Shape: {mask_50km_result.shape}")

    # 3.c) French territory constraint (heuristic): excludes Monaco & lon > 7.5 (Italy)
    print("\n   c) Creating French territory mask...")
    mask_french_result = mask_french_territory(lats, lons)
    french_count = int(np.sum(mask_french_result))
    french_pct = french_count / mask_french_result.size * 100.0
    print(f"      ✓ French territory mask created: {french_count:,} admissible points ({french_pct:.1f}%)")
    print(f"      ✓ Shape: {mask_french_result.shape}")
    print(f"      ✓ Excludes Monaco and Italy")

    # 3.d) Inland correction: exclude near-coast areas (convert land -> inland)
    print("\n   d) Creating inland mask (exclude near-coast areas)...")
    # Dilate the sea into the land to define a coastal exclusion belt.
    # iterations controls the width of the excluded coastal band.
    coastal_band = binary_dilation(~mask_land_result, iterations=3)

    # Keep only inland points (land AND not close to the coastline)
    mask_inland = mask_land_result & (~coastal_band)

    inland_count = int(np.sum(mask_inland))
    inland_pct = inland_count / mask_inland.size * 100.0
    print(f"      ✓ Inland mask created: {inland_count:,} admissible points ({inland_pct:.1f}%)")
    print(f"      ✓ Shape: {mask_inland.shape}")

    # ============================================================
    # 4. Combine masks
    # ============================================================
    print("\n4. Combining masks...")
    mask_combined = combine_masks(mask_inland, mask_50km_result, mask_french_result)
    combined_count = int(np.sum(mask_combined))
    combined_pct = combined_count / mask_combined.size * 100.0
    print(f"   ✓ Combined mask: {combined_count:,} admissible points ({combined_pct:.1f}%)")
    print(f"   ✓ Shape: {mask_combined.shape}")

    # Verify combination logic
    assert combined_count <= land_count, "Combined should have fewer or equal points than land mask"
    assert combined_count <= inland_count, "Combined should have fewer or equal points than inland mask"
    assert combined_count <= within_50km, "Combined should have fewer or equal points than 50km mask"
    assert combined_count <= french_count, "Combined should have fewer or equal points than French territory mask"
    print("   ✓ Combination logic verified (subset of all individual masks)")

    # ============================================================
    # 5. Display statistics
    # ============================================================
    print("\n5. Mask Statistics:")
    print("   " + "-" * 60)
    print(f"   {'Mask':<25} {'Admissible Points':<20} {'Percentage':<15}")
    print("   " + "-" * 60)
    print(f"   {'Land mask':<25} {land_count:>15,} {land_pct:>14.1f}%")
    print(f"   {'Inland mask (no coast)':<25} {inland_count:>15,} {inland_pct:>14.1f}%")
    print(f"   {'50km distance mask':<25} {within_50km:>15,} {within_pct:>14.1f}%")
    print(f"   {'French territory mask':<25} {french_count:>15,} {french_pct:>14.1f}%")
    print(f"   {'Combined mask':<25} {combined_count:>15,} {combined_pct:>14.1f}%")
    print("   " + "-" * 60)

    # ============================================================
    # 6. Visualize masks (PNG overlay)
    # ============================================================
    print("\n6. Visualizing masks (PNG overlay on terrain)...")

    masks_dict = {
        "Land Mask (Onshore)": mask_land_result,
        "Inland Mask (No Coast)": mask_inland,
        "50km Distance Mask": mask_50km_result,
        "French Territory Mask": mask_french_result,
        "Combined Mask": mask_combined,
    }

    try:
        # For faster visualization, use a subset if grid is very large
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

        plot_masks_overlay(
            lats_viz,
            lons_viz,
            Z_viz,
            masks_viz,
            nice_lat,
            nice_lon,
            save_path="site_location_masks_overlay.png",
        )
        print("   ✓ PNG visualization saved (admissible=transparent, excluded=grey overlay)")
    except Exception as e:
        print(f"   ✗ Error in PNG visualization: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================
    # 7. Export to Google Earth (KMZ)
    # ============================================================
    print("\n7. Exporting masks to Google Earth (KMZ)...")

    try:
        # For faster export, use a subset if grid is very large
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

        export_masks_to_kmz(
            masks_export,
            lats_export,
            lons_export,
            "site_location_masks.kmz",
            nice_lat=nice_lat,
            nice_lon=nice_lon,
        )
        print("   ✓ KMZ file exported (open in Google Earth)")
    except Exception as e:
        print(f"   ✗ Error in KMZ export: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================
    # 8. Candidate site selection + rounding to 0.1° + save NPZ
    # ============================================================
    print("\n8. Example: Finding candidate sites...")

    admissible_indices = np.where(mask_combined)
    n_candidates = len(admissible_indices[0])
    print(f"   ✓ Found {n_candidates:,} admissible grid points")

    

    # --- Build authorized points array (lat, lon) ---
    authorized_points = np.column_stack((
        lats[admissible_indices[0]],
        lons[admissible_indices[1]]
    ))  # shape (N, 2)

    # --- Round to 0.1° and remove duplicates ---
    authorized_points_01 = np.round(authorized_points, 1)
    authorized_points_01 = np.unique(authorized_points_01, axis=0)

    # Re-filter after rounding to ensure points are still on land
    filtered_points = []
    for lat, lon in authorized_points_01:
        i = np.argmin(np.abs(lats - lat))
        j = np.argmin(np.abs(lons - lon))
        if mask_inland[i, j]:
            filtered_points.append([lat, lon])

    authorized_points_01 = np.array(filtered_points, dtype=float).reshape(-1, 2)
    authorized_points_01 = np.unique(authorized_points_01, axis=0)

    print(f"   ✓ Authorized candidate points after 0.1° rounding: {len(authorized_points_01):,}")

    np.savez("authorized_points_01deg.npz", points=authorized_points_01)
    print("   ✓ Saved authorized points to authorized_points_01deg.npz")


    # Optional sanity print: show first 5 rounded points
    if len(authorized_points_01) > 0:
        print("\n   Example candidate locations after 0.1° rounding (first 5):")
        print("   " + "-" * 45)
        print(f"   {'Index':<10} {'Latitude':<15} {'Longitude':<15}")
        print("   " + "-" * 45)
        for idx in range(min(5, len(authorized_points_01))):
            lat, lon = authorized_points_01[idx]
            print(f"   {idx+1:<10} {lat:>14.1f} {lon:>14.1f}")
        if len(authorized_points_01) > 5:
            print(f"   ... and {len(authorized_points_01) - 5:,} more rounded candidates")
        print("   " + "-" * 45)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"✓ Terrain grid: {len(lats)} x {len(lons)} = {len(lats)*len(lons):,} points")
    print(f"✓ Land mask: {land_count:,} admissible points ({land_pct:.1f}%)")
    print(f"✓ Inland mask: {inland_count:,} admissible points ({inland_pct:.1f}%)")
    print(f"✓ 50km mask: {within_50km:,} admissible points ({within_pct:.1f}%)")
    print(f"✓ French territory mask: {french_count:,} admissible points ({french_pct:.1f}%)")
    print(f"✓ Combined mask: {combined_count:,} admissible points ({combined_pct:.1f}%)")
    print(f"✓ Rounded candidates saved: {len(authorized_points_01):,} points -> authorized_points_01deg.npz")
    print("=" * 70)


if __name__ == "__main__":
    main()
