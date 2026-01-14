"""
Main Coverage Analysis Script

This script orchestrates the complete coverage analysis workflow:
1. Load terrain data (DTED 1 format)
2. Compute coverage maps for all flight levels
3. Visualize results
4. Export to KML/KMZ for Google Earth
"""

import numpy as np
from visualize_terrain import load_terrain_npz
from coverage_analysis import compute_all_coverage_maps
from visualize_coverage import plot_all_coverage_maps
from export_kml import export_all_coverage_to_kmz

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: tqdm not available. Install with 'pip install tqdm' for progress bars.")


def main():
    """Main execution function."""
    
    # ============================================================
    # Configuration
    # ============================================================
    
    # Terrain data file (DTED 1 format)
    terrain_file = 'terrain_mat.npz'
    
    # Radar position
    radar_lat = 43.6584   # Example: Nice Airport latitude
    radar_lon = 7.2159    # Example: Nice Airport longitude
    radar_height_agl_m = 50.0  # Radar height above ground level (meters)
    
    # Flight levels (tender requirement)
    flight_levels = [5, 10, 20, 50, 100, 200, 300, 400]
    
    # LOS parameters
    n_samples = 400  # Number of samples along LOS path
    margin_m = 0.0   # Safety margin (meters)
    
    # TEST MODE: Use smaller grid for faster testing
    # Set to True to use every Nth point (much faster but lower resolution)
    TEST_MODE = True
    TEST_GRID_STEP = 10  # Use every 10th point in test mode
    
    # Output file
    kmz_output = 'radar_coverage.kmz'
    
    # ============================================================
    # Load terrain data
    # ============================================================
    
    print("Loading terrain data...")
    try:
        lats_full, lons_full, Z_full = load_terrain_npz(terrain_file)
        print(f"Terrain loaded: {len(lats_full)} x {len(lons_full)} grid points")
        print(f"Latitude range: {lats_full.min():.6f} to {lats_full.max():.6f}")
        print(f"Longitude range: {lons_full.min():.6f} to {lons_full.max():.6f}")
        
        # Apply test mode if enabled
        if TEST_MODE:
            print(f"\n⚠️  TEST MODE ENABLED: Using every {TEST_GRID_STEP}th point")
            lats = lats_full[::TEST_GRID_STEP]
            lons = lons_full[::TEST_GRID_STEP]
            Z = Z_full[::TEST_GRID_STEP, ::TEST_GRID_STEP]
            print(f"Reduced grid: {len(lats)} x {len(lons)} = {len(lats)*len(lons):,} points")
        else:
            lats, lons, Z = lats_full, lons_full, Z_full
            
    except FileNotFoundError:
        print(f"Error: Terrain file '{terrain_file}' not found.")
        return
    except Exception as e:
        print(f"Error loading terrain: {e}")
        return
    
    # ============================================================
    # Compute coverage maps
    # ============================================================
    
    print("\nComputing coverage maps...")
    grid_size = len(lats) * len(lons)
    total_calculations = grid_size * len(flight_levels)
    print(f"Grid size: {len(lats)} x {len(lons)} = {grid_size:,} points")
    print(f"Total LOS calculations: {total_calculations:,} ({total_calculations/1e6:.1f} million)")
    
    # Rough time estimate (very approximate)
    # Assuming ~0.001-0.01 seconds per LOS calculation depending on complexity
    est_seconds = total_calculations * 0.005  # Conservative estimate
    est_minutes = est_seconds / 60
    est_hours = est_minutes / 60
    
    if est_hours >= 1:
        print(f"\n⚠️  WARNING: Estimated computation time: {est_hours:.1f} hours ({est_minutes:.0f} minutes)")
        print("   This will take a VERY long time for the full grid!")
        print("   Consider:")
        print("   - Setting TEST_MODE = True in the script for faster testing")
        print("   - Reducing n_samples (currently 400)")
        print("   - Processing fewer flight levels")
        response = input("\nContinue with full computation? (y/n): ").lower().strip()
        if response != 'y':
            print("Computation cancelled. Edit main_coverage.py to enable TEST_MODE for faster testing.")
            return
    elif est_minutes > 10:
        print(f"Estimated time: {est_minutes:.1f} minutes")
    else:
        print(f"Estimated time: {est_seconds:.0f} seconds")
    
    print("\nStarting computation...")
    print("(Progress: each flight level will print when complete)")
    
    # Track current flight level being processed
    current_fl = [None]  # Use list to allow modification in nested function
    
    # Progress callback for flight levels
    def fl_progress_callback(fl, current, total_fl):
        print(f"  ✓ FL{fl:3.0f} complete ({current}/{total_fl})")
        if current < total_fl:
            print(f"  → Starting FL{flight_levels[current]}...")
    
    # Compute all coverage maps
    try:
        coverage_maps = compute_all_coverage_maps(
            radar_lat, radar_lon, radar_height_agl_m,
            flight_levels, lats, lons, Z,
            n_samples=n_samples, margin_m=margin_m,
            progress_callback=fl_progress_callback
        )
        print("Coverage maps computed successfully!")
    except Exception as e:
        print(f"Error computing coverage maps: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print statistics
    print("\nCoverage Statistics:")
    print("-" * 50)
    for fl in sorted(coverage_maps.keys()):
        coverage_pct = np.sum(coverage_maps[fl]) / coverage_maps[fl].size * 100
        print(f"FL{fl:3.0f}: {coverage_pct:6.2f}% visible")
    print("-" * 50)
    
    # ============================================================
    # Visualize results
    # ============================================================
    
    print("\nGenerating coverage maps...")
    try:
        plot_all_coverage_maps(
            coverage_maps, lats, lons, radar_lat, radar_lon
        )
        print("Coverage maps displayed successfully!")
    except Exception as e:
        print(f"Error generating maps: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================================
    # Export to KML/KMZ
    # ============================================================
    
    export_kmz = input(f"\nExport to KMZ file '{kmz_output}'? (y/n): ").lower().strip() == 'y'
    if export_kmz:
        print("Exporting to KMZ...")
        try:
            export_all_coverage_to_kmz(
                coverage_maps, lats, lons, radar_lat, radar_lon,
                output_path=kmz_output
            )
            print(f"Export complete! Open '{kmz_output}' in Google Earth to view.")
        except Exception as e:
            print(f"Error exporting KMZ: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
