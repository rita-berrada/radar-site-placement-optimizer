"""
Test script for KML/KMZ export functionality.

This script tests the export_kml.py module by:
1. Loading terrain data
2. Computing coverage maps
3. Testing all export functions (single KML, all KMZ)
4. Verifying files are created correctly
"""

import numpy as np
import os
import zipfile
from pathlib import Path

from visualize_terrain import load_terrain_npz
from coverage_analysis import compute_coverage_map, compute_all_coverage_maps
from export_kml import export_coverage_to_kml, export_all_coverage_to_kmz, create_visibility_map_kml


def test_export_kml():
    """Test KML/KMZ export functionality."""
    
    print("="*60)
    print("Testing KML/KMZ Export Functionality")
    print("="*60)
    
    # Load terrain data
    print("\n1. Loading terrain data...")
    try:
        lats, lons, Z = load_terrain_npz('terrain_mat.npz')
        print(f"   ✓ Terrain loaded: {len(lats)} x {len(lons)} grid")
    except FileNotFoundError:
        print("   ✗ Error: terrain_mat.npz not found")
        return False
    except Exception as e:
        print(f"   ✗ Error loading terrain: {e}")
        return False
    
    # Create small subset for fast testing
    print("\n2. Creating small grid subset (50x50)...")
    lat_step = max(1, len(lats) // 50)
    lon_step = max(1, len(lons) // 50)
    
    lats_small = lats[::lat_step][:50]
    lons_small = lons[::lon_step][:50]
    Z_small = Z[::lat_step, ::lon_step][:50, :50]
    
    # Ensure Z_small has correct shape
    if Z_small.shape != (len(lats_small), len(lons_small)):
        min_lat = min(len(lats_small), Z_small.shape[0])
        min_lon = min(len(lons_small), Z_small.shape[1])
        lats_small = lats_small[:min_lat]
        lons_small = lons_small[:min_lon]
        Z_small = Z_small[:min_lat, :min_lon]
    
    print(f"   ✓ Small grid: {len(lats_small)} x {len(lons_small)}")
    
    # Test radar position (center of small grid)
    radar_lat = (lats_small.min() + lats_small.max()) / 2
    radar_lon = (lons_small.min() + lons_small.max()) / 2
    radar_height_agl_m = 50.0
    
    print(f"\n3. Test radar position:")
    print(f"   Latitude: {radar_lat:.6f}")
    print(f"   Longitude: {radar_lon:.6f}")
    print(f"   Height AGL: {radar_height_agl_m} m")
    
    # Compute a single coverage map for testing
    print("\n4. Computing coverage map (FL100)...")
    try:
        coverage_map = compute_coverage_map(
            radar_lat, radar_lon, radar_height_agl_m,
            100,  # FL100
            lats_small, lons_small, Z_small,
            n_samples=30  # Reduced samples for faster testing
        )
        
        visible_count = np.sum(coverage_map)
        coverage_pct = visible_count / coverage_map.size * 100
        print(f"   ✓ Coverage map computed: shape {coverage_map.shape}")
        print(f"   ✓ Coverage: {visible_count}/{coverage_map.size} visible ({coverage_pct:.1f}%)")
        
    except Exception as e:
        print(f"   ✗ Coverage map computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 1: create_visibility_map_kml() function
    print("\n5. Testing create_visibility_map_kml()...")
    try:
        kml_element = create_visibility_map_kml(
            coverage_map, lats_small, lons_small, 100,
            radar_lat, radar_lon
        )
        
        # Verify it returns an Element
        from xml.etree.ElementTree import Element
        if isinstance(kml_element, Element):
            print("   ✓ create_visibility_map_kml() returned valid XML Element")
            print(f"   ✓ Element tag: {kml_element.tag}")
        else:
            print("   ✗ create_visibility_map_kml() did not return XML Element")
            return False
            
    except Exception as e:
        print(f"   ✗ create_visibility_map_kml() test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: export_coverage_to_kml() - single KML file
    print("\n6. Testing export_coverage_to_kml() (single KML)...")
    test_kml_file = "test_coverage_fl100.kml"
    try:
        # Remove file if it exists
        if os.path.exists(test_kml_file):
            os.remove(test_kml_file)
        
        export_coverage_to_kml(
            coverage_map, lats_small, lons_small, 100,
            test_kml_file,
            radar_lat, radar_lon
        )
        
        # Verify file was created
        if os.path.exists(test_kml_file):
            file_size = os.path.getsize(test_kml_file)
            print(f"   ✓ KML file created: {test_kml_file}")
            print(f"   ✓ File size: {file_size:,} bytes")
            
            # Verify it's valid XML (basic check)
            try:
                from xml.etree.ElementTree import parse
                tree = parse(test_kml_file)
                root = tree.getroot()
                print(f"   ✓ Valid XML structure (root: {root.tag})")
            except Exception as e:
                print(f"   ⚠ Warning: XML validation failed: {e}")
        else:
            print(f"   ✗ KML file was not created: {test_kml_file}")
            return False
            
    except Exception as e:
        print(f"   ✗ export_coverage_to_kml() test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: export_all_coverage_to_kmz() - all flight levels to KMZ
    print("\n7. Testing export_all_coverage_to_kmz() (all 8 FLs to KMZ)...")
    test_flight_levels = [5, 10, 20, 50, 100, 200, 300, 400]  # All 8 flight levels
    test_kmz_file = "test_radar_coverage.kmz"
    
    try:
        # Compute coverage maps for all flight levels
        print(f"   Computing coverage maps for all {len(test_flight_levels)} flight levels...")
        coverage_maps = compute_all_coverage_maps(
            radar_lat, radar_lon, radar_height_agl_m,
            test_flight_levels,
            lats_small, lons_small, Z_small,
            n_samples=30  # Reduced samples for faster testing
        )
        
        print(f"   ✓ Computed {len(coverage_maps)} coverage maps")
        
        # Display coverage statistics for all flight levels (showing variations)
        print("\n   Coverage statistics by flight level:")
        print("   " + "-" * 50)
        for fl in sorted(coverage_maps.keys()):
            coverage_map = coverage_maps[fl]
            visible_count = np.sum(coverage_map)
            blocked_count = np.sum(~coverage_map)
            coverage_pct = visible_count / coverage_map.size * 100
            blocked_pct = blocked_count / coverage_map.size * 100
            print(f"   FL{fl:3.0f}: {coverage_pct:6.2f}% visible, {blocked_pct:6.2f}% blocked")
        print("   " + "-" * 50)
        
        # Remove file if it exists
        if os.path.exists(test_kmz_file):
            os.remove(test_kmz_file)
        
        # Export to KMZ
        export_all_coverage_to_kmz(
            coverage_maps, lats_small, lons_small,
            radar_lat, radar_lon,
            output_path=test_kmz_file
        )
        
        # Verify file was created
        if os.path.exists(test_kmz_file):
            file_size = os.path.getsize(test_kmz_file)
            print(f"   ✓ KMZ file created: {test_kmz_file}")
            print(f"   ✓ File size: {file_size:,} bytes")
            
            # Verify it's a valid ZIP file (KMZ is a ZIP)
            try:
                with zipfile.ZipFile(test_kmz_file, 'r') as kmz:
                    file_list = kmz.namelist()
                    print(f"   ✓ Valid KMZ/ZIP archive")
                    print(f"   ✓ Contains {len(file_list)} file(s): {file_list}")
                    
                    # Check if doc.kml is inside
                    if 'doc.kml' in file_list:
                        print("   ✓ Contains doc.kml file")
                        
                        # Try to read the KML from ZIP
                        kml_content = kmz.read('doc.kml')
                        print(f"   ✓ KML content size: {len(kml_content):,} bytes")
                    else:
                        print("   ⚠ Warning: doc.kml not found in KMZ file")
                        
            except zipfile.BadZipFile:
                print(f"   ✗ Invalid ZIP/KMZ file")
                return False
            except Exception as e:
                print(f"   ⚠ Warning: ZIP validation failed: {e}")
        else:
            print(f"   ✗ KMZ file was not created: {test_kmz_file}")
            return False
            
    except Exception as e:
        print(f"   ✗ export_all_coverage_to_kmz() test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "="*60)
    print("All export tests passed! ✓")
    print("="*60)
    print(f"\nTest files created:")
    print(f"  - {test_kml_file} (single flight level KML - FL100)")
    print(f"  - {test_kmz_file} (all {len(test_flight_levels)} flight levels KMZ)")
    print(f"\nFlight levels tested: {', '.join([f'FL{fl}' for fl in sorted(test_flight_levels)])}")
    print(f"\nYou can open these files in Google Earth to verify!")
    print(f"   - Green areas = Visible from radar")
    print(f"   - Red areas = Blocked by terrain")
    
    return True


if __name__ == "__main__":
    success = test_export_kml()
    exit(0 if success else 1)
