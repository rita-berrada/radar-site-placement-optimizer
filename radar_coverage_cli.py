#!/usr/bin/env python3
"""
Radar Coverage Analysis - Command Line Interface

Standalone CLI for radar coverage analysis that can be used for batch processing
or when a web interface is not needed.

Usage:
    python radar_coverage_cli.py terrain.npz --output coverage_results
    python radar_coverage_cli.py terrain.npz --fl 10,50,100,200 --export-kmz
    python radar_coverage_cli.py terrain.npz --visualize --show-3d

Author: Radar Analysis Team
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Import functions from main application
from radar_coverage_app import (
    load_terrain_npz,
    convert_to_enu,
    normalize_xy_grid,
    compute_all_coverages,
    plot_terrain_2d,
    plot_terrain_3d,
    plot_coverage_map,
    plot_all_coverages_grid,
    export_all_coverages_kmz,
    fl_to_m,
    DEFAULT_REF_LAT,
    DEFAULT_REF_LON,
    HAS_NUMBA
)


def print_banner():
    """Print application banner."""
    print("=" * 70)
    print("   RADAR COVERAGE ANALYSIS - Command Line Interface")
    print("   Terrain Visualization, Coverage Analysis, KMZ Export")
    print("=" * 70)
    if HAS_NUMBA:
        print("   [✓] Numba acceleration enabled")
    else:
        print("   [!] Numba not available - using slower Python implementation")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Radar Coverage Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic coverage analysis with default settings
  python radar_coverage_cli.py terrain_mat.npz

  # Specify flight levels and export to KMZ
  python radar_coverage_cli.py terrain.npz --fl 10,50,100,200 --export-kmz

  # Visualize terrain only
  python radar_coverage_cli.py terrain.npz --visualize --no-coverage

  # Custom radar position
  python radar_coverage_cli.py terrain.npz --radar-lat 43.70 --radar-lon 7.25

  # Save all figures to output directory
  python radar_coverage_cli.py terrain.npz --output ./results --save-figures
        """
    )
    
    # Required arguments
    parser.add_argument(
        "terrain_file",
        type=str,
        help="Path to terrain NPZ file containing 'lat', 'lon', 'ter' arrays"
    )
    
    # Radar configuration
    parser.add_argument(
        "--radar-lat", type=float, default=DEFAULT_REF_LAT,
        help=f"Radar latitude (default: {DEFAULT_REF_LAT})"
    )
    parser.add_argument(
        "--radar-lon", type=float, default=DEFAULT_REF_LON,
        help=f"Radar longitude (default: {DEFAULT_REF_LON})"
    )
    parser.add_argument(
        "--radar-height", type=float, default=20.0,
        help="Radar height above ground level in meters (default: 20.0)"
    )
    
    # Reference point
    parser.add_argument(
        "--ref-lat", type=float, default=DEFAULT_REF_LAT,
        help=f"Reference latitude for ENU conversion (default: {DEFAULT_REF_LAT})"
    )
    parser.add_argument(
        "--ref-lon", type=float, default=DEFAULT_REF_LON,
        help=f"Reference longitude for ENU conversion (default: {DEFAULT_REF_LON})"
    )
    
    # Flight levels (8 standard levels: 5, 10, 20, 50, 100, 200, 300, 400)
    parser.add_argument(
        "--fl", type=str, default="10,50,100,200",
        help="Comma-separated list of flight levels from [5,10,20,50,100,200,300,400] (default: 10,50,100,200)"
    )
    
    # Coverage parameters
    parser.add_argument(
        "--n-samples", type=int, default=400,
        help="Number of LOS samples (default: 400)"
    )
    parser.add_argument(
        "--margin", type=float, default=0.0,
        help="Safety margin in meters (default: 0.0)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o", type=str, default=".",
        help="Output directory for results (default: current directory)"
    )
    parser.add_argument(
        "--export-kmz", action="store_true",
        help="Export coverage maps to KMZ file"
    )
    parser.add_argument(
        "--export-npz", action="store_true",
        help="Export coverage data to NPZ file"
    )
    parser.add_argument(
        "--save-figures", action="store_true",
        help="Save visualization figures to output directory"
    )
    
    # Visualization options
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show visualization windows"
    )
    parser.add_argument(
        "--show-3d", action="store_true",
        help="Show 3D terrain visualization"
    )
    parser.add_argument(
        "--no-coverage", action="store_true",
        help="Skip coverage computation (terrain visualization only)"
    )
    parser.add_argument(
        "--background", type=str, default="relief",
        choices=["relief", "hillshade", "basemap", "basemap+relief", "none"],
        help="Background type for coverage maps (default: relief)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    print_banner()
    
    # Validate terrain file
    terrain_path = Path(args.terrain_file)
    if not terrain_path.exists():
        print(f"Error: Terrain file not found: {terrain_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse flight levels
    try:
        flight_levels = [float(x.strip()) for x in args.fl.split(',') if x.strip()]
    except ValueError:
        print(f"Error: Invalid flight level format: {args.fl}")
        sys.exit(1)
    
    # Load terrain data
    print(f"Loading terrain from: {terrain_path}")
    try:
        lats, lons, Z = load_terrain_npz(str(terrain_path))
        print(f"  Grid size: {len(lats)} x {len(lons)} ({len(lats) * len(lons):,} points)")
        
        valid_z = Z[Z >= 0]
        if len(valid_z) > 0:
            print(f"  Elevation range: {valid_z.min():.1f}m - {valid_z.max():.1f}m")
    except Exception as e:
        print(f"Error loading terrain: {e}")
        sys.exit(1)
    
    # Convert to ENU coordinates
    print(f"\nConverting to ENU coordinates (ref: {args.ref_lat}°N, {args.ref_lon}°E)")
    X_m, Y_m = convert_to_enu(lats, lons, args.ref_lat, args.ref_lon)
    X_m, Y_m, Z_enu = normalize_xy_grid(X_m, Y_m, Z.copy())
    
    # Terrain visualization
    if args.visualize or args.save_figures:
        print("\nGenerating terrain visualizations...")
        
        # 2D terrain map
        fig_2d = plot_terrain_2d(lats, lons, Z, args.radar_lat, args.radar_lon,
                                  title="Terrain Elevation Map")
        if args.save_figures:
            fig_2d.savefig(output_dir / "terrain_2d.png", dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_dir / 'terrain_2d.png'}")
        
        # 3D terrain surface
        if args.show_3d:
            fig_3d = plot_terrain_3d(lats, lons, Z, args.radar_lat, args.radar_lon,
                                      title="3D Terrain Surface")
            if args.save_figures:
                fig_3d.savefig(output_dir / "terrain_3d.png", dpi=150, bbox_inches='tight')
                print(f"  Saved: {output_dir / 'terrain_3d.png'}")
        
        if args.visualize and not args.no_coverage:
            plt.show(block=False)
    
    # Coverage computation
    coverage_maps = None
    if not args.no_coverage:
        print(f"\nComputing coverage for FL: {', '.join([str(int(fl)) for fl in flight_levels])}")
        print(f"  Radar: ({args.radar_lat}°N, {args.radar_lon}°E) at {args.radar_height}m AGL")
        print(f"  LOS samples: {args.n_samples}, margin: {args.margin}m")
        
        def progress_callback(pct):
            bar_len = 40
            filled = int(bar_len * pct)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"\r  Progress: |{bar}| {pct*100:.0f}%", end='', flush=True)
        
        coverage_maps = compute_all_coverages(
            args.radar_lat, args.radar_lon, args.radar_height,
            flight_levels,
            X_m, Y_m, Z_enu,
            args.ref_lat, args.ref_lon,
            args.n_samples, args.margin,
            progress_callback=progress_callback
        )
        print("\n")
        
        # Print coverage statistics
        print("Coverage Statistics:")
        print("-" * 60)
        print(f"{'FL':>6} | {'Alt (m)':>10} | {'Alt (ft)':>10} | {'Coverage':>12}")
        print("-" * 60)
        
        for fl in sorted(coverage_maps.keys()):
            cov = coverage_maps[fl]
            visible = np.sum(cov)
            total = cov.size
            pct = 100.0 * visible / total
            alt_m = fl_to_m(fl)
            alt_ft = fl * 100
            print(f"{fl:>6.0f} | {alt_m:>10.0f} | {alt_ft:>10.0f} | {pct:>11.2f}%")
        print("-" * 60)
        
        # Coverage visualizations
        if args.visualize or args.save_figures:
            print("\nGenerating coverage visualizations...")
            
            for fl in sorted(coverage_maps.keys()):
                fig = plot_coverage_map(
                    coverage_maps[fl], lats, lons, fl,
                    terrain=Z if args.background != "none" else None,
                    radar_lat=args.radar_lat,
                    radar_lon=args.radar_lon,
                    background=args.background
                )
                
                if args.save_figures:
                    fig_path = output_dir / f"coverage_FL{int(fl)}.png"
                    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                    print(f"  Saved: {fig_path}")
                
                plt.close(fig)
            
            # All FLs grid
            if len(coverage_maps) > 1:
                fig_all = plot_all_coverages_grid(
                    coverage_maps, lats, lons,
                    terrain=Z,
                    radar_lat=args.radar_lat,
                    radar_lon=args.radar_lon,
                    background=args.background
                )
                
                if args.save_figures:
                    fig_path = output_dir / "coverage_all_fl.png"
                    fig_all.savefig(fig_path, dpi=150, bbox_inches='tight')
                    print(f"  Saved: {fig_path}")
                
                if args.visualize:
                    plt.show(block=False)
                else:
                    plt.close(fig_all)
    
    # Export KMZ
    if args.export_kmz and coverage_maps is not None:
        print("\nExporting to KMZ...")
        kmz_path = output_dir / "radar_coverage.kmz"
        
        kmz_bytes = export_all_coverages_kmz(
            coverage_maps, lats, lons,
            args.radar_lat, args.radar_lon
        )
        
        with open(kmz_path, 'wb') as f:
            f.write(kmz_bytes)
        
        print(f"  Saved: {kmz_path}")
    
    # Export NPZ
    if args.export_npz and coverage_maps is not None:
        print("\nExporting coverage data to NPZ...")
        npz_path = output_dir / "coverage_data.npz"
        
        np.savez_compressed(
            npz_path,
            lats=lats,
            lons=lons,
            terrain=Z,
            radar_lat=args.radar_lat,
            radar_lon=args.radar_lon,
            **{f'coverage_fl{int(fl)}': cov for fl, cov in coverage_maps.items()}
        )
        
        print(f"  Saved: {npz_path}")
    
    # Show visualizations if requested
    if args.visualize:
        print("\nDisplaying visualizations (close windows to exit)...")
        plt.show()
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
