"""
Optimized Coverage Visualization Module

Provides multiple visualization options for radar coverage maps:
1. Full quality visualization (slower but accurate)
2. Quick preview with downsampling (much faster)
3. Single FL detailed view
4. Comparison views
5. Statistics bar chart
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


def plot_all_coverage_maps(
    coverage_maps: Dict[float, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    radar_lat: float,
    radar_lon: float,
    downsample: int = 1,
    use_imshow: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot all coverage maps in a grid layout.
    
    Parameters:
    -----------
    coverage_maps : dict
        Dictionary mapping flight levels to coverage boolean arrays
    lats, lons : np.ndarray
        Latitude and longitude grid coordinates
    radar_lat, radar_lon : float
        Radar position
    downsample : int
        Downsampling factor (1=full resolution, 2=half, etc.)
    use_imshow : bool
        If True, use imshow (faster), otherwise pcolormesh (slower but more accurate)
    save_path : str, optional
        If provided, save figure to this path instead of showing
    """
    
    # Apply downsampling
    if downsample > 1:
        lats_plot = lats[::downsample]
        lons_plot = lons[::downsample]
    else:
        lats_plot = lats
        lons_plot = lons
    
    # Setup figure
    n_maps = len(coverage_maps)
    cols = 4
    rows = (n_maps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    if n_maps == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each coverage map
    for idx, (fl, coverage) in enumerate(sorted(coverage_maps.items())):
        ax = axes[idx]
        
        # Downsample coverage
        if downsample > 1:
            coverage_plot = coverage[::downsample, ::downsample]
        else:
            coverage_plot = coverage
        
        # Calculate coverage percentage
        coverage_pct = np.sum(coverage) / coverage.size * 100
        
        if use_imshow:
            # Fast rendering with imshow
            extent = [lons_plot[0], lons_plot[-1], lats_plot[0], lats_plot[-1]]
            im = ax.imshow(
                coverage_plot,
                extent=extent,
                origin='lower',
                cmap='RdYlGn',
                aspect='auto',
                interpolation='nearest'
            )
        else:
            # More accurate but slower with pcolormesh
            lon_mesh, lat_mesh = np.meshgrid(lons_plot, lats_plot)
            im = ax.pcolormesh(
                lon_mesh, lat_mesh, coverage_plot,
                cmap='RdYlGn',
                shading='auto'
            )
        
        # Add radar position
        ax.plot(radar_lon, radar_lat, 'r*', markersize=15, 
                markeredgecolor='black', markeredgewidth=1, label='Radar')
        
        # Formatting
        ax.set_title(f'FL{fl:.0f} - Coverage: {coverage_pct:.1f}%', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Visible', rotation=270, labelpad=15)
    
    # Hide unused subplots
    for idx in range(n_maps, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Radar Coverage Analysis by Flight Level', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        print(f"Saving figure to {save_path}...")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved!")
    else:
        plt.show()


def quick_plot_coverage(
    coverage_maps: Dict[float, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    radar_lat: float,
    radar_lon: float,
    selected_fls: Optional[List[float]] = None,
    downsample: int = 5
):
    """
    Ultra-fast preview with aggressive downsampling.
    Perfect for quick iterations and testing.
    
    Parameters:
    -----------
    selected_fls : list, optional
        List of flight levels to display (e.g., [5, 50, 200, 400])
        If None, displays all
    downsample : int
        Downsampling factor (default: 5 = 1 pixel every 5)
    """
    
    if selected_fls is None:
        selected_fls = sorted(coverage_maps.keys())
    
    n_maps = len(selected_fls)
    cols = min(4, n_maps)
    rows = (n_maps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_maps == 1:
        axes = [axes]
    elif rows > 1:
        axes = axes.flatten()
    
    lats_ds = lats[::downsample]
    lons_ds = lons[::downsample]
    
    for idx, fl in enumerate(selected_fls):
        if idx >= len(axes):
            break
        
        ax = axes[idx] if isinstance(axes, (list, np.ndarray)) else axes
        coverage_ds = coverage_maps[fl][::downsample, ::downsample]
        coverage_pct = np.sum(coverage_maps[fl]) / coverage_maps[fl].size * 100
        
        extent = [lons_ds[0], lons_ds[-1], lats_ds[0], lats_ds[-1]]
        ax.imshow(coverage_ds, extent=extent, origin='lower',
                  cmap='RdYlGn', aspect='auto', interpolation='nearest')
        
        ax.plot(radar_lon, radar_lat, 'r*', markersize=12, 
                markeredgecolor='black', markeredgewidth=0.5)
        ax.set_title(f'FL{fl:.0f} ({coverage_pct:.1f}%)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Lon')
        ax.set_ylabel('Lat')
    
    # Hide unused subplots
    if isinstance(axes, (list, np.ndarray)):
        for idx in range(n_maps, len(axes)):
            axes[idx].axis('off')
    
    plt.suptitle(f'Quick Coverage Preview (1:{downsample} downsampling)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_single_fl(
    coverage_map: np.ndarray,
    fl: float,
    lats: np.ndarray,
    lons: np.ndarray,
    radar_lat: float,
    radar_lon: float,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Plot a single flight level coverage map with detailed information.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate statistics
    coverage_pct = np.sum(coverage_map) / coverage_map.size * 100
    n_visible = np.sum(coverage_map)
    n_total = coverage_map.size
    
    # Plot coverage
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    im = ax.pcolormesh(lon_mesh, lat_mesh, coverage_map, 
                       cmap='RdYlGn', shading='auto')
    
    # Add radar
    ax.plot(radar_lon, radar_lat, 'r*', markersize=20, 
            markeredgecolor='black', markeredgewidth=2, label='Radar', zorder=5)
    
    # Formatting
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Flight Level {fl:.0f} Coverage Map', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.legend(fontsize=12)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Line of Sight Status', rotation=270, labelpad=20, fontsize=11)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Blocked', 'Visible'])
    
    # Add text box with statistics
    textstr = f'Coverage: {coverage_pct:.2f}%\n'
    textstr += f'Visible: {n_visible:,} / {n_total:,} points\n'
    textstr += f'Altitude: {fl*100:.0f} ft ({fl*30.48:.0f} m)'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()


def plot_coverage_comparison(
    coverage_maps: Dict[float, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    radar_lat: float,
    radar_lon: float,
    fl_pairs: List[Tuple[float, float]] = [(5, 400), (50, 200)]
):
    """
    Compare coverage between pairs of flight levels side-by-side.
    
    Parameters:
    -----------
    fl_pairs : list of tuples
        Pairs of flight levels to compare, e.g., [(5, 400), (50, 200)]
    """
    n_pairs = len(fl_pairs)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(16, 6*n_pairs))
    
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    for pair_idx, (fl1, fl2) in enumerate(fl_pairs):
        for col_idx, fl in enumerate([fl1, fl2]):
            ax = axes[pair_idx, col_idx]
            coverage = coverage_maps[fl]
            coverage_pct = np.sum(coverage) / coverage.size * 100
            
            lon_mesh, lat_mesh = np.meshgrid(lons, lats)
            im = ax.pcolormesh(lon_mesh, lat_mesh, coverage,
                              cmap='RdYlGn', shading='auto')
            
            ax.plot(radar_lon, radar_lat, 'r*', markersize=15,
                   markeredgecolor='black', markeredgewidth=1)
            
            ax.set_title(f'FL{fl:.0f} - {coverage_pct:.1f}% coverage',
                        fontsize=13, fontweight='bold')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.3)
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Flight Level Coverage Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_coverage_statistics(coverage_maps: Dict[float, np.ndarray]):
    """
    Plot coverage statistics as a bar chart.
    """
    fls = sorted(coverage_maps.keys())
    coverages = [np.sum(coverage_maps[fl]) / coverage_maps[fl].size * 100 
                 for fl in fls]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(len(fls)), coverages, color='skyblue', 
                  edgecolor='navy', linewidth=1.5)
    
    # Color bars based on coverage percentage
    for bar, cov in zip(bars, coverages):
        if cov < 30:
            bar.set_color('salmon')
        elif cov < 70:
            bar.set_color('khaki')
        else:
            bar.set_color('lightgreen')
    
    ax.set_xlabel('Flight Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Radar Coverage by Flight Level', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(fls)))
    ax.set_xticklabels([f'FL{fl:.0f}' for fl in fls])
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    
    # Add percentage labels on bars
    for i, (bar, cov) in enumerate(zip(bars, coverages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{cov:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()


# Module info
if __name__ == "__main__":
    print("✓ Coverage visualization module loaded successfully!")
    print("\nAvailable functions:")
    print("  - plot_all_coverage_maps()      : Full grid of all FLs")
    print("  - quick_plot_coverage()         : Fast preview (downsampled)")
    print("  - plot_single_fl()              : Detailed single FL view")
    print("  - plot_coverage_comparison()    : Side-by-side FL comparison")
    print("  - plot_coverage_statistics()    : Bar chart of coverage %")