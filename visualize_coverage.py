"""
Coverage Visualization Module

This module provides functions to visualize radar coverage maps using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from matplotlib.colors import ListedColormap


def plot_coverage_map(
    coverage_map: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    flight_level: float,
    radar_lat: Optional[float] = None,
    radar_lon: Optional[float] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot a single coverage map as a simple 2D map.
    
    Parameters:
    -----------
    coverage_map : np.ndarray
        2D boolean array (True=visible, False=blocked)
    lats : np.ndarray
        1D array of latitude values
    lons : np.ndarray
        1D array of longitude values
    flight_level : float
        Flight level for title
    radar_lat : float, optional
        Radar latitude to mark on map
    radar_lon : float, optional
        Radar longitude to mark on map
    save_path : str, optional
        If provided, save figure to this path
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Convert boolean to float for visualization
    coverage_float = coverage_map.astype(float)
    
    # Create custom colormap: green for visible (True=1), red for blocked (False=0)
    colors = ['red', 'green']  # red=blocked, green=visible
    cmap = ListedColormap(colors)
    
    # Use imshow for simple 2D map (extent sets the coordinate system)
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    im = ax.imshow(coverage_float, cmap=cmap, aspect='auto', origin='lower', extent=extent, interpolation='nearest')
    
    # Mark radar position if provided
    if radar_lat is not None and radar_lon is not None:
        ax.plot(radar_lon, radar_lat, 'k*', markersize=20, label='Radar', zorder=10, markeredgewidth=2, markeredgecolor='white')
        ax.text(radar_lon, radar_lat, '  Radar', fontsize=12, va='bottom', zorder=10, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Calculate and display statistics
    coverage_pct = np.sum(coverage_map) / coverage_map.size * 100
    stats_text = f'Coverage: {coverage_pct:.1f}% visible'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Radar Coverage - FL{flight_level}', fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75], pad=0.02)
    cbar.set_ticklabels(['Blocked', 'Visible'])
    cbar.set_label('Coverage Status', fontsize=11)
    
    # Add legend if radar is marked
    if radar_lat is not None and radar_lon is not None:
        ax.legend(loc='upper right', fontsize=10)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save if path provided, otherwise show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()  # This blocks until window is closed
        plt.close(fig)


def plot_all_coverage_maps(
    coverage_maps: Dict[float, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    radar_lat: Optional[float] = None,
    radar_lon: Optional[float] = None
) -> None:
    """
    Plot all coverage maps in a grid layout (2 rows x 4 columns).
    
    Parameters:
    -----------
    coverage_maps : Dict[float, np.ndarray]
        Dictionary mapping flight level to coverage map
    lats : np.ndarray
        1D array of latitude values
    lons : np.ndarray
        1D array of longitude values
    radar_lat : float, optional
        Radar latitude to mark on maps
    radar_lon : float, optional
        Radar longitude to mark on maps
    """
    flight_levels = sorted(coverage_maps.keys())
    n_maps = len(flight_levels)
    
    # Create subplot grid: 2 rows x 4 columns for 8 maps
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Convert boolean to float for visualization
    coverage_float_base = coverage_maps[flight_levels[0]].astype(float)
    
    # Create custom colormap
    colors = ['red', 'green']  # red=blocked, green=visible
    cmap = ListedColormap(colors)
    
    # Set extent for all maps
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    
    for idx, fl in enumerate(flight_levels):
        ax = axes[idx]
        coverage_float = coverage_maps[fl].astype(float)
        
        # Use imshow for simple 2D map
        im = ax.imshow(coverage_float, cmap=cmap, aspect='auto', origin='lower', 
                      extent=extent, interpolation='nearest')
        
        # Mark radar position if provided
        if radar_lat is not None and radar_lon is not None:
            ax.plot(radar_lon, radar_lat, 'k*', markersize=12, label='Radar', 
                   zorder=10, markeredgewidth=1.5, markeredgecolor='white')
            ax.text(radar_lon, radar_lat, '  Radar', fontsize=9, va='bottom', zorder=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Calculate and display statistics
        coverage_pct = np.sum(coverage_maps[fl]) / coverage_maps[fl].size * 100
        stats_text = f'Coverage: {coverage_pct:.1f}% visible'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Set labels and title
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.set_title(f'Radar Coverage - FL{fl}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add legend for first subplot only
        if idx == 0 and radar_lat is not None and radar_lon is not None:
            ax.legend(loc='upper right', fontsize=9)
    
    # Add single colorbar for all subplots
    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticklabels(['Blocked', 'Visible'])
    cbar.set_label('Coverage Status', fontsize=11)
    
    plt.suptitle('Radar Coverage Maps - All Flight Levels', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])
    
    plt.show()
