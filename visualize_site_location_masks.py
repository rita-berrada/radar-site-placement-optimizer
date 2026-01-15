"""
Site Location Masks Visualization Module

This module provides functions to visualize geographical masks as PNG images.
Masks are overlaid on terrain maps with:
- Admissible areas: transparent (showing terrain underneath)
- Excluded areas: grey with partial opacity (overlaying terrain)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from matplotlib.colors import ListedColormap
from matplotlib import cm


def plot_masks_overlay(
    lats: np.ndarray,
    lons: np.ndarray,
    Z: np.ndarray,
    masks_dict: Dict[str, np.ndarray],
    nice_lat: Optional[float] = None,
    nice_lon: Optional[float] = None,
    save_path: Optional[str] = None,
    excluded_alpha: float = 0.75,
    excluded_color: str = 'grey'
) -> None:
    """
    Visualize geographical masks overlaid on terrain map.
    
    Admissible areas are transparent (showing terrain), excluded areas are
    grey with partial opacity overlaying the terrain.
    
    Parameters:
    -----------
    lats : np.ndarray
        1D array of latitude values (degrees)
    lons : np.ndarray
        1D array of longitude values (degrees)
    Z : np.ndarray
        2D array of terrain elevation (meters above sea level)
    masks_dict : Dict[str, np.ndarray]
        Dictionary mapping mask names to boolean arrays
        All masks must have shape (len(lats), len(lons))
        True = admissible, False = excluded
    nice_lat : float, optional
        Nice airport latitude to mark on maps
    nice_lon : float, optional
        Nice airport longitude to mark on maps
    save_path : str, optional
        If provided, save figure to this path
    excluded_alpha : float, optional
        Opacity for excluded areas overlay (default: 0.6)
    excluded_color : str, optional
        Color for excluded areas (default: 'grey')
    """
    n_masks = len(masks_dict)
    if n_masks == 0:
        raise ValueError("At least one mask must be provided")
    
    # Verify terrain shape
    if Z.shape != (len(lats), len(lons)):
        raise ValueError(f"Terrain shape mismatch: Z{Z.shape} vs ({len(lats)}, {len(lons)})")
    
    # Determine grid layout
    if n_masks == 1:
        n_rows, n_cols = 1, 1
    elif n_masks == 2:
        n_rows, n_cols = 1, 2
    elif n_masks <= 4:
        n_rows, n_cols = 2, 2
    elif n_masks <= 6:
        n_rows, n_cols = 2, 3
    else:
        n_rows = int(np.ceil(np.sqrt(n_masks)))
        n_cols = int(np.ceil(n_masks / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    
    # Handle single subplot case
    if n_masks == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Set extent for all maps
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    
    # Create meshgrid for terrain
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    for idx, (mask_name, mask) in enumerate(masks_dict.items()):
        ax = axes[idx]
        
        # Verify mask shape
        if mask.shape != (len(lats), len(lons)):
            raise ValueError(f"Mask '{mask_name}' has shape {mask.shape}, expected ({len(lats)}, {len(lons)})")
        
        # 1. Display terrain as base layer
        terrain_im = ax.contourf(lon_grid, lat_grid, Z, levels=20, cmap='terrain', 
                                 extent=extent, alpha=1.0, zorder=1)
        
        # 2. Overlay excluded areas (False = excluded) with grey and partial opacity
        # Create a mask for excluded areas only
        excluded_mask = ~mask  # Invert: True where excluded
        
        # Create RGBA array for overlay
        overlay = np.zeros((*mask.shape, 4))  # RGBA
        
        # Set excluded areas to grey with alpha
        excluded_color_rgba = plt.cm.colors.to_rgba(excluded_color, alpha=excluded_alpha)
        overlay[excluded_mask] = excluded_color_rgba
        
        # Admissible areas remain transparent (alpha=0)
        overlay[~excluded_mask] = (0, 0, 0, 0)  # Transparent
        
        # Display overlay
        ax.imshow(overlay, aspect='auto', origin='lower', extent=extent, 
                 interpolation='nearest', zorder=2)
        
        # Mark Nice airport if provided
        if nice_lat is not None and nice_lon is not None:
            ax.plot(nice_lon, nice_lat, 'k*', markersize=15, label='Nice Airport', 
                   zorder=10, markeredgewidth=2, markeredgecolor='white')
            ax.text(nice_lon, nice_lat, '  Nice Airport', fontsize=10, va='bottom', zorder=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Calculate and display statistics
        admissible_pct = np.sum(mask) / mask.size * 100
        excluded_pct = 100 - admissible_pct
        stats_text = f'Admissible: {admissible_pct:.1f}%\nExcluded: {excluded_pct:.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Set labels and title
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_title(mask_name, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add colorbar for terrain
        if idx == 0:  # Add colorbar only to first subplot
            cbar = plt.colorbar(terrain_im, ax=ax, pad=0.02)
            cbar.set_label('Elevation (m)', fontsize=10)
        
        # Add legend for first subplot only
        if idx == 0 and nice_lat is not None and nice_lon is not None:
            ax.legend(loc='upper right', fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_masks, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Geographical Masks for Radar Site Location (Overlay on Terrain)', 
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_single_mask_overlay(
    lats: np.ndarray,
    lons: np.ndarray,
    Z: np.ndarray,
    mask: np.ndarray,
    mask_name: str = "Mask",
    nice_lat: Optional[float] = None,
    nice_lon: Optional[float] = None,
    save_path: Optional[str] = None,
    excluded_alpha: float = 0.75,
    excluded_color: str = 'grey'
) -> None:
    """
    Visualize a single mask overlaid on terrain map.
    
    Parameters:
    -----------
    lats : np.ndarray
        1D array of latitude values (degrees)
    lons : np.ndarray
        1D array of longitude values (degrees)
    Z : np.ndarray
        2D array of terrain elevation (meters above sea level)
    mask : np.ndarray
        Boolean array of shape (len(lats), len(lons))
        True = admissible, False = excluded
    mask_name : str, optional
        Name for the mask (default: "Mask")
    nice_lat : float, optional
        Nice airport latitude to mark on map
    nice_lon : float, optional
        Nice airport longitude to mark on map
    save_path : str, optional
        If provided, save figure to this path
    excluded_alpha : float, optional
        Opacity for excluded areas overlay (default: 0.6)
    excluded_color : str, optional
        Color for excluded areas (default: 'grey')
    """
    plot_masks_overlay(
        lats, lons, Z, {mask_name: mask}, nice_lat, nice_lon, 
        save_path, excluded_alpha, excluded_color
    )
