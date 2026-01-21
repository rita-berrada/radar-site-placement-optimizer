"""
mask_slope.py

Implements PROJECT_REQ_10: Slope constraint for the radar platform.
Adapted to work with the centralized ENU metric coordinate system.
"""

import numpy as np

def mask_slope(X_m, Y_m, Z_m, max_slope_percent=15.0):
    """
    Creates a boolean mask identifying areas where the terrain is flat enough.
    
    Now expects metric grids directly, avoiding redundant data loading.
    
    Parameters:
    -----------
    X_m : np.ndarray (1D)
        Distance in meters along the East-West axis.
    Y_m : np.ndarray (1D)
        Distance in meters along the North-South axis.
    Z_m : np.ndarray (2D)
        Altitude in meters (Corrected ENU Z provided by the main script).
    max_slope_percent : float
        Maximum allowable slope in percentage (default 15.0).
        
    Returns:
    --------
    mask : np.ndarray (bool)
        True where slope <= max_slope_percent (Admissible).
        False where slope > max_slope_percent (Excluded).
    """
    
    # 1. Calculate Grid Resolution (Step size in meters)
    # We use the metric axes passed as arguments
    dx = np.abs(X_m[1] - X_m[0])
    dy = np.abs(Y_m[1] - Y_m[0])

    # 2. Compute Gradient (The Slope)
    # np.gradient computes the change in Z for each step in Y (axis 0) and X (axis 1)
    grad_y, grad_x = np.gradient(Z_m, dy, dx)

    # 3. Compute Slope Percentage
    # Slope magnitude = sqrt( (dZ/dX)^2 + (dZ/dY)^2 )
    slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    slope_percent = slope_magnitude * 100.0

    # 4. Create and Return Boolean Mask
    return slope_percent <= max_slope_percent