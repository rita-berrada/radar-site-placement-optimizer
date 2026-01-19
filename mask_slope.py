"""
mask_slope.py

Implements PROJECT_REQ_10: Slope constraint for the radar platform.
Uses the shared metric coordinate system from geo_utils.
"""

import numpy as np
from geo_utils import load_and_convert_to_enu

def mask_slope(npz_file, max_slope_percent=15.0):
    """
    Creates a boolean mask identifying areas where the terrain is flat enough.
    
    Parameters:
    -----------
    npz_file : str
        Path to the terrain .npz file.
    max_slope_percent : float
        Maximum allowable slope in percentage (default 15.0 for REQ_10).
        
    Returns:
    --------
    mask : np.ndarray (bool)
        True where slope <= max_slope_percent (Admissible).
        False where slope > max_slope_percent (Excluded).
    """
    
    # 1. Get data in Meters (ENU System)
    X_m, Y_m, Z, _, _ = load_and_convert_to_enu(npz_file)

    # 2. Calculate Grid Resolution (Step size in meters)
    dx = np.abs(X_m[1] - X_m[0])  
    dy = np.abs(Y_m[1] - Y_m[0])  

    # 3. Compute Gradient (The Slope)
    grad_y, grad_x = np.gradient(Z, dy, dx)

    # 4. Compute Slope Percentage
    slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    slope_percent = slope_magnitude * 100.0

    # 5. Create and Return Boolean Mask
    return slope_percent <= max_slope_percent