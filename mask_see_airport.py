"""
mask_see_airport.py

Handles Line of Sight (LOS) checks for the radar.
Updates:
- Uses the high-performance Numba engine (LOS_numba_enu.py).
- Works with the centralized ENU metric coordinate system.
- Checks visibility on 'Z_corrected' (Earth curvature already applied).
"""

import numpy as np
from numba import njit, prange

# Import the high-performance logic from your new file
from LOS_numba_enu import los_visible_numba_xy, z_bilinear_uniform_xy, latlon_to_xy_m

# Internal Numba function for parallel batch processing
# We define it here to avoid modifying LOS_numba_enu.py
@njit(parallel=True)
def _batch_process_los(
    cand_x_arr, cand_y_arr,
    radar_h_agl,
    target_x, target_y, target_alt_msl,
    x0, y0, dx, dy, Z_corrected,
    n_samples, margin_m
):
    """
    Parallel loop over all candidates to check LOS.
    """
    n_cands = cand_x_arr.shape[0]
    results = np.zeros(n_cands, dtype=np.bool_)

    for i in prange(n_cands):
        # Coordinates of the current candidate (Radar position)
        rx = cand_x_arr[i]
        ry = cand_y_arr[i]

        # Check visibility to Target
        is_visible = los_visible_numba_xy(
            rx, ry, radar_h_agl,
            target_x, target_y, target_alt_msl,
            x0, y0, dx, dy, Z_corrected,
            n_samples, margin_m
        )
        results[i] = is_visible

    return results


def check_visibility_batch(X_axis, Y_axis, Z_corrected, candidates_indices, 
                           target_lat, target_lon, 
                           radar_height_m=20.0, target_height_m=10.0):
    """
    Checks Line of Sight (LOS) for a list of candidate points using Numba.
    
    Parameters:
    -----------
    X_axis : np.ndarray (1D)
        X coordinates of the grid columns (meters).
    Y_axis : np.ndarray (1D)
        Y coordinates of the grid rows (meters).
    Z_corrected : np.ndarray (2D)
        Elevation matrix with Earth curvature ALREADY SUBTRACTED.
    candidates_indices : Tuple (rows, cols)
        Indices of candidate points to test (from np.where).
    target_lat, target_lon : float
        Coordinates of the airport (Target).
    radar_height_m : float
        Radar height above ground (at candidate position).
    target_height_m : float
        Target tower height above ground (at airport position).
        
    Returns:
    --------
    np.ndarray : Boolean array (True = Visible, False = Obstructed).
    """
    
    # 1. Setup Grid Parameters for Numba
    # We assume uniform grid steps
    dx = np.abs(X_axis[1] - X_axis[0])
    dy = np.abs(Y_axis[1] - Y_axis[0])
    x0 = X_axis[0]
    y0 = Y_axis[0]

    # 2. Locate Target (Airport) in ENU Meter System
    # We use the helper from LOS_numba_enu to be consistent
    target_x, target_y = latlon_to_xy_m(target_lat, target_lon)
    
    # 3. Get Target Altitude (Ground + Tower)
    # We use bilinear interpolation on the corrected Z grid
    # (At 0,0 or near Ref, Z_corrected is roughly equal to Z_terrain)
    target_z_ground = z_bilinear_uniform_xy(target_x, target_y, x0, y0, dx, dy, Z_corrected)
    
    if np.isnan(target_z_ground):
        print("   [Warn] Target (Airport) is outside the grid! Assuming Z=0.")
        target_z_ground = 0.0
        
    target_alt_msl = target_z_ground + target_height_m

    # 4. Prepare Candidate Coordinates
    cand_rows = candidates_indices[0]
    cand_cols = candidates_indices[1]
    
    # Extract X and Y for all candidates using the indices
    # We convert to contiguous arrays for Numba efficiency
    cand_x_arr = np.ascontiguousarray(X_axis[cand_cols])
    cand_y_arr = np.ascontiguousarray(Y_axis[cand_rows])
    
    num_candidates = len(cand_x_arr)
    print(f"   ... Computing Numba-Accelerated LOS for {num_candidates} candidates ...")

    # 5. Determine Sampling
    # Calculate max distance to adapt n_samples dynamically or use fixed
    # Here we use a robust fixed logic: roughly 1 point every 100m is usually enough
    # But for Numba speed, we can be generous, e.g., 500 samples per line.
    n_samples = 500 
    margin_m = 2.0  # 2m safety margin (clearing vegetation/errors)

    # 6. Run the Numba Parallel Function
    visible_mask = _batch_process_los(
        cand_x_arr, cand_y_arr,
        float(radar_height_m),
        float(target_x), float(target_y), float(target_alt_msl),
        float(x0), float(y0), float(dx), float(dy), Z_corrected,
        int(n_samples), float(margin_m)
    )
    
    return visible_mask