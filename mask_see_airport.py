"""
mask_see_airport.py

Wrapper for LOS_np.py.
Handles batch processing of candidate sites against the airport target.
"""

import numpy as np
from LOS_np import los_visible, z_terrain_vec

def check_visibility_batch(lats, lons, Z, candidates_indices, target_lat, target_lon, 
                           radar_height_m=20.0, target_height_m=10.0):
    """
    Checks Line of Sight (LOS) for a list of candidate points using LOS_np logic.
    
    Parameters:
    -----------
    lats, lons : Grid coordinate arrays (1D).
    Z : Terrain elevation matrix (2D).
    candidates_indices : Tuple (rows, cols) of points to test.
    target_lat, target_lon : Coordinates of the airport.
    radar_height_m : Height of the radar tower above ground (default 20m).
    target_height_m : Height of the target tower above ground (default 10m).
    
    Returns:
    --------
    np.ndarray : Boolean array (True = Visible, False = Obstructed).
    """
    
    # 1. Get Airport Altitude (Ground + Tower)
    # We use z_terrain_vec to get the exact interpolated ground height at the airport
    target_z_ground = z_terrain_vec([target_lat], [target_lon], lats, lons, Z)[0]
    
    if np.isnan(target_z_ground):
        # Fallback if airport is slightly out of grid or on sea level 0
        target_z_ground = 0.0
        
    target_z_total = target_z_ground + target_height_m
    
    # 2. Prepare Loop
    cand_rows = candidates_indices[0]
    cand_cols = candidates_indices[1]
    num_candidates = len(cand_rows)
    
    visible_mask = np.zeros(num_candidates, dtype=bool)

    print(f"   ... Computing High-Precision LOS for {num_candidates} candidates ...")
    print("       (This uses bilinear interpolation, please wait...)")

    # 3. Iterate over each candidate
    # Note: LOS_np.los_visible is for a single path, so we loop.
    for k in range(num_candidates):
        r, c = cand_rows[k], cand_cols[k]
        
        cand_lat = lats[r]
        cand_lon = lons[c]
        
        # Call the robust function from your file
        is_clear = los_visible(
            radar_lat=cand_lat,
            radar_lon=cand_lon,
            radar_height_agl_m=radar_height_m,
            target_lat=target_lat,
            target_lon=target_lon,
            target_alt_m_msl=target_z_total,
            lats=lats,
            lons=lons,
            Z=Z,
            n_samples=200,  # 200 samples along the line is usually enough for accuracy
            margin_m=1.0    # 1m safety margin
        )
        
        visible_mask[k] = is_clear

        # Progress indicator every 500 points
        if (k + 1) % 500 == 0:
            print(f"       Processed {k + 1}/{num_candidates}...")

    return visible_mask