# Radar Site Placement Optimizer

Interactive radar coverage analysis + site placement optimization.

This project is a **radar coverage analysis tool** where you load terrain data and then:

1. **Compute coverage** for **8 flight levels** from a radar defined by `(lat, lon, height)` using a terrain-aware LOS model.
2. **Create constraints/requirements** on the terrain (example: close to electrical stations, far from dwellings/buildings, outside protected areas, slope limits) to generate **candidate locations** where the radar can be placed.
3. **Score and rank candidates** by computing coverage at all flight levels for each candidate, then export the best locations.

The main deliverable is a **Streamlit UI** (`radar_coverage_app.py`) with 3 pages: Coverage Analysis, Site Selection, and Scoring Results.

---

## Quick Start (UI)

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows (PowerShell)
pip install -r requirements.txt
```

Launch the app:

```bash
streamlit run radar_coverage_app.py
```

The UI will open in your browser (typically `http://localhost:8501`).

---

## What You Can Do In The App

### 1) Coverage Analysis
- Load a terrain NPZ file
- Set radar position (`lat`, `lon`) and radar height
- Compute LOS coverage maps for the supported flight levels
- Export results (KMZ for Google Earth, PNG/CSV depending on the page options)

### 2) Site Selection (Constraints)
- Build constraint masks (allowed / forbidden) from your data and thresholds
- Combine constraints to generate admissible candidate locations
- Export masks and candidate lists

Examples of constraints this project supports:
- Electrical stations proximity / keep-away distance
- Buildings / dwellings exclusion buffers
- Protected areas exclusion
- Road proximity requirements
- Slope thresholds
- (Optional) airport visibility constraints, depending on available data

### 3) Scoring Results
- Compute coverage for each candidate at all flight levels
- Aggregate into a score and rank sites
- Export ranked results (including KMZ for visualization)

---

## Terrain Data Format (NPZ)

The terrain file must contain:

- `lat`: 1D array of latitudes (degrees, WGS84)
- `lon`: 1D array of longitudes (degrees, WGS84)
- `ter`: 2D terrain elevation array in meters above sea level (MSL), shape `(len(lat), len(lon))`

**Example:**
```python
import numpy as np

lats = np.linspace(43.3, 44.0, 500)
lons = np.linspace(6.7, 7.9, 600)
Z = np.random.rand(500, 600) * 1000

np.savez("terrain_mat.npz", lat=lats, lon=lons, ter=Z)
```

---

## Coordinate System (ENU) + Earth Curvature Correction

To keep all physical computations consistent (distance, slope, LOS sampling), this project centralizes coordinate conversion using:

- `geo_utils.py` / `geo_utils_earth_curvature.py`

It converts (lat, lon) into a local tangent plane (ENU):

- **X (East)** meters from reference point
- **Y (North)** meters from reference point

and applies a Z correction for Earth curvature:

$$Z_{\text{ENU}} = Z_{\text{terrain}} - \frac{d^2}{2R}$$

This is important for 50km-scale geometry.

---

## LOS Algorithm (Exact)

A target point is **visible** if along the segment radar → target:

- At every sampled point, the terrain altitude is strictly below the LOS altitude.

**Core logic:**

- Interpolate terrain at sampled points (bilinear interpolation)
- Compute LOS line altitude
- Early exit as soon as blocked

### Implementations

- **Reference (Python)**: `LOS.py`
- **Vectorized prototype**: `LOS_np.py`
- **Fast exact ENU + Numba**: `LOS_numba_enu.py` ✅ (recommended)

---

## Flight Levels

**Supported Flight Levels:**

- FL5, FL10, FL20, FL50, FL100, FL200, FL300, FL400

**Conversion:**

$$\text{alt}_m = \text{FL} \times 100 \times 0.3048$$

---

## CLI / Scripts (Optional)

The Streamlit app is the easiest way to explore the project, but the repository also contains standalone scripts (useful for batch runs).

### 6.1 Reference version (clear but slow)

- `coverage_analysis.py`
- `main_coverage.py`

### 6.2 Accelerated versions

- `coverage_analysis_fast.py` (approx / batching style)
- **Numba ENU full-grid**:
  - `LOS_numba_enu.py`
  - `FLs_numba_enu.py`
  - `score_numba_enu.py`
  - `run_scoring_numba_enu.py`

---

## Masks & Constraints (Site Selection)

Constraint masks are boolean grids where:

- `True` = site allowed
- `False` = site forbidden

**Typical masks:**

- **Road proximity**: keep sites close to roads / accessible
- **Buildings buffer**: forbid installation around buildings
- **Residential / protected areas**: exclude protected zones
- **Electrical stations**: optional keep-away constraint
- **Slope**: forbid terrain that exceeds a max slope threshold
- **Airport visibility**: ensure LOS constraints to airport if required

**Mask modules** (names may vary depending on the branch/version):

- `site_location_masks.py` (base/orchestration)
- `buildings_masks.py` / `buildings.py`
- `roads_masks.py`
- `protected_areas_masks.py`
- `electrical_stations_masks.py`
- `mask_slope.py`

**Candidate generation scripts:**

- `generate_candidates_full_constraints.py`
- `generate_candidates_relaxed.py`
- `generate_candidates_no_residential.py`

**Outputs** typically saved as NPZ:

- `authorized_points_all_masks.npz` containing arrays `lat`, `lon`

---

## Scoring & Ranking Candidates (Numba ENU)

The scoring stage:

1. Loads terrain once
2. For each candidate site, computes coverage % at each FL
3. Aggregates with weights, and sorts candidates by score

**Main scripts:**

- `score_numba_enu.py`
- `run_scoring_numba_enu.py`

**Typical output:**

- `scored_candidates_fullgrid.npz` with:
  - `lat`, `lon`
  - `score`
  - `cov_by_fl`

---

## 9) Visualization

**Coverage plotting:**

- `visualize_coverage.py`
- `plot_coverage_map(...)`
- `plot_all_coverage_maps(...)`

**Terrain plotting:**

- `visualize_terrain.py`

**Optional basemap tiles:**

- `contextily` (if enabled)

---

## Google Earth Export (KML/KMZ)

Exports are handled by:

- `export_kml.py`
- `export_scored_points_weighted_kml.py`
- Related `export_*_kml.py` utilities

---

## Project Structure
```text
modelling_radar_thales/
├── radar_coverage_app.py
├── radar_coverage_cli.py
├── main_coverage.py
├── coverage_analysis.py
├── coverage_analysis_fast.py
├── LOS.py
├── LOS_np.py
├── LOS_numba_enu.py
├── FLs_numba_enu.py
├── score_numba_enu.py
├── run_scoring_numba_enu.py
├── geo_utils.py
├── geo_utils_earth_curvature.py
├── visualize_terrain.py
├── visualize_coverage.py
├── export_kml.py
├── export_*_kml.py
├── mask_*.py
├── generate_candidates_*.py
├── requirements.txt
├── terrain_mat.npz
└── geographical_data/
    ├── buildings.geojson
    ├── roads_nice_50km.geojson
    ├── protected_areas.geojson
    └── export.geojson
```

---

## Troubleshooting

### First run is slow

Numba compiles on the first execution. Run once, then rerun → much faster.

### Coverage changes when changing grid size

Changing resolution changes the evaluated target points. For fair comparisons:

- Keep the same grid/subset
- Same radar height
- Same `n_samples`
- Same coordinate reference (ENU) and terrain normalization

### Lat/Lon decreasing issues

Some datasets store axes descending. Use normalization utilities to avoid `searchsorted` and plotting inconsistencies.

---

## Notes
- Large datasets (terrain/GeoJSON/KMZ) can make the repository heavy. If you want a lightweight version for sharing, keep small sample files in the repo and host the full datasets elsewhere.
