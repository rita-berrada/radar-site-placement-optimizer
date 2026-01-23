# Modelling Radar Thales — Coverage, Masks & Scoring (BoGE DMW)

This repository contains a complete pipeline to:

1. Load DTED terrain
2. Compute **Line-of-Sight (LOS)** radar coverage maps at required Flight Levels
3. Build **constraint masks** (roads/buildings/protected areas/slope/etc.)
4. Generate **authorized candidate sites**
5. Score and rank candidates using a **Numba-accelerated exact LOS** method
6. Export results to **KML/KMZ** (Google Earth) and NPZ (reproducibility)

> **Project context:** CentraleSupélec — Bachelor of Global Engineering (Data & Modeling Weeks) with Thales.

---

## 1) Quick Start (Recommended Path)

### 1.1 Install
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows (PowerShell)
pip install -r requirements.txt
```

### 1.2 Verify terrain loading
```bash
python visualize_terrain.py
```

### 1.3 Compute coverage (single radar position)

**Reference version (slower, clear):**
```bash
python main_coverage.py
```

**Numba ENU exact LOS (recommended for full-grid performance):**
```bash
python FLs_numba_enu.py
```

**One FL full-map (Numba):**
```bash
python fl_fullmap_numba.py
```

### 1.4 Full pipeline: masks → candidates → scoring → export

1. Generate authorized candidates from constraints:
```bash
python generate_candidates_full_constraints.py

# or relaxed versions:
python generate_candidates_relaxed.py
python generate_candidates_no_residential.py
```

2. Score candidates (Numba exact LOS in ENU):
```bash
python run_scoring_numba_enu.py
```

3. Export top-scored sites to Google Earth:
```bash
python export_scored_points_weighted_kml.py
```

---

## 2) Terrain Data Format (NPZ)

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

## 3) Coordinate System: ENU + Earth Curvature Correction

To keep all physical computations consistent (distance, slope, LOS sampling), this project centralizes coordinate conversion using:

- `geo_utils.py` / `geo_utils_earth_curvature.py`

It converts (lat, lon) into a local tangent plane (ENU):

- **X (East)** meters from reference point
- **Y (North)** meters from reference point

and applies a Z correction for Earth curvature:

$$Z_{\text{ENU}} = Z_{\text{terrain}} - \frac{d^2}{2R}$$

This is important for 50km-scale geometry.

---

## 4) LOS Algorithm (Exact)

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

## 5) Flight Levels (Tender Requirement)

**Supported Flight Levels:**

- FL5, FL10, FL20, FL50, FL100, FL200, FL300, FL400

**Conversion:**

$$\text{alt}_m = \text{FL} \times 100 \times 0.3048$$

---

## 6) Coverage Computation Modules

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

## 7) Masks & Constraints (Site Selection)

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

## 8) Scoring & Ranking Candidates (Numba ENU)

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

## 10) Google Earth Export (KML/KMZ)

Exports are handled by:

- `export_kml.py`
- `export_scored_points_weighted_kml.py`
- Related `export_*_kml.py` utilities

---

## 11) Streamlit App (Interactive)

**Launch:**
```bash
streamlit run radar_coverage_app.py
```

The app allows:

- Terrain inspection
- Running coverage
- Viewing results with different backgrounds
- Exporting outputs

---

## 12) Project Structure
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

## 13) Troubleshooting

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

## 14) Contributing / Git Workflow

### Add README to your Git repo

1. Create or replace the README file:
```bash
touch README.md
# paste this content into README.md
```

2. Commit and push:
```bash
git add README.md
git commit -m "Add project README"
git push origin main
```

If you are working on a branch:
```bash
git push origin <branch-name>
# then open a Merge Request / Pull Request to main
```

---

## License

Educational use (CentraleSupélec / Thales project).