# Radar Coverage Analysis Software

A comprehensive radar coverage analysis tool for computing and visualizing line-of-sight (LOS) coverage maps over terrain data.

## Features

### Core Functionality

- **Terrain Visualization**: Display terrain elevation data in 2D contour maps and interactive 3D surface plots
- **Coverage Analysis**: Compute radar visibility for multiple Flight Levels (FL) using optimized Numba-accelerated LOS calculations
- **Multiple Background Options**: Visualize coverage with various backgrounds:
  - Colored terrain relief with hillshading
  - Grayscale hillshade
  - OpenStreetMap/CartoDB basemap tiles (Google Maps-like)
  - Combined basemap + terrain relief
- **KMZ Export**: Export coverage results to KMZ format for Google Earth visualization
- **NPZ Export**: Save coverage data for further analysis

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://gitlab-student.centralesupelec.fr/soraya.essekkat/modelling_radar_thales.git
cd modelling_radar_thales
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- **numpy**: Array operations and terrain data handling
- **matplotlib**: Visualization and plotting
- **streamlit**: Web application framework
- **numba**: JIT compilation for performance (highly recommended)
- **contextily**: Basemap tiles (optional, for Google Maps-like backgrounds)

## Usage

### Web Application (Streamlit)

Launch the interactive web interface:

```bash
streamlit run radar_coverage_app.py
```

The application provides:
1. **Terrain View Tab**: 2D and 3D terrain visualization
2. **Coverage Analysis Tab**: Configure and compute coverage for selected Flight Levels
3. **Results Tab**: View coverage maps with different backgrounds
4. **Export Tab**: Download KMZ files for Google Earth

### Command Line Interface

For batch processing or automated workflows:

```bash
# Basic usage with default settings
python radar_coverage_cli.py terrain_mat.npz

# Specify flight levels and export to KMZ
python radar_coverage_cli.py terrain.npz --fl 10,50,100,200 --export-kmz

# Visualize terrain only (no coverage computation)
python radar_coverage_cli.py terrain.npz --visualize --no-coverage --show-3d

# Custom radar position
python radar_coverage_cli.py terrain.npz --radar-lat 43.70 --radar-lon 7.25 --radar-height 30

# Save all figures to output directory
python radar_coverage_cli.py terrain.npz --output ./results --save-figures --export-kmz --export-npz

# Full options
python radar_coverage_cli.py terrain.npz \
    --fl 5,10,20,50,100,200 \
    --radar-lat 43.6584 \
    --radar-lon 7.2159 \
    --radar-height 20 \
    --n-samples 400 \
    --background relief \
    --output ./output \
    --save-figures \
    --export-kmz \
    --export-npz \
    --visualize
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--radar-lat` | Radar latitude (degrees) | 43.6584 |
| `--radar-lon` | Radar longitude (degrees) | 7.2159 |
| `--radar-height` | Radar height AGL (meters) | 20.0 |
| `--fl` | Flight levels (comma-separated) | 10,50,100,200 |
| `--n-samples` | LOS ray samples | 400 |
| `--margin` | Safety margin (meters) | 0.0 |
| `--background` | Map background type | relief |
| `--output` | Output directory | . |
| `--export-kmz` | Export to KMZ | False |
| `--export-npz` | Export coverage data | False |
| `--save-figures` | Save PNG figures | False |
| `--visualize` | Show interactive plots | False |
| `--show-3d` | Include 3D terrain view | False |

## Terrain Data Format

The terrain NPZ file must contain:

- `lat`: 1D array of latitude values (degrees, WGS84)
- `lon`: 1D array of longitude values (degrees, WGS84)
- `ter`: 2D array of terrain elevation (meters MSL), shape (len(lat), len(lon))

Example:
```python
import numpy as np

# Create sample terrain data
lats = np.linspace(43.0, 44.0, 500)
lons = np.linspace(6.5, 8.0, 600)
terrain = np.random.rand(500, 600) * 1000  # Elevation in meters

np.savez('my_terrain.npz', lat=lats, lon=lons, ter=terrain)
```

## Flight Levels

The software supports 8 standard Flight Levels:
- FL5 = 500 ft = ~152 m
- FL10 = 1,000 ft = ~305 m
- FL20 = 2,000 ft = ~610 m
- FL50 = 5,000 ft = ~1,524 m
- FL100 = 10,000 ft = ~3,048 m
- FL200 = 20,000 ft = ~6,096 m
- FL300 = 30,000 ft = ~9,144 m
- FL400 = 40,000 ft = ~12,192 m

Conversion: `altitude_meters = FL * 100 * 0.3048`

## Project Structure

```
modelling_radar_thales/
├── radar_coverage_app.py     # Main Streamlit web application
├── radar_coverage_cli.py     # Command-line interface
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── # Core modules (used by the application):
├── visualize_terrain.py      # Terrain visualization utilities
├── LOS_numba_enu.py          # Numba-optimized LOS calculations
├── coverage_analysis.py      # Coverage computation module
├── visualize_coverage.py     # Coverage visualization utilities
├── export_kml.py             # KML/KMZ export functionality
├── geo_utils.py              # Geographic coordinate utilities
│
├── # Sample data:
├── terrain_mat.npz           # Sample terrain data (Nice area)
├── terrain_req01_50km.npz    # Extended terrain data
│
└── geographical_data/        # GeoJSON reference data
    ├── buildings.geojson
    ├── roads_nice_50km.geojson
    └── protected_areas.geojson
```

## Algorithm

The coverage analysis uses a Line-of-Sight (LOS) algorithm:

1. **Terrain Loading**: Load elevation data and convert to ENU (East-North-Up) coordinates
2. **Grid Normalization**: Ensure grid axes are properly ordered for interpolation
3. **LOS Ray Tracing**: For each grid point at a given Flight Level:
   - Cast a ray from radar to target point
   - Sample terrain elevation along the ray
   - Check if ray clears terrain at all sample points
4. **Coverage Map**: Generate boolean map indicating visible (True) or blocked (False)

The algorithm is optimized using Numba JIT compilation with parallel processing for significant performance improvements.

## Output Formats

### KMZ (Google Earth)
- Polygons colored by visibility status (green=visible, red=blocked)
- Radar position marker
- Organized by Flight Level folders

### PNG (Figures)
- High-resolution coverage maps with terrain backgrounds
- Coverage statistics overlay
- Radar position marker

### NPZ (Data)
- Full coverage arrays for each Flight Level
- Terrain and coordinate data
- Radar parameters

## Performance

With Numba acceleration enabled:
- ~500x500 grid: ~2-5 seconds per Flight Level
- ~1000x1000 grid: ~10-20 seconds per Flight Level

Without Numba (pure Python):
- Significantly slower (10-100x)
- Recommended only for small grids

## License

This project is developed for educational purposes at CentraleSupelec.

## Authors

- Radar Analysis Team
- CentraleSupelec / Thales collaboration
