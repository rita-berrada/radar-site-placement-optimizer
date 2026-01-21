"""
optimize_zones.py

Pre-processing script:
1. Loads residential areas (export.geojson) and buildings (buildings.geojson).
2. Cleans geometries (fixes invalid polygons).
3. Merges EVERYTHING into a single layer.
4. Dissolves overlapping zones into one unified shape.

Result: A single, optimized 'forbidden_zones_optimized.geojson' file.
"""

import geopandas as gpd
import pandas as pd
import os

def optimize_geojson():
    print("1. Loading GeoJSON files...")
    
    # Charger les fichiers s'ils existent
    gdfs = []
    
    if os.path.exists("export.geojson"):
        print("   -> Loading export.geojson (Residential)...")
        gdf_res = gpd.read_file("export.geojson")
        # On ne garde que la géométrie pour alléger
        gdf_res = gdf_res[['geometry']]
        gdfs.append(gdf_res)
    else:
        print("   [!] export.geojson not found!")

    if os.path.exists("buildings.geojson"):
        print("   -> Loading buildings.geojson...")
        gdf_build = gpd.read_file("buildings.geojson")
        gdf_build = gdf_build[['geometry']]
        gdfs.append(gdf_build)
    else:
        print("   [!] buildings.geojson not found!")

    if not gdfs:
        print("Error: No files to process.")
        return

    # Concaténer tout
    print("2. Combining layers...")
    combined = pd.concat(gdfs, ignore_index=True)
    print(f"   -> Total polygons before optimization: {len(combined)}")

    # Nettoyage des géométries invalides (Self-intersection, etc.)
    # L'astuce .buffer(0) répare souvent les polygones cassés
    print("3. Fixing invalid geometries...")
    combined['geometry'] = combined.geometry.buffer(0)

    # DISSOLVE (La fusion magique)
    print("4. Dissolving overlapping zones (This may take a moment)...")
    # On crée une colonne commune pour tout fusionner en une seule ligne
    combined['group'] = 1
    optimized = combined.dissolve(by='group')

    print(f"   -> Optimization complete. Everything merged into {len(optimized)} multipolygon.")

    # Sauvegarde
    output_file = "forbidden_zones_optimized.geojson"
    optimized.to_file(output_file, driver='GeoJSON')
    print(f"5. Saved to {output_file}")

if __name__ == "__main__":
    optimize_geojson()