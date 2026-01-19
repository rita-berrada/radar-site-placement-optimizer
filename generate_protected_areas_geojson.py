import json

def create_geojson():
    # Coordinates format: [Longitude, Latitude]
    
    # 1. Mercantour National Park (Approximate North-East zone)
    mercantour_coords = [
        [6.80, 44.00], 
        [7.20, 43.90], 
        [7.50, 43.85],
        [7.70, 43.90], 
        [7.70, 44.30], 
        [6.80, 44.30], 
        [6.80, 44.00]
    ]

    # 2. Prealpes d'Azur Regional Park (Approximate West zone)
    prealpes_coords = [
        [6.80, 43.65], 
        [7.15, 43.68], 
        [7.15, 43.85],
        [6.80, 43.85], 
        [6.80, 43.65]
    ]

    # 3. Nice Urban Area (City center exclusion)
    nice_urban_coords = [
        [7.22, 43.68],
        [7.30, 43.68],
        [7.30, 43.72],
        [7.22, 43.72],
        [7.22, 43.68]
    ]

    # Build GeoJSON structure
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Mercantour National Park"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [mercantour_coords]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": "Prealpes d'Azur Park"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [prealpes_coords]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": "Nice Urban Area"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [nice_urban_coords]
                }
            }
        ]
    }

    output_file = "protected_areas.geojson"
    with open(output_file, "w") as f:
        json.dump(geojson_data, f, indent=2)

    print(f"Success: {output_file} created.")

if __name__ == "__main__":
    create_geojson()