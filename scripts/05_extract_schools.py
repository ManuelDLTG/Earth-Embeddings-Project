"""
Script 05: Distancia y densidad de escuelas — OSM + NCES
Fuente principal: OpenStreetMap via osmnx (sin API key)
Fuente secundaria: NCES (si OSM falla)

Features generadas:
  - dist_nearest_school_m: distancia en metros a la escuela más cercana
  - schools_1km:           número de escuelas dentro de 1km

Instalación:
    pip install osmnx geopandas shapely pandas tqdm
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from tqdm import tqdm
import requests

INPUT_CSV  = "../data/redfin_stlouis_clean.csv"
OUTPUT_CSV = "../data/stlouis_sf_schools.csv"

Path("../data").mkdir(exist_ok=True)

# ── 1. Descargar escuelas via OSM ─────────────────────────────────────────────
print("Descargando escuelas de St. Louis City via OSM...")
try:
    import osmnx as ox
    features = ox.features_from_place(
        "St. Louis City, Missouri, USA",
        tags={"amenity": ["school","kindergarten","college","university"]}
    )
    # Filtrar solo los que tienen geometría válida
    features = features[features.geometry.notna()].copy()

    # Convertir todo a centroides en WGS84
    features_proj = features.to_crs(epsg=3857)
    centroids     = features_proj.geometry.centroid.to_crs(epsg=4326)

    df_schools = pd.DataFrame({
        "lat":  centroids.y.values,
        "lon":  centroids.x.values,
        "name": features.get("name", pd.Series(["Unknown"]*len(features))).values,
    }).dropna(subset=["lat","lon"])

    print(f"  Escuelas OSM: {len(df_schools)}")

except Exception as e:
    print(f"  OSM error: {e} — intentando NCES...")
    # Fallback: NCES API
    try:
        NCES_URL = (
            "https://nces.ed.gov/opengis/rest/services/K12_School_Locations/"
            "EDGE_GEOCODE_PUBLICSCH_2324/MapServer/0/query"
        )
        params = {
            "where":             "LSTATE='MO' AND LCITY='ST LOUIS'",
            "outFields":         "SCH_NAME,LAT,LON",
            "returnGeometry":    "false",
            "f":                 "json",
            "resultRecordCount": 500,
        }
        r = requests.get(NCES_URL, params=params, timeout=30)
        feats = r.json().get("features", [])
        records = []
        for feat in feats:
            p = feat.get("attributes", {})
            if p.get("LAT") and p.get("LON"):
                records.append({
                    "lat":  float(p["LAT"]),
                    "lon":  float(p["LON"]),
                    "name": p.get("SCH_NAME",""),
                })
        df_schools = pd.DataFrame(records)
        print(f"  Escuelas NCES: {len(df_schools)}")
    except Exception as e2:
        print(f"  NCES también falló: {e2}")
        df_schools = pd.DataFrame(columns=["lat","lon","name"])

if len(df_schools) == 0:
    print("⚠ Sin escuelas disponibles — generando ceros.")
    df = pd.read_csv(INPUT_CSV)
    df_sf = df[df["property_type"]=="Single Family Residential"].dropna(subset=["lat","lon"])
    pd.DataFrame({
        "url":                    df_sf["url"].values,
        "dist_nearest_school_m":  np.full(len(df_sf), np.nan),
        "schools_1km":            np.zeros(len(df_sf)),
    }).to_csv(OUTPUT_CSV, index=False)
    print(f"✓ {OUTPUT_CSV}")
    exit(0)

# ── 2. Calcular distancias ────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df_sf = df[df["property_type"]=="Single Family Residential"].dropna(
    subset=["lat","lon"]).reset_index(drop=True)

gdf_schools = gpd.GeoDataFrame(
    df_schools,
    geometry=[Point(lon, lat) for lon, lat in
              zip(df_schools["lon"], df_schools["lat"])],
    crs="EPSG:4326"
).to_crs(epsg=3857)

gdf_prop = gpd.GeoDataFrame(
    df_sf,
    geometry=[Point(lon, lat) for lon, lat in
              zip(df_sf["lon"], df_sf["lat"])],
    crs="EPSG:4326"
).to_crs(epsg=3857)

print(f"\nCalculando distancias a escuelas ({len(df_sf)} propiedades)...")
results = []
for _, row in tqdm(gdf_prop.iterrows(), total=len(gdf_prop)):
    pt    = row.geometry
    dists = gdf_schools.geometry.distance(pt)
    results.append({
        "url":                    row.get("url",""),
        "dist_nearest_school_m":  float(dists.min()),
        "schools_1km":            int(dists.lt(1000).sum()),
    })

df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT_CSV, index=False)

print(f"\n✓ Guardado: {OUTPUT_CSV}")
print(f"  Propiedades: {len(df_out)}")
print(f"  dist_nearest_school_m: media={df_out['dist_nearest_school_m'].mean():.0f}m")
print(f"  schools_1km:           media={df_out['schools_1km'].mean():.1f}")