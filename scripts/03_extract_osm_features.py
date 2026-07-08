"""
Script 07: Features de OpenStreetMap por propiedad — St. Louis City
Gratuito, sin API key. Usa osmnx para descargar la red vial y puntos de
interés (POIs) de la ciudad una sola vez, y calcula distancias para
cada propiedad.

Features generadas:
  - dist_main_road_m:   distancia a la vía principal más cercana
  - dist_park_m:        distancia al parque más cercano
  - dist_school_m:      distancia a la escuela más cercana
  - dist_transit_m:     distancia a la parada de transporte público más cercana
  - dist_commercial_m:  distancia al uso de suelo comercial más cercano
  - road_density_300m:  metros de vía dentro de un radio de 300m (proxy de conectividad)

Instalación:
    pip install osmnx geopandas shapely pandas tqdm
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from pathlib import Path
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
INPUT_CSV  = "../data/redfin_stlouis_clean.csv"
OUTPUT_CSV = "../data/stlouis_sf_osm_features.csv"
PLACE      = "St. Louis City, Missouri, USA"

Path("../data").mkdir(exist_ok=True)
ox.settings.use_cache = True
ox.settings.log_console = True

# ── 1. Cargar propiedades ──────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df_sf = df[df["property_type"] == "Single Family Residential"].copy()
df_sf = df_sf.dropna(subset=["lat", "lon"]).reset_index(drop=True)
print(f"Propiedades Single Family: {len(df_sf)}")

gdf_prop = gpd.GeoDataFrame(
    df_sf,
    geometry=[Point(lon, lat) for lon, lat in zip(df_sf["lon"], df_sf["lat"])],
    crs="EPSG:4326",
).to_crs(epsg=3857)  # proyección métrica para calcular distancias en metros

# ── 2. Descargar red vial ──────────────────────────────────────────────────────
print(f"\nDescargando red vial de {PLACE} (puede tardar unos minutos)...")
G = ox.graph_from_place(PLACE, network_type="drive")
edges = ox.graph_to_gdfs(G, nodes=False, edges=True).to_crs(epsg=3857)

# Vías principales: clasificación OSM highway = primary/secondary/trunk
main_road_types = ["primary", "secondary", "trunk", "primary_link", "secondary_link"]
edges["highway_str"] = edges["highway"].astype(str)
main_roads = edges[edges["highway_str"].str.contains("|".join(main_road_types), na=False)]
print(f"  Segmentos de vía total: {len(edges)}")
print(f"  Segmentos de vía principal: {len(main_roads)}")

# ── 3. Descargar POIs por categoría ────────────────────────────────────────────
print("\nDescargando POIs (parques, escuelas, transporte, comercio)...")

tags_park = {"leisure": "park"}
tags_school = {"amenity": ["school", "kindergarten"]}
tags_transit = {"public_transport": True, "highway": "bus_stop"}
tags_commercial = {"landuse": "commercial", "shop": True}

def safe_features_from_place(place, tags, label):
    try:
        gdf = ox.features_from_place(place, tags)
        gdf = gdf[gdf.geometry.notna()].to_crs(epsg=3857)
        # Usar centroide para polígonos (parques, zonas comerciales)
        gdf["geometry"] = gdf.geometry.centroid
        print(f"  {label}: {len(gdf)} elementos")
        return gdf
    except Exception as e:
        print(f"  {label}: error ({e}) — se omite esta capa")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")

gdf_parks      = safe_features_from_place(PLACE, tags_park, "Parques")
gdf_schools    = safe_features_from_place(PLACE, tags_school, "Escuelas")
gdf_transit    = safe_features_from_place(PLACE, tags_transit, "Transporte público")
gdf_commercial = safe_features_from_place(PLACE, tags_commercial, "Comercio")

# ── 4. Calcular distancias por propiedad ───────────────────────────────────────

def nearest_distance(point, gdf_targets):
    """Distancia en metros al elemento más cercano de gdf_targets."""
    if len(gdf_targets) == 0:
        return np.nan
    dists = gdf_targets.geometry.distance(point)
    return dists.min()

def nearest_road_distance(point, gdf_roads):
    if len(gdf_roads) == 0:
        return np.nan
    dists = gdf_roads.geometry.distance(point)
    return dists.min()

def road_density(point, gdf_roads, radius=300):
    """Metros totales de vía dentro de un buffer — proxy de conectividad."""
    buf = point.buffer(radius)
    nearby = gdf_roads[gdf_roads.intersects(buf)]
    if len(nearby) == 0:
        return 0.0
    return nearby.geometry.intersection(buf).length.sum()

print("\nCalculando distancias por propiedad...")
results = []
for idx, row in tqdm(gdf_prop.iterrows(), total=len(gdf_prop), desc="Propiedades"):
    pt = row.geometry
    results.append({
        "url":                row.get("url", f"{row.geometry.y}_{row.geometry.x}"),
        "dist_main_road_m":   nearest_road_distance(pt, main_roads),
        "dist_park_m":        nearest_distance(pt, gdf_parks),
        "dist_school_m":      nearest_distance(pt, gdf_schools),
        "dist_transit_m":     nearest_distance(pt, gdf_transit),
        "dist_commercial_m":  nearest_distance(pt, gdf_commercial),
        "road_density_300m":  road_density(pt, edges),
    })

df_osm = pd.DataFrame(results)
df_osm.to_csv(OUTPUT_CSV, index=False)

print(f"\n✓ Guardado: {OUTPUT_CSV}")
print(f"  Propiedades: {len(df_osm)}")
print(f"\nEstadísticas:")
for col in ["dist_main_road_m", "dist_park_m", "dist_school_m",
            "dist_transit_m", "dist_commercial_m", "road_density_300m"]:
    print(f"  {col:<22} media={df_osm[col].mean():.0f}  NaN={df_osm[col].isna().sum()}")
