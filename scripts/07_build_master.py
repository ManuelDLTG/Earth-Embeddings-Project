"""
Script 07: Construcción del dataset maestro completo
Combina todas las capas de datos en un solo CSV listo para el modelo.

Capas:
  1. Redfin (precio, características físicas)
  2. AlphaEarth embeddings (64 dims, 200m buffer)
  3. NDVI Sentinel-2 (3 escalas)
  4. OSM features (6 distancias urbanas)
  5. Crimen SLMPD (2 radios)
  6. Escuelas NCES (distancia + densidad)
  7. Walk Score / Transit Score / Bike Score
  8. ACS 2023 block group (26 variables censales)

Output:
    ../data/stlouis_master_full.csv

Instalación:
    pip install pandas geopandas shapely requests pygris getpass
"""

import pandas as pd
import numpy as np
import requests
import geopandas as gpd
from shapely.geometry import Point
import getpass
import warnings
warnings.filterwarnings("ignore")

try:
    import pygris
except ImportError:
    import subprocess; subprocess.run(["pip","install","pygris","-q"]); import pygris

# ── rutas ─────────────────────────────────────────────────────────────────────
DATA = "../data"
FILES = {
    "redfin":    f"{DATA}/redfin_stlouis_clean.csv",
    "emb":       f"{DATA}/stlouis_sf_embeddings_200.csv",
    "ndvi":      f"{DATA}/stlouis_sf_ndvi.csv",
    "osm":       f"{DATA}/stlouis_sf_osm_features.csv",
    "crime":     f"{DATA}/stlouis_sf_crime.csv",
    "schools":   f"{DATA}/stlouis_sf_schools.csv",
    "walkscore": f"{DATA}/stlouis_sf_walkscore.csv",
}

# ── 1. Cargar archivos disponibles ────────────────────────────────────────────
import os
print("Cargando archivos de datos...")
loaded = {}
for name, path in FILES.items():
    if os.path.exists(path):
        loaded[name] = pd.read_csv(path)
        print(f"  ✓ {name:<12} {len(loaded[name])} filas  {path.split('/')[-1]}")
    else:
        print(f"  ✗ {name:<12} no encontrado — se omite esta capa")

# ── 2. Base: Single Family de Redfin ─────────────────────────────────────────
df = loaded["redfin"]
df = df[df["property_type"]=="Single Family Residential"].dropna(
    subset=["lat","lon"]).reset_index(drop=True)
print(f"\nBase Single Family: {len(df)} propiedades")

# ── 3. Merge por lat/lon (embeddings y NDVI) ──────────────────────────────────
for name in ["emb","ndvi"]:
    if name not in loaded:
        continue
    src = loaded[name].copy()
    feat_cols = [c for c in src.columns if c not in
                 ["url","address","lat","lon","lat_r","lon_r"]]
    src["lat_r"] = src["lat"].round(6)
    src["lon_r"] = src["lon"].round(6)
    df["lat_r"]  = df["lat"].round(6)
    df["lon_r"]  = df["lon"].round(6)
    df = df.merge(src[["lat_r","lon_r"]+feat_cols],
                  on=["lat_r","lon_r"], how="left")
    df = df.drop(columns=["lat_r","lon_r"])
    print(f"  Merge {name}: {df.shape[1]} columnas")

# ── 4. Merge por URL (OSM, crimen, escuelas, walkscore) ──────────────────────
for name in ["osm","crime","schools","walkscore"]:
    if name not in loaded:
        continue
    src = loaded[name].copy()
    feat_cols = [c for c in src.columns if c != "url"]
    df = df.merge(src[["url"]+feat_cols], on="url", how="left")
    print(f"  Merge {name}: {df.shape[1]} columnas")

df = df.drop_duplicates(subset=["url"])
print(f"\nTras todos los merges: {len(df)} propiedades × {df.shape[1]} columnas")

# ── 5. ACS via Census API ─────────────────────────────────────────────────────
CENSUS_KEY = getpass.getpass("\nCensus API Key: ")

VARIABLES = {
    "B01003_001E":"pop_total","B01002_001E":"median_age",
    "B19013_001E":"median_hh_income","B15003_022E":"pop_bachelors",
    "B15003_023E":"pop_masters","B15003_025E":"pop_doctorate",
    "B15003_002E":"pop_no_schooling","B23025_002E":"pop_labor_force",
    "B23025_005E":"pop_unemployed","B25001_001E":"housing_units_total",
    "B25002_002E":"housing_units_occupied","B25003_002E":"owner_occupied",
    "B25003_003E":"renter_occupied","B25064_001E":"median_gross_rent",
    "B25077_001E":"median_home_value","B25035_001E":"median_year_built",
    "B08301_001E":"commuters_total","B08301_003E":"commute_car",
    "B08301_010E":"commute_transit","B02001_002E":"pop_white",
    "B02001_003E":"pop_black","B02001_005E":"pop_asian",
}

print("Descargando ACS 2023...")
acs_url = (f"https://api.census.gov/data/2023/acs/acs5"
           f"?get=NAME,{','.join(VARIABLES.keys())}"
           f"&for=block%20group:*&in=state:29%20county:510"
           f"&key={CENSUS_KEY}")
data = requests.get(acs_url, timeout=30).json()
df_acs = pd.DataFrame(data[1:], columns=data[0]).rename(columns=VARIABLES)
df_acs["GEOID"] = (df_acs["state"].str.zfill(2) + df_acs["county"].str.zfill(3) +
                   df_acs["tract"].str.zfill(6) + df_acs["block group"].str.zfill(1))
for col in VARIABLES.values():
    df_acs[col] = pd.to_numeric(df_acs[col], errors="coerce")
    df_acs[col] = df_acs[col].where(df_acs[col] > -1, other=pd.NA)
df_acs["pct_owner_occupied"] = df_acs["owner_occupied"]/df_acs["housing_units_occupied"]*100
df_acs["pct_college_plus"]   = (df_acs["pop_bachelors"]+df_acs["pop_masters"]+
                                 df_acs["pop_doctorate"])/df_acs["pop_total"]*100
df_acs["unemployment_rate"]  = df_acs["pop_unemployed"]/df_acs["pop_labor_force"]*100
df_acs["pct_renter"]         = df_acs["renter_occupied"]/df_acs["housing_units_occupied"]*100
print(f"  ACS: {len(df_acs)} block groups")

# Spatial join → block group
print("Descargando shapefile TIGER...")
gdf_bg = pygris.block_groups(state="MO", county="510", year=2023).to_crs(epsg=4326)
gdf_prop = gpd.GeoDataFrame(df,
    geometry=[Point(lon,lat) for lon,lat in zip(df["lon"],df["lat"])],
    crs="EPSG:4326")
joined = gpd.sjoin(gdf_prop, gdf_bg[["GEOID","geometry"]],
                   how="left", predicate="within")
joined = joined.drop(columns=["geometry","index_right"], errors="ignore")

ACS_KEEP = list(VARIABLES.values()) + ["pct_owner_occupied","pct_college_plus",
                                        "unemployment_rate","pct_renter","GEOID"]
master = joined.merge(df_acs[ACS_KEEP], on="GEOID", how="left")

# Imputar ACS con mediana global
acs_num = [c for c in ACS_KEEP if c != "GEOID"]
for col in acs_num:
    master[col] = master[col].fillna(master[col].dropna().median())

# ── 6. Imputar Walk Score con mediana si faltan ───────────────────────────────
for col in ["walk_score","transit_score","bike_score"]:
    if col in master.columns:
        master[col] = master[col].fillna(master[col].dropna().median())

# ── 7. Resumen final ──────────────────────────────────────────────────────────
emb_cols  = [c for c in master.columns if c.startswith("emb_")]
ndvi_cols = [c for c in master.columns if c.startswith("ndvi_")]
osm_cols  = [c for c in ["dist_main_road_m","dist_park_m","dist_school_m",
             "dist_transit_m","dist_commercial_m","road_density_300m"]
             if c in master.columns]
crime_cols   = [c for c in ["crimes_250m","crimes_500m"] if c in master.columns]
school_cols  = [c for c in ["dist_nearest_school_m","schools_1km"] if c in master.columns]
ws_cols      = [c for c in ["walk_score","transit_score","bike_score"] if c in master.columns]

print(f"\n{'='*55}")
print(f"DATASET MAESTRO FINAL")
print(f"{'='*55}")
print(f"  Propiedades:      {len(master)}")
print(f"  Columnas total:   {master.shape[1]}")
print(f"  ── Features ──")
print(f"  Propiedad:        {len(['sqft','beds','baths','year_built','lot_size'])} vars")
print(f"  Embeddings:       {len(emb_cols)} dims")
print(f"  NDVI:             {len(ndvi_cols)} escalas")
print(f"  OSM:              {len(osm_cols)} features")
print(f"  Crimen:           {len(crime_cols)} features")
print(f"  Escuelas:         {len(school_cols)} features")
print(f"  Walk/Transit:     {len(ws_cols)} scores")
print(f"  ACS census:       {len(acs_num)} vars")
print(f"  ── Target ──")
print(f"  price/sqft med:   ${master['price_per_sqft'].median():.0f}")
print(f"  NaN total feats:  {master[emb_cols+ndvi_cols].isna().sum().sum()}")

master.to_csv(f"{DATA}/stlouis_master_full.csv", index=False)
print(f"\n✓ {DATA}/stlouis_master_full.csv")