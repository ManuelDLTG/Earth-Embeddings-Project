"""
Script: Extracción de embeddings AlphaEarth para propiedades Single Family en St. Louis City
Método: buffer de 50m alrededor del punto lat/lon de cada propiedad → media de píxeles
Fuente: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL — año 2024

Instrucciones:
    pip install earthengine-api pandas tqdm
    python 01_extract_embeddings_stlouis.py
"""

import ee
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
INPUT_CSV     = "../data/redfin_stlouis_clean.csv"
OUTPUT_CSV    = "../data/stlouis_sf_embeddings_200.csv"
CHECKPOINT    = "../data/stlouis_sf_checkpoint.json"
GEE_PROJECT   = "earth-embeddings-project"
GSED_YEAR     = 2024
BUFFER_M      = 200      # buffer en metros alrededor de cada propiedad
EMBED_DIM     = 64
PAUSE_EVERY   = 50      # pausa breve cada N propiedades
PAUSE_SECS    = 1.5

Path("data").mkdir(exist_ok=True)

# ── autenticación GEE ─────────────────────────────────────────────────────────
ee.Authenticate()
ee.Initialize(project=GEE_PROJECT)
print("GEE inicializado ✓")

# ── cargar dataset GSED ───────────────────────────────────────────────────────
gsed = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{GSED_YEAR}-01-01", f"{GSED_YEAR}-12-31")
        .mosaic())

band_names = gsed.bandNames().getInfo()
assert len(band_names) == EMBED_DIM, f"Esperaba {EMBED_DIM} bandas, encontró {len(band_names)}"
print(f"Bandas GSED: {len(band_names)} ✓")

# ── cargar propiedades ────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df_sf = df[df['property_type'] == 'Single Family Residential'].copy()
df_sf = df_sf.dropna(subset=['lat', 'lon']).reset_index(drop=True)
print(f"Propiedades Single Family: {len(df_sf)}")

# ── checkpoint ────────────────────────────────────────────────────────────────
resultados = {}
if Path(CHECKPOINT).exists():
    with open(CHECKPOINT) as f:
        raw = json.load(f)
    resultados = {k: np.array(v) for k, v in raw.items()}
    print(f"Checkpoint cargado: {len(resultados)} propiedades ya procesadas")

# ── extracción ────────────────────────────────────────────────────────────────

def extract_embedding(row):
    """
    Extrae el embedding promedio dentro de un buffer de BUFFER_M metros
    alrededor del punto lat/lon de la propiedad.
    """
    key = row['url'] if pd.notna(row.get('url')) else f"{row['lat']}_{row['lon']}"
    try:
        point  = ee.Geometry.Point([row['lon'], row['lat']])
        buffer = point.buffer(BUFFER_M)

        mean_dict = gsed.reduceRegion(
            reducer    = ee.Reducer.mean(),
            geometry   = buffer,
            scale      = 10,
            maxPixels  = 1e6,
            bestEffort = True,
        ).getInfo()

        emb = np.array([mean_dict.get(b, np.nan) for b in band_names])
        return key, emb

    except Exception as e:
        print(f"\n  ✗ Error en {row.get('address', key)}: {e}")
        return key, None


print(f"\nExtrayendo embeddings para {len(df_sf)} propiedades...")
pendientes = [
    row for _, row in df_sf.iterrows()
    if (row['url'] if pd.notna(row.get('url')) else f"{row['lat']}_{row['lon']}") not in resultados
]
print(f"Pendientes: {len(pendientes)}")

for i, row in enumerate(tqdm(pendientes, desc="Propiedades")):
    key, emb = extract_embedding(row)
    if emb is not None:
        resultados[key] = emb

    # Checkpoint cada 100
    if (i + 1) % 100 == 0:
        with open(CHECKPOINT, "w") as f:
            json.dump({k: v.tolist() for k, v in resultados.items()}, f)

    # Pausa para no saturar GEE
    if (i + 1) % PAUSE_EVERY == 0:
        time.sleep(PAUSE_SECS)

# ── guardar ───────────────────────────────────────────────────────────────────

# Construir DataFrame con clave + 64 dims de embedding
rows_out = []
for _, row in df_sf.iterrows():
    key = row['url'] if pd.notna(row.get('url')) else f"{row['lat']}_{row['lon']}"
    emb = resultados.get(key)
    row_dict = {
        'url':     key,
        'address': row['address'],
        'lat':     row['lat'],
        'lon':     row['lon'],
    }
    for j in range(EMBED_DIM):
        row_dict[f'emb_{j:02d}'] = emb[j] if emb is not None else np.nan
    rows_out.append(row_dict)

df_emb = pd.DataFrame(rows_out)
df_emb.to_csv(OUTPUT_CSV, index=False)

n_ok  = df_emb['emb_00'].notna().sum()
n_nan = df_emb['emb_00'].isna().sum()

print(f"\n✓ Guardado: {OUTPUT_CSV}")
print(f"  Propiedades con embedding completo: {n_ok}")
print(f"  Propiedades con error (NaN):         {n_nan}")
print(f"\nEjemplo (primeras 3 filas, primeras 5 dims):")
print(df_emb[['address'] + [f'emb_{j:02d}' for j in range(5)]].head(3).to_string())

# Limpiar checkpoint
if Path(CHECKPOINT).exists() and n_nan == 0:
    Path(CHECKPOINT).unlink()
    print("Checkpoint eliminado ✓")
