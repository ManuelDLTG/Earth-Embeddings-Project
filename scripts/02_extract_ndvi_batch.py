"""
Script 06b: Extracción NDVI en BATCH — GEE reduceRegions
Procesa en grupos de 500 propiedades por llamada (evita timeout).
Para 4,258 propiedades: ~9 batches, ~3-5 minutos total.

Instalación:
    pip install earthengine-api pandas tqdm
"""

import ee
import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
INPUT_CSV   = "../data/redfin_stlouis_clean.csv"
OUTPUT_CSV  = "../data/stlouis_sf_ndvi.csv"
CHECKPOINT  = "../data/ndvi_batch_checkpoint.json"
GEE_PROJECT = "earth-embeddings-project"
YEAR        = 2024
CLOUD_PCT   = 30
BUFFERS_M   = [50, 100, 200]
BATCH_SIZE  = 500

Path("../data").mkdir(exist_ok=True)

# ── autenticación ─────────────────────────────────────────────────────────────
ee.Authenticate()
ee.Initialize(project=GEE_PROJECT)
print("GEE inicializado ✓")

# ── construir imagen NDVI anual ───────────────────────────────────────────────
def mask_clouds(image):
    return image.updateMask(image.select("cs").gte(0.6))

s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31")
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_PCT)))
cs_plus   = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
ndvi_img  = (s2.linkCollection(cs_plus, ["cs"])
               .map(mask_clouds)
               .median()
               .normalizedDifference(["B8", "B4"])
               .rename("NDVI"))
print("Imagen NDVI construida ✓")

# ── cargar propiedades ────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df_sf = (df[df["property_type"] == "Single Family Residential"]
         .dropna(subset=["lat","lon"])
         .reset_index(drop=True))
print(f"Propiedades a procesar: {len(df_sf)}")

# ── checkpoint ────────────────────────────────────────────────────────────────
resultados = {}
if Path(CHECKPOINT).exists():
    with open(CHECKPOINT) as f:
        resultados = json.load(f)
    print(f"Checkpoint: {len(resultados)} propiedades ya procesadas")

# ── función de extracción por batch ──────────────────────────────────────────
def extract_batch(batch_df, buffer_m):
    """Extrae NDVI para un grupo de propiedades con un buffer dado."""
    features = []
    for _, row in batch_df.iterrows():
        key  = str(row.get("url", f"{row['lat']}_{row['lon']}"))
        geom = ee.Geometry.Point([row["lon"], row["lat"]]).buffer(buffer_m)
        features.append(ee.Feature(geom, {"prop_id": key}))

    fc      = ee.FeatureCollection(features)
    sampled = ndvi_img.reduceRegions(
        collection = fc,
        reducer    = ee.Reducer.mean(),
        scale      = 10,
    )
    data = sampled.getInfo()

    batch_result = {}
    for feat in data["features"]:
        props = feat["properties"]
        batch_result[props["prop_id"]] = props.get("mean", np.nan)
    return batch_result

# ── procesar por buffer y batch ───────────────────────────────────────────────
# Estructura del checkpoint: {prop_id: {ndvi_50m: x, ndvi_100m: x, ndvi_200m: x}}
pendientes = [
    row for _, row in df_sf.iterrows()
    if str(row.get("url", f"{row['lat']}_{row['lon']}")) not in resultados
]
print(f"Pendientes: {len(pendientes)}")

if pendientes:
    batches = [pendientes[i:i+BATCH_SIZE] for i in range(0, len(pendientes), BATCH_SIZE)]
    print(f"Batches de {BATCH_SIZE}: {len(batches)}")

    for buf in BUFFERS_M:
        print(f"\n--- Buffer {buf}m ---")
        for i, batch in enumerate(tqdm(batches, desc=f"NDVI {buf}m")):
            batch_df = pd.DataFrame(batch)
            try:
                batch_result = extract_batch(batch_df, buf)
                for key, val in batch_result.items():
                    if key not in resultados:
                        resultados[key] = {}
                    resultados[key][f"ndvi_{buf}m"] = val
            except Exception as e:
                print(f"\n  ✗ Batch {i+1} error: {e}")

        # Checkpoint después de cada buffer completo
        with open(CHECKPOINT, "w") as f:
            json.dump(resultados, f)
        print(f"  Checkpoint guardado ✓")

# ── guardar resultado final ───────────────────────────────────────────────────
rows = []
for _, row in df_sf.iterrows():
    key  = str(row.get("url", f"{row['lat']}_{row['lon']}"))
    vals = resultados.get(key, {})
    rows.append({
        "url":     key,
        "address": row["address"],
        "lat":     row["lat"],
        "lon":     row["lon"],
        **{f"ndvi_{b}m": vals.get(f"ndvi_{b}m", np.nan) for b in BUFFERS_M},
    })

df_out = pd.DataFrame(rows)
df_out.to_csv(OUTPUT_CSV, index=False)

print(f"\n✓ Guardado: {OUTPUT_CSV}")
print(f"  Propiedades: {len(df_out)}")
print(f"\nEstadísticas NDVI:")
for buf in BUFFERS_M:
    col = f"ndvi_{buf}m"
    print(f"  {col}: media={df_out[col].mean():.3f}  NaN={df_out[col].isna().sum()}")

if Path(CHECKPOINT).exists():
    Path(CHECKPOINT).unlink()
    print("Checkpoint eliminado ✓")