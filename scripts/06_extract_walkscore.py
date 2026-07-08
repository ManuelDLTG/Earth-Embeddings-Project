"""
Script 06: Walk Score y Transit Score por propiedad
Fuente: Walk Score API (gratuita para investigación, requiere API key)
Registro gratuito en: https://www.walkscore.com/professional/api.php

Features generadas:
  - walk_score:    accesibilidad a pie (0-100)
  - transit_score: accesibilidad en transporte público (0-100)
  - bike_score:    accesibilidad en bicicleta (0-100)

Instalación:
    pip install requests pandas tqdm

Uso:
    1. Regístrate en https://www.walkscore.com/professional/api.php
    2. Copia tu API key abajo o déjala vacía para pedirla al correr
    3. python 06_extract_walkscore.py
"""

import requests
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from tqdm import tqdm

INPUT_CSV   = "../data/redfin_stlouis_clean.csv"
OUTPUT_CSV  = "../data/stlouis_sf_walkscore.csv"
CHECKPOINT  = "../data/walkscore_checkpoint.json"
WALKSCORE_KEY = ""  # ← pon tu API key aquí
PAUSE_SECS  = 0.5   # Walk Score permite ~5,000 requests/día en plan gratuito

Path("../data").mkdir(exist_ok=True)

if not WALKSCORE_KEY:
    WALKSCORE_KEY = input("Walk Score API Key: ").strip()

# ── cargar propiedades ────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df_sf = df[df["property_type"]=="Single Family Residential"].dropna(
    subset=["lat","lon","address"]
).reset_index(drop=True)
print(f"Propiedades a procesar: {len(df_sf)}")

# ── checkpoint ────────────────────────────────────────────────────────────────
resultados = {}
if Path(CHECKPOINT).exists():
    with open(CHECKPOINT) as f:
        resultados = json.load(f)
    print(f"Checkpoint: {len(resultados)} propiedades ya procesadas")

# ── función de extracción ─────────────────────────────────────────────────────
def get_walkscore(address, lat, lon):
    """
    Consulta Walk Score API para una dirección dada.
    Devuelve Walk Score, Transit Score y Bike Score (0-100 cada uno).
    """
    url = "https://api.walkscore.com/score"
    params = {
        "format":  "json",
        "address": address,
        "lat":     lat,
        "lon":     lon,
        "transit": 1,
        "bike":    1,
        "wsapikey": WALKSCORE_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        # Walk Score status codes:
        # 1 = score available, 2 = score unavailable, 30-40 = errors
        if data.get("status") not in [1, 2]:
            return None

        return {
            "walk_score":    data.get("walkscore", np.nan),
            "walk_desc":     data.get("description", ""),
            "transit_score": data.get("transit", {}).get("score", np.nan)
                             if data.get("transit") else np.nan,
            "transit_desc":  data.get("transit", {}).get("description", "")
                             if data.get("transit") else "",
            "bike_score":    data.get("bike", {}).get("score", np.nan)
                             if data.get("bike") else np.nan,
        }
    except Exception:
        return None

# ── extracción ────────────────────────────────────────────────────────────────
pendientes = [
    row for _, row in df_sf.iterrows()
    if str(row.get("url","")) not in resultados
]
print(f"Pendientes: {len(pendientes)}")

for i, row in enumerate(tqdm(pendientes, desc="Walk Score")):
    key     = str(row.get("url",""))
    address = f"{row['address']}, St. Louis, MO"
    vals    = get_walkscore(address, row["lat"], row["lon"])

    if vals:
        resultados[key] = vals

    if (i+1) % 100 == 0:
        with open(CHECKPOINT,"w") as f:
            json.dump(resultados, f)

    time.sleep(PAUSE_SECS)

# ── guardar ───────────────────────────────────────────────────────────────────
rows = []
for _, row in df_sf.iterrows():
    key  = str(row.get("url",""))
    vals = resultados.get(key, {})
    rows.append({
        "url":           key,
        "walk_score":    vals.get("walk_score", np.nan),
        "transit_score": vals.get("transit_score", np.nan),
        "bike_score":    vals.get("bike_score", np.nan),
    })

df_out = pd.DataFrame(rows)
df_out.to_csv(OUTPUT_CSV, index=False)

if Path(CHECKPOINT).exists():
    Path(CHECKPOINT).unlink()

print(f"\n✓ Guardado: {OUTPUT_CSV}")
print(f"  Propiedades:          {len(df_out)}")
print(f"  Con Walk Score:       {df_out['walk_score'].notna().sum()}")
print(f"  Con Transit Score:    {df_out['transit_score'].notna().sum()}")
print(f"\nEstadísticas:")
for col in ["walk_score","transit_score","bike_score"]:
    print(f"  {col}: media={df_out[col].mean():.1f}  NaN={df_out[col].isna().sum()}")
