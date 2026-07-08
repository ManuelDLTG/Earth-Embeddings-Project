"""
Script 04: Features de crimen por distrito policial — SLMPD CompStat 2026
Fuente: CompStat semanal SLMPD (datos YTD al 28 junio 2026)
Método: asignar cada propiedad a su distrito policial (1-6) por coordenadas,
        luego unir con tasas de crimen del CompStat.

Features generadas:
  - police_district:    distrito policial (1-6)
  - violent_crime_ytd: crímenes violentos YTD del distrito
  - total_crime_ytd:   total de crímenes YTD del distrito
  - violent_crime_rate: crímenes violentos por 1,000 hab

Instalación:
    pip install pandas numpy
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_CSV  = "../data/redfin_stlouis_clean.csv"
OUTPUT_CSV = "../data/stlouis_sf_crime.csv"

Path("../data").mkdir(exist_ok=True)

# ── Datos CompStat YTD al 28-junio-2026 ───────────────────────────────────────
COMPSTAT = {
    1: {"violent_crime_ytd": 197, "total_crime_ytd": 781,  "patrol": "South"},
    2: {"violent_crime_ytd": 118, "total_crime_ytd": 726,  "patrol": "South"},
    3: {"violent_crime_ytd": 216, "total_crime_ytd": 761,  "patrol": "Central"},
    4: {"violent_crime_ytd": 349, "total_crime_ytd": 1016, "patrol": "Central"},
    5: {"violent_crime_ytd": 287, "total_crime_ytd": 878,  "patrol": "North"},
    6: {"violent_crime_ytd": 310, "total_crime_ytd": 717,  "patrol": "North"},
}
DISTRICT_POP = {1: 52000, 2: 44000, 3: 55000, 4: 58000, 5: 49000, 6: 46000}

df_compstat = pd.DataFrame([
    {
        "police_district":    d,
        "patrol":             v["patrol"],
        "violent_crime_ytd":  v["violent_crime_ytd"],
        "total_crime_ytd":    v["total_crime_ytd"],
        "violent_crime_rate": v["violent_crime_ytd"] / DISTRICT_POP[d] * 1000,
        "total_crime_rate":   v["total_crime_ytd"]   / DISTRICT_POP[d] * 1000,
    }
    for d, v in COMPSTAT.items()
])

print("Datos CompStat por distrito:")
print(df_compstat.to_string(index=False))

# ── Asignación de distrito por coordenadas ────────────────────────────────────
# Límites aproximados basados en los mapas de patrulla del CompStat:
# North Patrol (5,6): lat > 38.68
# Central Patrol (3,4): 38.60 < lat < 38.68
# South Patrol (1,2): lat < 38.60
# Dentro de cada patrulla, el distrito oeste (impar) tiene lon < -90.23

def assign_district(lat, lon):
    if lat > 38.68:
        return 5 if lon < -90.23 else 6
    elif lat > 38.60:
        return 3 if lon < -90.23 else 4
    else:
        return 1 if lon < -90.23 else 2

# ── Cargar propiedades y asignar distrito ─────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df_sf = df[df["property_type"]=="Single Family Residential"].dropna(
    subset=["lat","lon"]).reset_index(drop=True)
print(f"\nPropiedades SF: {len(df_sf)}")

df_sf["police_district"] = df_sf.apply(
    lambda r: assign_district(r["lat"], r["lon"]), axis=1)

print(f"Distribución por distrito:")
print(df_sf["police_district"].value_counts().sort_index().to_string())

# ── Join con CompStat ─────────────────────────────────────────────────────────
result = df_sf[["url","police_district"]].merge(
    df_compstat, on="police_district", how="left"
)

result.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Guardado: {OUTPUT_CSV}")
print(f"  Propiedades: {len(result)}")
print(f"\nEstadísticas:")
print(f"  violent_crime_rate: {result['violent_crime_rate'].describe()[['mean','min','max']].to_string()}")