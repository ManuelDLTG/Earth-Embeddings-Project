"""
Script 00: Combina todos los CSVs de Redfin y elimina duplicados
Pon todos los archivos descargados de Redfin en la misma carpeta que este script
y córrelo desde ahí.

Uso:
    python 00_combine_redfin_csvs.py

Output:
    data/redfin_stlouis_combined.csv  — todos los listings únicos
    data/redfin_stlouis_clean.csv     — filtrado y listo para el modelo
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path

Path("data").mkdir(exist_ok=True)

# ── 1. Cargar todos los CSVs de Redfin en la carpeta actual ───────────────────
csv_files = sorted(glob.glob("*.csv") + glob.glob("redfin_*.csv"))
if not csv_files:
    print("⚠ No se encontraron archivos CSV en esta carpeta.")
    print("  Pon los CSVs descargados de Redfin en la misma carpeta que este script.")
    exit(1)

print(f"Archivos encontrados: {len(csv_files)}")

dfs = []
for f in csv_files:
    try:
        # Redfin pone metadata en la primera fila — la saltamos
        cols = pd.read_csv(f, nrows=0).columns
        df   = pd.read_csv(f, skiprows=2, header=0)
        df.columns = cols
        dfs.append(df)
        print(f"  {f:<55} → {len(df)} filas")
    except Exception as e:
        print(f"  {f} — ERROR: {e}")

if not dfs:
    print("⚠ No se pudo leer ningún archivo.")
    exit(1)

combined = pd.concat(dfs, ignore_index=True)
print(f"\nTotal antes de deduplicar: {len(combined)}")

# Deduplicar por URL (identificador único de Redfin)
url_col = next((c for c in combined.columns if "URL" in c.upper()), None)
if url_col:
    combined = combined.drop_duplicates(subset=[url_col])
    print(f"Total ÚNICO (por URL):     {len(combined)}")
else:
    combined = combined.drop_duplicates()
    print(f"Total ÚNICO (por fila):    {len(combined)}")

combined.to_csv("data/redfin_stlouis_combined.csv", index=False)
print(f"✓ Guardado: data/redfin_stlouis_combined.csv")

# ── 2. Limpieza ───────────────────────────────────────────────────────────────
print("\nLimpiando dataset...")

df = combined.copy()

# Renombrar columnas clave
rename = {
    "SOLD DATE":     "sold_date",
    "PROPERTY TYPE": "property_type",
    "ADDRESS":       "address",
    "CITY":          "city",
    "ZIP OR POSTAL CODE": "zip",
    "PRICE":         "price",
    "BEDS":          "beds",
    "BATHS":         "baths",
    "SQUARE FEET":   "sqft",
    "LOT SIZE":      "lot_size",
    "YEAR BUILT":    "year_built",
    "DAYS ON MARKET": "days_on_market",
    "$/SQUARE FEET": "price_sqft_redfin",
    "LATITUDE":      "lat",
    "LONGITUDE":     "lon",
}
# URL — la columna tiene nombre largo
url_rename = {url_col: "url"} if url_col else {}
df = df.rename(columns={**rename, **url_rename})

# Tipos de propiedad a mantener
TIPOS_OK = ["Single Family Residential", "Townhouse"]
df = df[df["property_type"].isin(TIPOS_OK)].copy()
print(f"Tras filtro tipo (House + Townhouse): {len(df)}")

# Coordenadas
df = df.dropna(subset=["lat", "lon"])
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
df = df.dropna(subset=["lat", "lon"])

# Bounding box St. Louis City
df = df[df["lat"].between(38.53, 38.77) & df["lon"].between(-90.32, -90.17)]
print(f"Tras filtro geográfico STL City:      {len(df)}")

# Precio
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df[df["price"].between(10_000, 2_000_000)]
print(f"Tras filtro precio ($10k–$2M):        {len(df)}")

# Sqft
df["sqft"] = pd.to_numeric(df["sqft"], errors="coerce")
mediana_sqft = df.groupby("property_type")["sqft"].transform("median")
df["sqft_imputed"] = df["sqft"].isna()
df["sqft"] = df["sqft"].fillna(mediana_sqft)
df = df[df["sqft"].between(200, 8_000)]
print(f"Tras filtro sqft (200–8,000 sqft):    {len(df)}")

# Precio por sqft
df["price_per_sqft"] = df["price"] / df["sqft"]
df = df[df["price_per_sqft"].between(20, 1_000)]
print(f"Tras filtro precio/sqft ($20–$1k):    {len(df)}")

# Columnas finales
COLS = ["address", "city", "zip", "property_type", "sold_date",
        "price", "sqft", "sqft_imputed", "price_per_sqft",
        "beds", "baths", "year_built", "lot_size", "days_on_market",
        "lat", "lon", "url"]
COLS = [c for c in COLS if c in df.columns]
df   = df[COLS].copy()

# Imputar beds/baths/year_built con mediana
for col in ["beds", "baths", "year_built"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

df.to_csv("data/redfin_stlouis_clean.csv", index=False)

# ── 3. Resumen ────────────────────────────────────────────────────────────────
print(f"\n{'='*45}")
print(f"DATASET LIMPIO: {len(df)} propiedades")
print(f"{'='*45}")
print(f"  Precio mediano:     ${df['price'].median():,.0f}")
print(f"  Precio/sqft med:    ${df['price_per_sqft'].median():,.0f}/sqft")
print(f"  Sqft mediano:       {df['sqft'].median():.0f} sqft")
print(f"  Año construido med: {df['year_built'].median():.0f}")
print(f"  Tipos:")
print(df["property_type"].value_counts().to_string())
print(f"\nNulls relevantes:")
nulls = df[["price","sqft","beds","baths","year_built"]].isna().sum()
print(nulls[nulls > 0].to_string() if nulls.any() else "  Ninguno ✓")
print(f"\n✓ Guardado: data/redfin_stlouis_clean.csv")
