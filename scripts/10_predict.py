"""
Script 10: MVP — Predicción de precio con solo lat/lon (Arquitectura G-Fusion)
El usuario solo ingresa coordenadas GPS. El sistema obtiene automáticamente:
  1. Características de la propiedad y del entorno urbano → KNN espacial sobre dataset maestro
  2. Variables de Crimen dinámicas → Asignación por cuadrante SLMPD CompStat 2026
  3. Earth Embedding → AlphaEarth via GEE (radio 200m)
  4. Variables censales → Census Geocoder + ACS API
  5. Inferencia Multimodal → Etapa 1 (Ridge Satelital) + Etapa 2 (LightGBM Completo)

Instalación:
    pip install earthengine-api joblib pandas numpy requests scikit-learn lightgbm
"""

import ee
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree
import warnings
warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "../data/best_model_C.pkl"  # Contiene el modelo G-Fusion empaquetado
DATASET_PATH = "../data/stlouis_master_full.csv"
GEE_PROJECT  = "earth-embeddings-project"
CENSUS_KEY   = ""     # ← Pon tu Census API key aquí, o déjala vacía para interactivo
GSED_YEAR    = 2024
BUFFER_M     = 200
KNN_K        = 5
KNN_MAX_M    = 500    # distancia máxima en metros para vecinos válidos

# ── datos estáticos de crimen (CompStat 2026 YTD al 28 de junio) ──────────────
COMPSTAT = {
    1: {"violent_crime_ytd": 197, "total_crime_ytd": 781, "violent_crime_rate": 3.788, "total_crime_rate": 15.019},
    2: {"violent_crime_ytd": 118, "total_crime_ytd": 726, "violent_crime_rate": 2.681, "total_crime_rate": 16.500},
    3: {"violent_crime_ytd": 216, "total_crime_ytd": 761, "violent_crime_rate": 3.927, "total_crime_rate": 13.836},
    4: {"violent_crime_ytd": 349, "total_crime_ytd": 1016, "violent_crime_rate": 6.017, "total_crime_rate": 17.517},
    5: {"violent_crime_ytd": 287, "total_crime_ytd": 878, "violent_crime_rate": 5.857, "total_crime_rate": 17.918},
    6: {"violent_crime_ytd": 310, "total_crime_ytd": 717, "violent_crime_rate": 6.739, "total_crime_rate": 15.586},
}

# ── cargar modelo y dataset ───────────────────────────────────────────────────
print("Cargando modelo G-Fusion y dataset maestro de referencia...")
pkg       = joblib.load(MODEL_PATH)
model     = pkg["model"]
FEATS     = pkg["feature_cols"]
MEDIANS   = pkg["medians"]
EMB_COLS  = pkg["emb_cols"]
ACS_COLS  = pkg["acs_cols"]
PROP_COLS = pkg["prop_cols"]

df_train = pd.read_csv(DATASET_PATH)
df_train = df_train.dropna(subset=["lat","lon"]).reset_index(drop=True)

# Construir BallTree para KNN espacial (coordenadas en radianes)
coords_rad = np.radians(df_train[["lat","lon"]].values)
tree       = BallTree(coords_rad, metric="haversine")
EARTH_R    = 6371000  # metros

print(f"Modelo G-Fusion cargado ✓ (Ensamble de 2 Etapas listo)")
print(f"Dataset de referencia: {len(df_train)} propiedades")

# ── autenticación GEE ─────────────────────────────────────────────────────────
ee.Authenticate()
ee.Initialize(project=GEE_PROJECT)
gsed       = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
              .filterDate(f"{GSED_YEAR}-01-01", f"{GSED_YEAR}-12-31")
              .mosaic())
band_names = gsed.bandNames().getInfo()
print(f"AlphaEarth GSED cargado ✓ ({len(band_names)} dims)\n")

if not CENSUS_KEY:
    import getpass
    CENSUS_KEY = getpass.getpass("Census API Key: ")

# ── funciones de extracción e imputación ──────────────────────────────────────

def assign_district(lat, lon):
    """Reglas geográficas del Script 04 para asignar distrito policial."""
    if lat > 38.68:
        return 5 if lon < -90.23 else 6
    elif lat > 38.60:
        return 3 if lon < -90.23 else 4
    else:
        return 1 if lon < -90.23 else 2


def get_spatial_context(lat, lon, k=KNN_K, max_m=KNN_MAX_M):
    """
    Usa el BallTree espacial para extraer las características físicas del inmueble
    así como las variables de infraestructura urbana (OSM) y cobertura vegetal (NDVI).
    """
    point_rad = np.radians([[lat, lon]])
    dists, idxs = tree.query(point_rad, k=k)
    dists_m = dists[0] * EARTH_R

    mask = dists_m <= max_m
    valid_idxs  = idxs[0][mask]
    valid_dists = dists_m[mask]

    if len(valid_idxs) == 0:
        return None, dists_m[0]

    weights = 1 / (valid_dists + 1e-6)
    weights /= weights.sum()

    neighbors = df_train.iloc[valid_idxs]
    context_data = {}
    
    # Todas las columnas que dependen de la ubicación física e inmediata de la casa
    COLS_A_IMPUTAR = PROP_COLS + [
        "dist_main_road_m", "dist_park_m", "dist_transit_m", "dist_commercial_m", 
        "road_density_300m", "dist_nearest_school_m", "schools_1km",
        "walk_score", "transit_score", "bike_score", "ndvi_50m", "ndvi_100m", "ndvi_200m"
    ]
    
    for col in COLS_A_IMPUTAR:
        if col in neighbors.columns:
            vals = neighbors[col].fillna(MEDIANS.get(col, 0))
            context_data[col] = float(np.average(vals.values, weights=weights))

    return context_data, valid_dists


def get_embedding(lat, lon):
    """Extrae embedding AlphaEarth en (lat, lon) con buffer de BUFFER_M metros."""
    point  = ee.Geometry.Point([lon, lat])
    buffer = point.buffer(BUFFER_M)
    result = gsed.reduceRegion(
        reducer    = ee.Reducer.mean(),
        geometry   = buffer,
        scale      = 10,
        maxPixels  = 1e6,
        bestEffort = True,
    ).getInfo()
    return np.array([result.get(b, np.nan) for b in band_names])


def get_acs(lat, lon):
    """Descarga variables ACS del block group que contiene (lat, lon)."""
    geo_url = (f"https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
               f"?x={lon}&y={lat}&benchmark=Public_AR_Current"
               f"&vintage=Current_Current&format=json")
    try:
        data  = requests.get(geo_url, timeout=15).json()
        geoid_info = data["result"]["geographies"]["Census Block Groups"]
        if not geoid_info:
            return None, None
        geoid  = geoid_info[0]["GEOID"]
        state  = geoid[:2]; county = geoid[2:5]
        tract  = geoid[5:11]; bg = geoid[11:12]
    except:
        return None, None

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
    acs_url = (f"https://api.census.gov/data/2023/acs/acs5"
               f"?get={','.join(VARIABLES.keys())}"
               f"&for=block%20group:{bg}"
               f"&in=state:{state}%20county:{county}%20tract:{tract}"
               f"&key={CENSUS_KEY}")
    try:
        data = requests.get(acs_url, timeout=15).json()
        row  = dict(zip(data[0], data[1]))
        acs  = {v: float(row.get(k, np.nan)) for k, v in VARIABLES.items()}
        occ  = max(acs.get("housing_units_occupied", 1), 1)
        tot  = max(acs.get("pop_total", 1), 1)
        lf   = max(acs.get("pop_labor_force", 1), 1)
        acs["pct_owner_occupied"] = acs.get("owner_occupied", 0) / occ * 100
        acs["pct_college_plus"]   = (acs.get("pop_bachelors",0) +
                                     acs.get("pop_masters",0) +
                                     acs.get("pop_doctorate",0)) / tot * 100
        acs["unemployment_rate"]  = acs.get("pop_unemployed", 0) / lf * 100
        acs["pct_renter"]         = acs.get("renter_occupied", 0) / occ * 100
        acs = {k: (v if v > -1 else np.nan) for k, v in acs.items()}
        return acs, geoid
    except:
        return None, None

# ── pipeline de inferencia ────────────────────────────────────────────────────

def predict(lat, lon):
    """Pipeline de predicción interactiva basado en la arquitectura Late Fusion."""
    print(f"\n{'═'*65}")
    print(f"  PREDICCIÓN MVP G-FUSION PARA COORDENADAS: ({lat:.6f}, {lon:.6f})")
    print(f"{'═'*65}")

    # 1. KNN — Extraer contexto estructural e infraestructura urbana
    print("  [1/4] Ejecutando KNN Espacial para características locales...", end="", flush=True)
    local_data, dists = get_spatial_context(lat, lon)

    if local_data is None:
        min_dist = dists if isinstance(dists, float) else dists[0]
        print(f"\n  ⚠ Fuera del radio de entrenamiento ({min_dist:.0f}m). Usando medianas.")
        local_data = {}
        confianza = "baja"
    else:
        min_d = min(dists) if hasattr(dists, '__len__') else dists
        print(f" ✓  {len(dists)} inmuebles cercanos mapeados.")
        confianza = "alta" if min_d < 200 else "media"

    # 2. Módulo de Crimen Estático (CompStat 2026)
    distrito = assign_district(lat, lon)
    crimen_loc = COMPSTAT[distrito]
    print(f"  [2/4] Mapeando seguridad: Distrito Policial YTD {distrito} ✓")

    # 3. Descarga de Embeddings desde Google Earth Engine
    print("  [3/4] Extrayendo Capa Satelital AlphaEarth (GEE)...", end="", flush=True)
    emb = get_embedding(lat, lon)
    n_valid = int(np.sum(~np.isnan(emb)))
    print(f" ✓  ({n_valid}/64 dimensiones procesadas)")

    # 4. API del Censo ACS
    print("  [4/4] Consultando Datos Socioeconómicos Census ACS...", end="", flush=True)
    acs_data, geoid = get_acs(lat, lon)
    if geoid:
        print(f" ✓  GEOID detectado: {geoid}")
    else:
        print(" usando medianas globales del censo")

    # ── 5. EJECUCIÓN ARQUITECTURA DE DOS ETAPAS (FUSIÓN TARDÍA) ────────────────
    feat_vec = {}
    
    if pkg.get("fusion_architecture", False):
        # A) Alinear el vector de 64 dimensiones de la imagen satelital
        emb_vector = []
        for i, col in enumerate(EMB_COLS):
            val = emb[i] if i < len(emb) and not np.isnan(emb[i]) else MEDIANS.get(col, 0)
            emb_vector.append(val)
            
        # B) ETAPA 1: Escalar y predecir señal de precio latente con el Ridge de Producción
        scaler = pkg["sat_scaler"]
        ridge  = pkg["sat_ridge"]
        emb_scaled = scaler.transform([emb_vector])
        
        # Guardamos la variable sintética clave que compacta la foto satelital
        feat_vec["pred_price_satelital"] = float(ridge.predict(emb_scaled)[0])
        
        # C) Rellenar las demás capas que solicita el LightGBM de la Etapa 2
        for col in FEATS:
            if col == "pred_price_satelital":
                continue
            # Censo ACS
            elif acs_data and col in acs_data and not np.isnan(acs_data.get(col, np.nan)):
                feat_vec[col] = acs_data[col]
            # Crimen CompStat
            elif col in crimen_loc:
                feat_vec[col] = crimen_loc[col]
            # KNN Espacial (Físicas de la propiedad, OSM y NDVI)
            elif col in local_data and not np.isnan(local_data.get(col, np.nan)):
                feat_vec[col] = local_data[col]
            # Mediana de respaldo
            else:
                feat_vec[col] = MEDIANS.get(col, 0)
    else:
        raise ValueError("El archivo .pkl cargado no corresponde a la arquitectura híbrida Late Fusion.")

    # ── 6. ETAPA 2: Inferencia final del LightGBM Combinado ───────────────────
    X              = np.array([[feat_vec[f] for f in FEATS]])
    price_per_sqft = float(np.expm1(model.predict(X)[0]))
    sqft_est       = local_data.get("sqft", MEDIANS.get("sqft", 1500))
    total_price    = price_per_sqft * sqft_est

    # Incertidumbre basada en el MedAE de tu modelo optimizado (~$32/sqft)
    MEDAE = 32
    low   = (price_per_sqft - MEDAE) * sqft_est
    high  = (price_per_sqft + MEDAE) * sqft_est

    print(f"\n  {'─'*55}")
    print(f"  RESULTADO DEL VALUADOR SATELITAL MULTIMODAL (R²=0.5187)")
    print(f"  {'─'*55}")
    print(f"  Precio Estimado Inmueble: ${total_price:>15,.0f}")
    print(f"  Rango de Mercado (±MedAE): ${low:>15,.0f} – ${high:,.0f}")
    print(f"  Valor Unitario Calculado:  ${price_per_sqft:>15,.2f}/sqft")
    print(f"  Tamaño de Construcción:    {sqft_est:>15,.0f} sqft (KNN imputado)")
    print(f"  Confianza del Entorno KNN: {confianza.upper()}")
    print(f"  {'─'*55}\n")

    return {"total_price": total_price, "price_per_sqft": price_per_sqft}


# ── modo interactivo de consola ───────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*65)
    print("  MVP VALUADOR GEOSPATIAL — ST. LOUIS CITY (PRODUCCIÓN)")
    print("  Core: Earth Embeddings + Late Fusion Tabular Ensamble")
    print("═"*65)
    print("\nCoordenadas de prueba sugeridas para el demo:")
    print("  Soulard (Oportunidad Histórica): Lat: 38.601670  Lon: -90.235610")
    print("  Lafayette (Alta Plusvalía):       Lat: 38.616082  Lon: -90.233371\n")

    while True:
        try:
            lat_in = float(input("Latitud  (ej. 38.601670): "))
            lon_in = float(input("Longitud (ej. -90.235610): "))
            predict(lat_in, lon_in)
            
            otro = input("¿Deseas evaluar otra ubicación en St. Louis City? (s/n): ").strip().lower()
            if otro != "s":
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n⚠ Error en el procesamiento del punto geográfico: {e}")
            break

    print("\nMVP finalizado correctamente.")