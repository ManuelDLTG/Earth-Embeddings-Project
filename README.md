# Earth Embeddings para Predicción de Precios Inmobiliarios
### St. Louis City, Missouri — Estancia de Investigación ITAM MCD 2026

Pipeline de extremo a extremo que integra **Earth Embeddings satelitales (AlphaEarth/GSED V1)** con datos censales del ACS, Walk Score, y CompStat de crimen para predecir el precio por pie cuadrado de casas unifamiliares en St. Louis City, Missouri.

**Supervisor:** Carlos López de la Cerda — Washington University in St. Louis  
**Autor:** Manuel Alonso De la Tejera González — ITAM MCD

---

## Resultados

| Modelo | Arquitectura | R² | RMSE | MedAE |
|---|---|---|---|---|
| A — Propiedad + ACS (baseline) | LightGBM | 0.501 | $53/sqft | $31/sqft |
| C — Propiedad + ACS + EMB | LightGBM | 0.506 | $53/sqft | $31/sqft |
| F — Propiedad + ACS + NDVI + OSM | LightGBM | 0.512 | $53/sqft | $31/sqft |
| **G-Fusion ★** | **Ridge (EMB) → LightGBM** | **0.5192** | **$52/sqft** | **$30/sqft** |

★ Modelo en producción — arquitectura de fusión tardía (Late Fusion) en dos etapas.

---

## Arquitectura G-Fusion (Late Fusion)

El hallazgo metodológico central de este proyecto es que inyectar las 64 dimensiones del embedding directamente en LightGBM causa ruido por la maldición de la dimensionalidad. La solución implementada usa fusión tardía en dos etapas:

```
Etapa 1 — Procesamiento Visual Continuo
  Earth Embeddings (64 dims)
       ↓
  StandardScaler → Ridge Regression (α=10)
       ↓
  pred_price_satelital  ← señal visual comprimida en 1 número limpio

Etapa 2 — Fusión Multimodal Tabular
  pred_price_satelital
  + PROP (sqft, beds, baths, year_built, lot_size)
  + ACS Census 2023 (26 vars, block group)
  + Crime SLMPD CompStat 2026 (violent_crime_rate, total_crime_rate)
  + Walk Score / Transit Score / Bike Score
  + OSM (distancias urbanas)
       ↓
  LightGBM (CV geográfica por ZIP, 5 folds)
       ↓
  R² = 0.5192
```

**Por qué Ridge y no LightGBM directo sobre los embeddings:** Ridge con regularización L2 maneja de forma nativa la multicolinealidad del espacio de embeddings — dimensiones que son linealmente dependientes se cancelan automáticamente. LightGBM en cambio trata cada dimensión como feature independiente y puede sobreajustar al ruido inter-dimensional con n=4,172.

---

## Estructura del repositorio

```
earth-embeddings-stlouis/
├── scripts/
│   ├── 00_combine_redfin_csvs.py     # Combina CSVs de Redfin y limpia
│   ├── 01_extract_embeddings.py      # AlphaEarth via GEE (buffer 200m)
│   ├── 02_extract_ndvi_batch.py      # NDVI Sentinel-2 via GEE batch
│   ├── 03_extract_osm_features.py    # Distancias urbanas (OpenStreetMap)
│   ├── 04_extract_crime.py           # Crimen por distrito (SLMPD CompStat)
│   ├── 05_extract_schools.py         # Escuelas más cercanas (NCES/OSM)
│   ├── 06_extract_walkscore.py       # Walk Score / Transit Score / Bike Score
│   ├── 07_build_master.py            # Merge todas las capas + ACS Census
│   ├── 09_final_model.py             # G-Fusion: Ridge + LightGBM, CV espacial
│   └── 10_predict.py                 # MVP: lat/lon → precio estimado
├── archive/
│   ├── 02_download_acs_stlouis.py    # Reemplazado por integración en 07
│   └── 09_neural_attention.py        # Experimento EmbGate (ver Notas)
├── reporte/
│   └── reporte_estancia_final.docx
├── data/                             # No versionado (.gitignore)
│   └── .gitkeep
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Pipeline

```
Redfin CSVs → 00_combine
                  ↓
  GEE AlphaEarth → 01_extract_embeddings   (64 dims, 200m buffer)
  GEE Sentinel-2 → 02_extract_ndvi_batch   (NDVI x3 escalas)
  OpenStreetMap  → 03_extract_osm_features (6 distancias)
  SLMPD CompStat → 04_extract_crime        (tasa por distrito policial)
  NCES / OSM     → 05_extract_schools      (dist. escuela más cercana)
  Walk Score API → 06_extract_walkscore    (Walk/Transit/Bike Score)
                  ↓
  Census ACS API → 07_build_master         (26 vars block group + merge)
                  ↓
       Ridge(EMB) → pred_satelital
                  ↓
       LightGBM  → 09_final_model          (G-Fusion, CV espacial por ZIP)
                  ↓
       lat/lon   → 10_predict              (MVP: precio en 4-6s)
```

---

## Datos

Las 4,172 propiedades Single Family se obtuvieron combinando múltiples descargas de Redfin (sold-1yr a sold-5yr). Los archivos CSV **no están versionados** por tamaño y privacidad:

1. Ir a `redfin.com/county/1714/MO/St-Louis-City/filter/include=sold-5yr`
2. Table View → Download CSV (máx. 350 por descarga, repetir con filtros)
3. `python scripts/00_combine_redfin_csvs.py`

---

## Instalación

```bash
git clone https://github.com/ManuelDLTG/earth-embeddings-stlouis.git
cd earth-embeddings-stlouis
pip install -r requirements.txt
earthengine authenticate
```

**APIs requeridas:**
- Google Earth Engine — `earthengine authenticate`
- Census Bureau — gratuita: `api.census.gov/data/key_signup.html`
- Walk Score — gratuita: `walkscore.com/professional/api.php`

---

## Uso del MVP

```bash
python scripts/10_predict.py
```

```
Latitud  (ej. 38.6085): 38.6085
Longitud (ej. -90.2042): -90.2042

[1/4] Buscando vecinos más cercanos... ✓  5 vecinos (más cercano: 87m)
      sqft: 1,842 | beds: 3.0 | baths: 2.0 | año: 1912
[2/4] Crimen del distrito... ✓  Distrito 1 — violent_rate: 3.79/1k
[3/4] Earth Embedding (GEE)... ✓  (64/64 dims válidas)
[4/4] Census ACS... ✓  GEOID: 295101231003

  RESULTADO (G-Fusion)
  ────────────────────────────────────────
  Precio estimado:      $   301,400
  Rango (±1 MedAE):     $   271,400 – $331,400
  Precio/sqft:          $       163/sqft
  Confianza KNN:        alta
```

---

## Hallazgos principales

1. **Late Fusion supera a la inyección directa de embeddings:** Ridge(EMB) → `pred_satelital` como feature sintético en LightGBM logra R²=0.5192 vs 0.506 con embeddings directos. La regularización L2 de Ridge maneja la multicolinealidad del espacio latente que LightGBM no puede resolver solo.

2. **Bug crítico en GEE:** `.first()` sobre GSED devuelve un tile aleatorio → todos None. Solución: `.filterDate(...).mosaic()`. No documentado oficialmente.

3. **El NDVI está implícito en el embedding:** la señal de vegetación (r=−0.24) existe en Sentinel-2, que AlphaEarth usa como input. Ridge lo recupera mejor que LightGBM directo.

4. **Crimen a nivel de distrito aporta** cuando se incluye como `violent_crime_rate` en la Etapa 2 de G-Fusion, junto con Walk Score y otras amenidades urbanas.

---

## Todos los modelos evaluados

| # | Modelo | Arquitectura | R² |
|---|---|---|---|
| A | Propiedad + ACS | LightGBM | 0.501 |
| B | Solo embeddings | LightGBM | 0.072 |
| C | Prop + ACS + EMB | LightGBM | 0.506 |
| D | Prop + ACS + NDVI | LightGBM | 0.499 |
| E | Prop + ACS + Crime | LightGBM | 0.471 |
| F | Prop + ACS + NDVI + OSM | LightGBM | 0.512 |
| G | Todo directo | LightGBM | 0.480 |
| **G-Fusion ★** | **Ridge(EMB) → LightGBM** | **Late Fusion** | **0.5192** |
| EmbGate NN | Red neuronal + gate EMB | PyTorch | 0.241 |

---

## Fase 2 propuesta: Computer Vision Micro

Usando las URLs de los 4,172 listings de Redfin, se puede automatizar la descarga de la imagen de portada de cada propiedad para entrenar un clasificador CNN que identifique materiales de construcción (ladrillo histórico de STL, vinilo, madera) e inyecte estas etiquetas categóricas en la Etapa 2 del G-Fusion. Esto rompería el techo actual del R² accediendo a información del interior visual de la propiedad que el embedding satelital no captura.

---

## Trabajo futuro

- [ ] Fase 2: descarga masiva de imágenes Redfin + CNN clasificador de materiales
- [ ] Ampliar a múltiples ciudades (Chicago, Atlanta) para variación cross-city
- [ ] Datos de crimen a nivel de incidente con lat/lon (SLMPD histórico)
- [ ] G-Fusion con 15,000+ propiedades para validar escalabilidad

---

## Referencia

```
De la Tejera González, M.A. (2026). Earth Embeddings para la Predicción
de Precios Inmobiliarios: Un experimento en St. Louis City, Missouri.
Reporte de Estancia de Investigación, ITAM MCD.
Supervisor: López de la Cerda, C. Washington University in St. Louis.
```

---

*Estancia de Investigación — Maestría en Ciencia de Datos, ITAM — Julio 2026*
