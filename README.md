# Earth Embeddings para Predicción de Precios Inmobiliarios
### St. Louis City, Missouri — Estancia de Investigación ITAM MCD 2026

Pipeline de extremo a extremo que integra **Earth Embeddings satelitales (AlphaEarth/GSED V1)** con datos censales del ACS y features de OpenStreetMap para predecir el precio por pie cuadrado de casas unifamiliares en St. Louis City, Missouri.

**Supervisor:** Carlos López de la Cerda — Washington University in St. Louis  
**Autor:** Manuel Alonso De la Tejera González — ITAM MCD

---

## Resultados

| Modelo | Features | R² | RMSE | MedAE |
|---|---|---|---|---|
| A — Propiedad + ACS (baseline) | 31 | 0.501 | $53/sqft | $31/sqft |
| **C — Propiedad + ACS + Embeddings ★** | **95** | **0.506** | **$53/sqft** | **$31/sqft** |
| F — Propiedad + ACS + NDVI + OSM | 40 | 0.512 | $53/sqft | $31/sqft |

★ Modelo seleccionado para producción — único que usa solo datos derivables de satélite + censo sin features adicionales.

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
│   ├── 06_extract_walkscore.py       # Walk Score / Transit Score
│   ├── 07_build_master.py            # Merge todas las capas + ACS Census
│   ├── 08_final_model.py             # Modelos A–I con CV espacial
│   └── 09_predict.py                 # MVP: lat/lon → precio estimado
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
         GEE Sentinel-2 → 02_extract_ndvi_batch    (NDVI x3 escalas)
         OpenStreetMap  → 03_extract_osm_features  (6 distancias)
         SLMPD CompStat → 04_extract_crime         (tasa por distrito)
         NCES / OSM     → 05_extract_schools       (dist. escuela)
         Walk Score API → 06_extract_walkscore      (Walk/Transit Score)
                  ↓
         Census ACS API → 07_build_master           (26 vars block group)
                  ↓
                       → 08_final_model             (LightGBM, CV espacial)
                  ↓
              lat/lon  → 09_predict                 (MVP: precio en 4-6s)
```

---

## Datos

Las 4,172 propiedades Single Family de St. Louis City se obtuvieron combinando múltiples descargas de Redfin (sold-1yr a sold-5yr). Los archivos CSV **no están versionados** por tamaño y privacidad — para reproducir el dataset:

1. Ir a `redfin.com/county/1714/MO/St-Louis-City/filter/include=sold-5yr`
2. Cambiar a Table View → Download CSV
3. Repetir con distintos filtros de período y precio
4. Correr `python scripts/00_combine_redfin_csvs.py`

---

## Instalación

```bash
git clone https://github.com/ManuelDLTG/earth-embeddings-stlouis.git
cd earth-embeddings-stlouis
pip install -r requirements.txt

# Autenticar Google Earth Engine
earthengine authenticate
```

**APIs requeridas:**
- Google Earth Engine — `earthengine authenticate`
- Census Bureau — gratuita en `api.census.gov/data/key_signup.html`
- Walk Score — gratuita en `walkscore.com/professional/api.php` (opcional)

---

## Uso del MVP

```bash
python scripts/09_predict.py
```

```
Latitud  (ej. 38.6085): 38.6085
Longitud (ej. -90.2042): -90.2042

[1/3] Buscando vecinos más cercanos... ✓  5 vecinos (más cercano: 87m)
      sqft imputado: 1,842 | beds: 3.0 | baths: 2.0 | año: 1912
[2/3] Extrayendo Earth Embedding (GEE)... ✓  (64/64 dims válidas)
[3/3] Consultando Census ACS... ✓  GEOID: 295101231003

  RESULTADO
  ────────────────────────────────────────
  Precio estimado:      $   297,000
  Rango (±1 MedAE):     $   264,000 – $330,000
  Precio/sqft:          $       161/sqft
  Confianza KNN:        alta
```

---

## Hallazgos principales

1. **AlphaEarth aporta señal incremental** sobre el ACS (ΔR²=+0.005), con 8 de las 15 features más importantes del modelo siendo dimensiones del embedding.

2. **Bug crítico en GEE:** usar `.first()` sobre el ImageCollection de GSED devuelve un tile aleatorio que puede no cubrir la zona de interés → todos los valores son `None`. La solución es `.filterDate(...).mosaic()`. No está documentado en GEE oficialmente.

3. **El NDVI está implícito en el embedding:** la señal de vegetación (r=−0.24 con precio/sqft) existe en los datos de Sentinel-2 que AlphaEarth usa como input. LightGBM no puede recuperarla implícitamente con n=4,172; una red neuronal con más datos podría hacerlo.

4. **Crimen y escuelas a nivel agregado no aportan** sobre el ACS de block group (ΔR² negativo). La granularidad del dato importa: datos de crimen a nivel de incidente con lat/lon tendrían mayor poder predictivo.

---

## Modelos explorados

| # | Modelo | R² | Notas |
|---|---|---|---|
| A | Propiedad + ACS | 0.501 | Baseline hedónico |
| B | Solo embeddings | 0.072 | Sin contexto censal |
| C | Prop + ACS + EMB ★ | 0.506 | **Producción** |
| D | Prop + ACS + NDVI | 0.499 | NDVI no añade sobre EMB |
| E | Prop + ACS + Crime | 0.471 | Crimen agregado resta |
| F | Prop + ACS + NDVI + OSM | 0.512 | Mejor R² total |
| G–I | Combinaciones adicionales | 0.480–0.484 | Más features ≠ mejor modelo |
| EmbGate NN | Red neuronal + gate EMB | 0.241 | n insuficiente para NN |

---

## Trabajo futuro

- [ ] Ampliar a múltiples ciudades (Chicago, Atlanta) para aprovechar variación cross-city
- [ ] EmbGate con n=15,000+ para recuperar señal implícita del NDVI
- [ ] Google Open Buildings como feature de área de huella del edificio
- [ ] Crimen a nivel de incidente con lat/lon (SLMPD datos históricos)
- [ ] Imágenes de fachada via Street View Static API + CLIP embeddings

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
