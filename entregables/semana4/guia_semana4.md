# Entregable — Semana 4

**Nombre:** Manuel Alonso De la Tejera González
**Fecha:** 25 de junio de 2026

---

## Aplicación

**Estimación del precio de mercado por metro cuadrado a nivel AGEB en la Ciudad de México usando Earth Embeddings de AlphaEarth como señal geoespacial.**

- **Pregunta principal:** ¿En qué medida los Earth Embeddings de AlphaEarth capturan variación en el precio de mercado de inmuebles a nivel de AGEB en la CDMX, más allá de lo que explican las variables socioeconómicas del Censo 2020?

- **Qué se quiere predecir/estimar:** Precio mediano por metro cuadrado (MXN/m²) de listings residenciales en venta, agregado a nivel de AGEB.

- **Unidad geográfica:** AGEB urbana (Área Geoestadística Básica) — la unidad más granular disponible en el Censo 2020 de INEGI, con ~2,400 AGEBs en la CDMX urbana.

- **Por qué tiene sentido usar Earth Embeddings aquí:** El precio de un inmueble depende en parte de características físicas del entorno que son directamente observables desde el satélite: densidad de construcción, cobertura vegetal, proximidad a infraestructura, calidad del tejido urbano. Estas características son costosas de medir directamente a escala de ciudad, pero AlphaEarth las captura de forma compacta (64 dimensiones) a 10m de resolución para toda la CDMX. La hipótesis es que los embeddings añaden señal predictiva sobre el precio/m² más allá de lo que explica el perfil socioeconómico del AGEB medido por el censo — una pregunta directamente análoga a la de Gong et al. (2026) para indicadores urbanos en ciudades de EE.UU., y a la de Yue et al. (2026) para actividad económica a nivel de país.

---

## Ground Truth

- **Fuente de datos:** Listings de inmuebles en venta de Inmuebles24.com — el portal inmobiliario líder en México — extraídos mediante scraping con Playwright/curl_cffi. Fuente alternativa en evaluación: actor de Apify `ecomscrape/inmuebles24-property-listings-scraper` con proxies residenciales de México.

- **Qué mide exactamente:** Precio de lista (asking price) en MXN por metro cuadrado de construcción, de propiedades residenciales (casas y departamentos) en venta activa en la CDMX durante junio 2026. Se agrega a nivel AGEB como la mediana de todas las observaciones dentro de cada polígono AGEB.

- **Ventajas:**
  - Cobertura amplia: Inmuebles24 concentra la mayoría de la oferta formal del mercado secundario en la CDMX.
  - Alta frecuencia: el sitio se actualiza en tiempo real, lo que hace el pipeline replicable con datos más recientes.
  - Granularidad individual: cada listing tiene coordenadas GPS que permiten el join espacial exacto al AGEB.

- **Problemas o sesgos:**
  - Precio de lista ≠ precio de transacción: en la CDMX la diferencia promedia entre 6% y 12% según el mercado. El modelo predice precios de oferta, no de cierre.
  - Cobertura heterogénea: AGEBs con mercado activo (Roma, Polanco, Del Valle) tendrán muchos listings; AGEBs periféricas o industriales tendrán pocos o ninguno. El modelo solo aplica a AGEBs con cobertura suficiente (umbral mínimo: 3 listings por AGEB).
  - Sesgo de selección: los inmuebles listados en portales formales tienden a ser más caros y de mejor calidad que el mercado informal (que en algunas alcaldías representa una fracción importante de las transacciones).
  - Temporalidad: los listings de junio 2026 se cruzan con embeddings del año 2024 (GSED V1) y variables censales de 2020 — hay un desfase de hasta 6 años en las variables de control. Se asume que la estructura socioespacial de las AGEBs cambia lentamente.

- **¿Es más cercano a mercado, valuación administrativa/catastral, registro oficial, proxy?** Mercado (precios de oferta), no transacciones reales ni valuación administrativa. Es el mismo tipo de dato que usan Gong et al. (2026) — listings de portales inmobiliarios.

- **¿Qué tan viable es usarlo?** Viable con la limitación técnica actual: el sitio de Inmuebles24 usa Cloudflare Bot Management que bloquea scrapers no-browser. Soluciones en evaluación: (a) actor de Apify con proxies residenciales de México ($1/1,000 resultados), (b) Playwright con Chrome real desde Terminal, (c) fuente alternativa como Lamudi o Propiedades.com. El pipeline está listo para recibir el CSV de listings en cuanto se resuelva la extracción.

---

## Check-in de media semana

- **Opción principal:** estimación de precio/m² residencial a nivel AGEB en CDMX.
- **Fuentes encontradas:** Inmuebles24 (scraping en curso), Lamudi (alternativa), Apify actor con proxies (en evaluación).
- **Dudas abiertas:** cómo resolver el bloqueo de Cloudflare para el scraping; cuántas AGEBs quedarán con cobertura suficiente de listings (umbral mínimo a definir).
- **Decisión preliminar:** mantener la aplicación de valuación inmobiliaria. El diseño del pipeline y los datos de entrada (embeddings + censo) ya están completos. El ground truth es la pieza pendiente.

---

## Mapa del MVP

### Inputs
| Capa | Archivo | Fuente | AGEBs |
|------|---------|--------|-------|
| Embeddings geoespaciales | `alphaearth_embeddings_cdmx_2024.csv` | GEE — GSED V1 (2024) | 2,431 |
| Variables socioeconómicas | `censo2020_ageb_cdmx.csv` | INEGI — Censo 2020 | 2,433 |
| Geometrías | `poligono_ageb_urbanas_cdmx.shp` | Marco Geoestadístico INEGI 2020 | 2,431 |
| Ground truth (pendiente) | `listings_ageb_cdmx.csv` | Inmuebles24 — junio 2026 | ~600–900 AGEBs estimadas |

### Embeddings a utilizar
AlphaEarth/GSED V1 — 64 dimensiones, promedio de todos los píxeles de 10m dentro de cada polígono AGEB para el año 2024. Fue seleccionado sobre Clay, Prithvi y SatCLIP por tres razones: (a) ya viene precomputado en GEE para toda la superficie terrestre, lo que elimina el costo de GPU y de descarga de imágenes por AGEB; (b) integra imágenes ópticas (Sentinel-2) y radar (Sentinel-1) en un solo vector; (c) en el benchmark de Gong et al. (2026) sobre los mismos tres modelos, AlphaEarth fue el más consistente en tareas de predicción de indicadores urbanos a nivel de vecindario.

### Ground truth
Precio mediano por m² (MXN/m²) de listings residenciales en venta, calculado dentro de cada AGEB con al menos 3 observaciones.

### Modelo inicial
Diseño incremental con tres modelos comparados:
- **Modelo A (baseline):** regresión con solo variables del Censo 2020 (escolaridad, hacinamiento, tenencia de bienes, etc.)
- **Modelo B (embeddings):** regresión con solo los 64 dims de AlphaEarth
- **Modelo C (completo):** regresión con Censo 2020 + embeddings

Para cada modelo: LightGBM con validación cruzada espacial de 5 folds (los folds se definen por alcaldía para evitar data leakage espacial). La comparación entre modelos se hace sobre el R² y RMSE en el conjunto de prueba.

### Evaluación
- Métrica principal: R² (variación en precio/m² explicada por el modelo)
- Métrica secundaria: RMSE en MXN/m²
- Pregunta clave: ΔR²(C − A) = incremento de R² al agregar embeddings sobre el baseline censal. Si ΔR² > 0 y es estadísticamente significativo, los embeddings aportan señal geoespacial independiente.
- Validación cruzada espacial: folds por alcaldía (16 alcaldías → 5 folds agrupados), no aleatoria, para evaluar generalización geográfica.

### Output esperado
Tabla de R² y RMSE por modelo (A, B, C) con intervalo de confianza. Mapa de CDMX coloreado por precio/m² predicho vs. real. Interpretación de las dimensiones más importantes del embedding (SHAP values).

### Cómo se actualizaría el pipeline con nuevos datos

```
[Semanal/mensual]
1. Scraping de nuevos listings de Inmuebles24 → listings_ageb_cdmx.csv
2. Si hay nuevo año de GSED disponible en GEE → rerun 03_extract_alphaearth_embeddings.py
3. Merge de las tres capas (embeddings + censo + listings) → dataset_modelo.csv
4. Reentrenamiento de los tres modelos con los datos actualizados
5. Evaluación y comparación con versión anterior
6. Actualización del mapa de predicción
```

El único componente que requiere intervención manual en cada actualización es el scraping de listings — el resto del pipeline es reproducible con los scripts ya existentes (`03`, `04`).

---

## Tutoriales semana 3

| Modelo | Carpeta en repo | Descripción |
|--------|-----------------|-------------|
| AlphaEarth | `alphaearth/` | Extracción de embeddings precalculados desde GEE con Python |
| Clay | `clay/` | Tutorial completo: descarga de Sentinel-2, normalización per-banda, encoder ViT-MAE |
| Prithvi | `prithvi/` | Tutorial vía TerraTorch: 6 bandas HLS, normalización desde código fuente de terratorch |
| SatCLIP | `satclip/` | Location encoder implícito: solo coordenadas, análisis de límite de resolución L=40 |

Todos los tutoriales están en inglés, con celdas markdown que explican las decisiones técnicas y los bugs encontrados durante la implementación.

---

## Reporte

**Secciones ya redactadas:**
1. ¿Qué son los Earth Embeddings? (incluyendo las cuatro funciones de Klemmer et al.)
2. Las imágenes satelitales como insumo (tensores, bandas, escala)
3. Cómo se producen los embeddings (ViT, patchification, MAE)
4. Panorama de modelos: explícitos vs. implícitos (tabla comparativa de los cuatro modelos)
5. Pipeline aplicado: síntesis de Gong et al. y Yue et al. + pipeline de la estancia

**Falta redactar:**
- Sección de aplicación específica (esta guía sirve de base)
- Resultados y análisis (depende de tener el ground truth)
- Conclusiones

---

## Para el viernes

- **Decisión de aplicación:** confirmada — estimación de precio/m² a nivel AGEB en CDMX.
- **Problema técnico abierto:** scraping de Inmuebles24 (Cloudflare). Propuesta para la reunión: ¿autorizas usar $5 USD del actor de Apify con proxies residenciales para obtener ~5,000 listings, o prefieres explorar una fuente alternativa (Lamudi, SHF, valor catastral)?
- **Demo:** pipeline de extracción de embeddings funcionando para las 2,431 AGEBs de la CDMX — listo para mostrar.
