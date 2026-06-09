# Estancia de Investigación — Earth Embedding Benchmarks for Geospatial Prediction

**Estudiante:** Manuel Alonso De la Tejera González  
**Programa:** Maestría en Ciencia de Datos — ITAM  
**Supervisor:** Carlos López de la Cerda — Washington University in St. Louis  
**Periodo:** 2026

---

## Descripción

Proyecto de estancia de investigación que benchmarka Earth Embeddings (AlphaEarth, SatCLIP, Clay) para predicción de variables urbanas e inmobiliarias en la Ciudad de México a nivel AGEB.

El objetivo central es evaluar si los embeddings generados por AlphaEarth capturan características del entorno construido que tienen poder predictivo sobre precios inmobiliarios más allá de las variables sociodemográficas del Censo 2020.

---

## Estructura del repositorio

```
estancia-earth-embeddings/
├── scripts/          # Scripts de GEE (JavaScript) y Python
│   └── 01_gee_exploration.js
├── notebooks/        # Jupyter notebooks de análisis
├── data/             # Referencias a fuentes de datos (no datos crudos)
├── entregables/      # Resúmenes semanales
└── README.md
```

---

## Fuentes de datos

| Fuente | Descripción | Acceso |
|--------|-------------|--------|
| AlphaEarth GSED | Embeddings 64D, 10m, 2017–2024 | Google Earth Engine |
| Censo INEGI 2020 | Variables sociodemográficas por AGEB | inegi.org.mx |
| AGEBs CDMX | Polígonos Marco Geoestadístico 2020 | datos.cdmx.gob.mx |
| Precios inmobiliarios | Listings scrapeados de Inmuebles24 | scraping |

---

## Pipeline del proyecto

```
AGEBs CDMX (centroides)
    ↓
Query AlphaEarth en GEE → vector 64D por AGEB
    ↓
Merge con Censo 2020 + precios inmobiliarios
    ↓
Tres modelos: solo censo / solo embeddings / ambos
    ↓
Evaluación incremental (R²) — valor del embedding sobre el censo
```

---

## Referencias principales

- Klemmer et al. (2025). Earth Embeddings: Towards AI-centric representations of our planet.
- Brown et al. (2025). AlphaEarth Foundations. arXiv:2507.22291
- Gong et al. (2026). Earth embeddings reveal diverse urban signals from space. arXiv:2604.03456
- López de la Cerda, C. (2026). Earth Embeddings for Political Science. WashU Qualifying Paper.
