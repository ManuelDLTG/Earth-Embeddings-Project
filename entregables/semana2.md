# Entregable — Semana 2

**Nombre:** Manuel Alonso De la Tejera González  
**Fecha:** junio 2026

---

## Parte A — Resumen de la maquinaria



---

## Parte B — Earth Engine + repo

### Estado de Earth Engine

- **Cuenta creada:** ✓
- **Proyecto GEE:** `earth-embeddings-project`
- **Dataset accesible:** `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` ✓
- **Total tiles en colección:** 97,155
- **Tile de CDMX identificada:** `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL/xsumurnsa2vtb2xvt`

### Embeddings extraídos

Primeras 5 dimensiones del vector de 64D para tres zonas de CDMX (año 2022):

| Zona | A00 | A01 | A02 | A03 | A04 |
|------|-----|-----|-----|-----|-----|
| Roma Norte | 0.0984 | −0.1861 | 0.2365 | −0.0591 | 0.0089 |
| Bosque Chapultepec | 0.1861 | −0.3278 | 0.0354 | −0.0022 | 0.0178 |
| Tepito | 0.1538 | −0.1929 | 0.0039 | −0.0888 | −0.0222 |

Los tres vectores son claramente distintos entre sí, especialmente en A02: Roma Norte (0.236) vs. Chapultepec (0.035) vs. Tepito (0.004). Esa variación refleja diferencias reales en el entorno físico capturadas desde el satélite.

### Repo de GitHub

- **URL:** https://github.com/ManuelDLTG/Earth-Embeddings-Project
- **Colaborador agregado:** `kennyldc` 
- **Script de exploración:** `scripts/01_gee_exploration.js` ✓
