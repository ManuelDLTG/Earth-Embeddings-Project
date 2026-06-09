# Fuentes de datos — referencia

## AlphaEarth GSED (Google Earth Engine)
- Asset ID: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- Tile CDMX: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL/xsumurnsa2vtb2xvt`
- Resolución: 10 metros por pixel
- Dimensiones: 64 (bandas A00–A63)
- Cobertura temporal: 2017–2024 (anual)
- Acceso: Google Earth Engine (proyecto: earth-embeddings-project)

## AGEBs Ciudad de México
- URL: https://datos.cdmx.gob.mx/dataset/ageb-urbanas-areas-geoestadisticas-basicas-urbanas
- Formato: Shapefile (.shp)
- Fuente: INEGI Marco Geoestadístico 2020
- ~6,000 AGEBs urbanas en CDMX

## Censo INEGI 2020 — variables por AGEB
- URL: https://www.inegi.org.mx/programas/ccpv/2020/
- Variables clave:
  - GRAPROES: grado promedio de escolaridad
  - PHOGAR: % hogares con hacinamiento
  - VPH_AGUADV: % viviendas con agua dentro
  - VPH_DRENAJ: % viviendas con drenaje
  - VPH_ELECT: % viviendas con electricidad
  - VPH_INTER: % viviendas con internet
  - TVIVPARHAB: total viviendas particulares habitadas

## Precios inmobiliarios (pendiente)
- Fuente: Inmuebles24 (scraping)
- Variables: precio_renta, precio_venta, precio_m2, coords, tipo
- Meta: 10,000–15,000 listings con coordenadas en CDMX
