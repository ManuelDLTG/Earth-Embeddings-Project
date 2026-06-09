// ============================================================
// SCRIPT 01 — Exploración Earth Embeddings en GEE
// Estancia: Earth Embedding Benchmarks for Geospatial Prediction
// Estudiante: Manuel Alonso De la Tejera González — MCD ITAM
// Supervisor: Carlos López de la Cerda — WashU
// Semana: 2 | Fecha: junio 2026
// ============================================================
//
// Descripción:
//   Primer script de exploración del dataset GSED de AlphaEarth
//   en Google Earth Engine. Extrae embeddings de 64 dimensiones
//   para tres puntos con entornos físicos distintos en CDMX
//   y los visualiza como falso color en el mapa.
//
// Dataset:
//   GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
//   64 bandas (A00–A63) | resolución 10m | anual 2017–2024
//   Tile de CDMX: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL/xsumurnsa2vtb2xvt
// ============================================================


// ------------------------------------------------------------
// 1. CARGAR COLECCIÓN Y FILTRAR TILE DE CDMX
// ------------------------------------------------------------

var col = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');

// Punto de referencia en Roma Norte para filtrar la tile correcta
var puntoCDMX = ee.Geometry.Point([-99.1588, 19.4178]);

// Filtrar por ubicación geográfica y año
// Nota: el dataset está organizado en tiles UTM, no como imagen global.
// filterBounds() selecciona solo la tile que cubre CDMX.
var img = col
  .filterBounds(puntoCDMX)
  .filter(ee.Filter.date('2022-01-01', '2022-12-31'))
  .first();

print('Dataset cargado correctamente');
print('Bandas disponibles (64 dims):', img.bandNames());
print('Tile ID:', img.id());


// ------------------------------------------------------------
// 2. DEFINIR PUNTOS DE INTERÉS — tres entornos distintos
// ------------------------------------------------------------

// Roma Norte: zona residencial densa, clase media-alta
var puntoRoma = ee.Geometry.Point([-99.1588, 19.4178]);

// Bosque de Chapultepec: área verde extensa en medio urbano
var puntoBosque = ee.Geometry.Point([-99.1939, 19.4116]);

// Tepito: zona comercial popular, alta densidad, uso mixto
var puntoTepito = ee.Geometry.Point([-99.1338, 19.4412]);


// ------------------------------------------------------------
// 3. EXTRAER EMBEDDINGS — vector 64D por punto
// ------------------------------------------------------------

// Roma Norte
var embRoma = img.reduceRegion({
  reducer: ee.Reducer.first(),
  geometry: puntoRoma,
  scale: 10,
  maxPixels: 1e9
});

// Bosque de Chapultepec
var embBosque = img.reduceRegion({
  reducer: ee.Reducer.first(),
  geometry: puntoBosque,
  scale: 10,
  maxPixels: 1e9
});

// Tepito
var embTepito = img.reduceRegion({
  reducer: ee.Reducer.first(),
  geometry: puntoTepito,
  scale: 10,
  maxPixels: 1e9
});


// ------------------------------------------------------------
// 4. IMPRIMIR RESULTADOS EN CONSOLA
// ------------------------------------------------------------

print('=== EMBEDDING ROMA NORTE (64 dims) ===');
print(embRoma);

print('=== COMPARACIÓN PRIMERAS 5 DIMENSIONES ===');
print('Zona           | A00    | A01    | A02    | A03    | A04');
print('Roma Norte     :', embRoma.get('A00'),  embRoma.get('A01'),  embRoma.get('A02'));
print('Chapultepec    :', embBosque.get('A00'), embBosque.get('A01'), embBosque.get('A02'));
print('Tepito         :', embTepito.get('A00'), embTepito.get('A01'), embTepito.get('A02'));

// Nota: los vectores completos de 64 dims son los que se usarán
// en el pipeline de Python para el análisis inmobiliario.
// Aquí solo mostramos las primeras dimensiones para diagnóstico visual.


// ------------------------------------------------------------
// 5. VISUALIZACIÓN EN MAPA — falso color con A00, A01, A02
// ------------------------------------------------------------

Map.centerObject(puntoCDMX, 11);

// Capa principal: embeddings como falso color
// Las tres primeras dimensiones se mapean a R, G, B
// Zonas con el mismo color tienen entorno físico similar
Map.addLayer(
  img,
  {bands: ['A00', 'A01', 'A02'], min: -0.5, max: 0.5},
  'AlphaEarth Embeddings CDMX 2022'
);

// Marcar los tres puntos de interés
Map.addLayer(puntoRoma,   {color: 'FF0000'}, 'Roma Norte');
Map.addLayer(puntoBosque, {color: '00FF00'}, 'Bosque Chapultepec');
Map.addLayer(puntoTepito, {color: '0000FF'}, 'Tepito');


// ------------------------------------------------------------
// 6. PRÓXIMOS PASOS (semanas 3–4)
// ------------------------------------------------------------
//
// Este script es exploración. El pipeline completo será en Python:
//
//   1. Descargar shapefile de AGEBs de CDMX (datos.cdmx.gob.mx)
//   2. Calcular centroide de cada AGEB
//   3. Extraer embedding 64D por centroide con earthengine-api Python
//      usando reduceRegions() para procesar todos los AGEBs en batch
//   4. Exportar tabla a CSV: cve_ageb | A00 | A01 | ... | A63
//   5. Merge con Censo INEGI 2020 y precios scrapeados de Inmuebles24
//   6. Correr modelos: solo censo / solo embeddings / ambos
//   7. Evaluar R² incremental
//
// Referencia: Gong et al. (2026) usaron el mismo enfoque para
// 14 indicadores urbanos en 6 ciudades de EE.UU. con LightGBM.
