# AlphaEarth Embeddings: A Practical Tutorial

**Author:** Manuel Alonso De la Tejera González  
**Project:** Earth Embedding Benchmarks for Geospatial Prediction  
**Supervisor:** Carlos López de la Cerda — Washington University in St. Louis  

---

## What is AlphaEarth?

AlphaEarth Foundations is a geospatial foundation model developed by Google DeepMind. It takes satellite imagery as input and produces a compact vector representation — an **embedding** — that summarizes the physical characteristics of a location as observed from space.

The model fuses three data sources simultaneously:
- **Sentinel-2** — optical multispectral imagery (13 bands, 10m resolution)
- **Sentinel-1** — SAR radar imagery (penetrates clouds, captures surface structure)
- **Auxiliary layers** — elevation, temperature, and other geophysical variables

It is trained with a **supervised multi-task objective**: instead of reconstructing masked patches (like Clay or Prithvi), it learns to predict several known variables for each location simultaneously — land cover, temperature, elevation, and others. This means the 64 dimensions of the output vector are oriented toward variables that meaningfully describe a place, rather than toward visual texture alone.

### The product: GSED

AlphaEarth's applied product is the **Global Satellite Embedding Dataset (GSED)** — a pre-computed database of 64-dimensional embedding vectors at 10-meter resolution, covering the entire Earth annually from 2017 to 2024. It is hosted on **Google Earth Engine** and queryable by coordinates.

Key specs:
| Property | Value |
|----------|-------|
| Embedding dimensions | 64 (bands A00–A63) |
| Spatial resolution | 10 meters per pixel |
| Temporal coverage | 2017–2024 (annual) |
| GEE Asset ID | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` |
| Value range | −1 to 1 (unit vectors on S⁶³) |

The key advantage for applied researchers: **you do not need to run any model or process any imagery**. You query coordinates and receive the vector. No GPU, no deep learning infrastructure required.

---

## Setup

### 1. Create a Google Earth Engine account

Go to https://earthengine.google.com/noncommercial/ and register for non-commercial access. Select **Community tier** (free). You will need a Google Cloud project — create one and name it something descriptive (e.g., `earth-embeddings-project`).

### 2. Install Python dependencies

```bash
pip install earthengine-api geemap geopandas pandas numpy
```

### 3. Authenticate

Run this once from the terminal:

```bash
earthengine authenticate
```

This opens a browser window. Sign in with your Google account and copy the authentication token back to the terminal.

### 4. Initialize in Python

```python
import ee

ee.Initialize(project='your-project-name')
```

Replace `'your-project-name'` with the Google Cloud project you created.

---

## Important: How the dataset is organized

The GSED is **not a single global image**. It is organized as ~97,000 tiles in UTM projection, one tile per geographic area. This means you cannot query the collection directly with a point — you first need to filter to the tile that covers your area of interest using `.filterBounds()`.

```python
# This is the correct pattern
col = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
img = col.filterBounds(your_point).filterDate('2022-01-01', '2022-12-31').first()

# This will return null — do NOT do this
img = col.filterDate('2022-01-01', '2022-12-31').first()  # wrong tile
```

---

## Method 1: Single point extraction

Extract the 64-dimensional embedding for one coordinate.

```python
import ee
import pandas as pd

ee.Initialize(project='earth-embeddings-project')

# Load the collection and filter to your area + year
col = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
point = ee.Geometry.Point([-99.1588, 19.4178])  # Roma Norte, Mexico City

img = col.filterBounds(point).filter(
    ee.Filter.date('2022-01-01', '2022-12-31')
).first()

# Extract all 64 dimensions
embedding = img.reduceRegion(
    reducer=ee.Reducer.first(),
    geometry=point,
    scale=10,
    maxPixels=1e9
).getInfo()

print(f"Number of dimensions: {len(embedding)}")
print(f"First 5 values:")
for i in range(5):
    key = f'A{i:02d}'
    print(f"  {key}: {embedding[key]:.6f}")
```

**Expected output:**
```
Number of dimensions: 64
First 5 values:
  A00: 0.098424
  A01: -0.186082
  A02: 0.236463
  A03: -0.059116
  A04: 0.008858
```

---

## Method 2: Multiple points (batch extraction)

For research applications, you typically need embeddings for many locations at once. The efficient approach is `reduceRegions()`, which processes all points server-side in a single call.

```python
import ee
import pandas as pd

ee.Initialize(project='earth-embeddings-project')

# Define multiple points of interest
locations = {
    'Roma Norte':          [-99.1588, 19.4178],
    'Bosque Chapultepec':  [-99.1939, 19.4116],
    'Tepito':              [-99.1338, 19.4412],
    'Polanco':             [-99.1950, 19.4320],
    'Iztapalapa':          [-99.0600, 19.3600],
}

# Build a FeatureCollection from the points
features = [
    ee.Feature(
        ee.Geometry.Point(coords),
        {'name': name}
    )
    for name, coords in locations.items()
]
fc = ee.FeatureCollection(features)

# Load the tile covering Mexico City
reference_point = ee.Geometry.Point([-99.1588, 19.4178])
img = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
    .filterBounds(reference_point) \
    .filter(ee.Filter.date('2022-01-01', '2022-12-31')) \
    .first()

# Extract embeddings for all points at once
results = img.reduceRegions(
    collection=fc,
    reducer=ee.Reducer.first(),
    scale=10
).getInfo()

# Parse results into a DataFrame
rows = []
for feature in results['features']:
    props = feature['properties']
    row = {'name': props.get('name')}
    for i in range(64):
        key = f'A{i:02d}'
        row[key] = props.get(key)
    rows.append(row)

df = pd.DataFrame(rows)
print(df[['name', 'A00', 'A01', 'A02', 'A03', 'A04']])
```

**Expected output:**
```
                 name       A00       A01       A02       A03       A04
0          Roma Norte  0.098424 -0.186082  0.236463 -0.059116  0.008858
1  Bosque Chapultepec  0.186082 -0.327812  0.035433 -0.002215  0.017778
2              Tepito  0.153787 -0.192910  0.003937 -0.088827 -0.022207
3             Polanco       ...       ...       ...       ...       ...
4         Iztapalapa       ...       ...       ...       ...       ...
```

---

## Method 3: Extraction for polygon areas (AGEBs)

When your unit of analysis is a polygon rather than a point — such as census block groups or AGEBs — you aggregate the pixel-level embeddings within each polygon using a spatial mean.

```python
import ee
import geopandas as gpd
import pandas as pd
import json

ee.Initialize(project='earth-embeddings-project')

# Load your AGEB shapefile
# Download from: https://datos.cdmx.gob.mx/dataset/ageb-urbanas
gdf = gpd.read_file('agebs_cdmx.shp').to_crs('EPSG:4326')

# Convert GeoDataFrame to EE FeatureCollection
def row_to_feature(row):
    geom = json.loads(row.geometry.to_json())
    return ee.Feature(ee.Geometry(geom), {'cve_ageb': row['CVEGEO']})

features = [row_to_feature(row) for _, row in gdf.iterrows()]
fc = ee.FeatureCollection(features)

# Load AlphaEarth tile for CDMX
reference_point = ee.Geometry.Point([-99.1588, 19.4178])
img = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
    .filterBounds(reference_point) \
    .filter(ee.Filter.date('2022-01-01', '2022-12-31')) \
    .first()

# Extract mean embedding per AGEB
results = img.reduceRegions(
    collection=fc,
    reducer=ee.Reducer.mean(),  # mean over all pixels within the polygon
    scale=10
).getInfo()

# Parse into DataFrame
rows = []
for feature in results['features']:
    props = feature['properties']
    row = {'cve_ageb': props.get('cve_ageb')}
    for i in range(64):
        key = f'A{i:02d}'
        row[key] = props.get(key)
    rows.append(row)

df_embeddings = pd.DataFrame(rows)
df_embeddings.to_csv('agebs_embeddings_2022.csv', index=False)
print(f"Extracted embeddings for {len(df_embeddings)} AGEBs")
print(df_embeddings.head())
```

> **Note:** For large FeatureCollections (thousands of polygons), GEE may time out on `.getInfo()`. In that case, use `ee.batch.Export.table.toDrive()` to export results to Google Drive and download them.

---

## Method 4: Visualization in JavaScript (Code Editor)

For visual exploration, use the GEE Code Editor at https://code.earthengine.google.com

```javascript
// Load collection and filter to CDMX
var col = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');
var punto = ee.Geometry.Point([-99.1588, 19.4178]);

var img = col
  .filterBounds(punto)
  .filter(ee.Filter.date('2022-01-01', '2022-12-31'))
  .first();

// Visualize first 3 dimensions as false color
// Same-color areas have similar physical environments
Map.centerObject(punto, 11);
Map.addLayer(
  img,
  {bands: ['A00', 'A01', 'A02'], min: -0.5, max: 0.5},
  'AlphaEarth Embeddings CDMX 2022'
);

// Extract embedding for a specific point
var embedding = img.reduceRegion({
  reducer: ee.Reducer.first(),
  geometry: punto,
  scale: 10,
  maxPixels: 1e9
});

print('Embedding (64 dims):', embedding);
```

---

## Using embeddings in a predictive model

Once you have the embeddings as a DataFrame, they integrate directly into any standard statistical workflow:

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# Load embeddings (from Method 3 above)
df_emb = pd.read_csv('agebs_embeddings_2022.csv')

# Load your outcome variable (e.g., median rent price per AGEB)
df_prices = pd.read_csv('agebs_prices.csv')  # cve_ageb + precio_renta_mediano

# Load census covariates
df_censo = pd.read_csv('agebs_censo2020.csv')  # cve_ageb + sociodemographic vars

# Merge all sources
df = df_emb.merge(df_prices, on='cve_ageb').merge(df_censo, on='cve_ageb')
df = df.dropna()

# Define feature sets
emb_cols   = [f'A{i:02d}' for i in range(64)]
censo_cols = ['GRAPROES', 'PHOGAR', 'VPH_AGUADV', 'VPH_ELECT', 'VPH_INTER']
y = np.log(df['precio_renta_mediano'])

# Three models: incremental evaluation
for name, cols in [
    ('Census only',      censo_cols),
    ('Embeddings only',  emb_cols),
    ('Census + Embeddings', censo_cols + emb_cols),
]:
    X = df[cols]
    r2 = cross_val_score(
        RandomForestRegressor(n_estimators=200, random_state=42),
        X, y, cv=5, scoring='r2'
    ).mean()
    print(f"{name:30s}  R² = {r2:.3f}")
```

The comparison across these three models answers the core research question: **do embeddings capture information about the built environment that census variables cannot?**

---

## Comparing locations with cosine similarity

Since AlphaEarth vectors are unit-length (they live on the unit sphere S⁶³), cosine similarity is the natural comparison metric:

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Extract embedding vectors for three locations
roma       = np.array([df[df.name=='Roma Norte'][emb_cols].values[0]])
chapultepec = np.array([df[df.name=='Bosque Chapultepec'][emb_cols].values[0]])
tepito     = np.array([df[df.name=='Tepito'][emb_cols].values[0]])

print("Cosine similarities:")
print(f"  Roma Norte vs Chapultepec: {cosine_similarity(roma, chapultepec)[0][0]:.4f}")
print(f"  Roma Norte vs Tepito:      {cosine_similarity(roma, tepito)[0][0]:.4f}")
print(f"  Chapultepec vs Tepito:     {cosine_similarity(chapultepec, tepito)[0][0]:.4f}")
```

Higher similarity means the two locations share a similar physical environment as captured by satellite.

---

## Temporal comparison (2017–2024)

Because embeddings are consistent across years, you can detect physical change in a location over time:

```python
# Extract embeddings for the same point across multiple years
years = [2017, 2019, 2021, 2022, 2023]
punto = ee.Geometry.Point([-99.1588, 19.4178])
col = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')

temporal_embeddings = {}
for year in years:
    img = col.filterBounds(punto).filter(
        ee.Filter.date(f'{year}-01-01', f'{year}-12-31')
    ).first()

    emb = img.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=punto,
        scale=10,
        maxPixels=1e9
    ).getInfo()

    temporal_embeddings[year] = emb

# Compare 2017 vs 2023
v2017 = np.array([temporal_embeddings[2017][f'A{i:02d}'] for i in range(64)])
v2023 = np.array([temporal_embeddings[2023][f'A{i:02d}'] for i in range(64)])

sim = np.dot(v2017, v2023)  # dot product of unit vectors = cosine similarity
print(f"Similarity 2017 vs 2023: {sim:.4f}")
print("(1.0 = no change, lower = physical environment changed)")
```

---

## Summary

| Task | Method | Code |
|------|--------|------|
| Single point | `reduceRegion` + `.getInfo()` | Method 1 |
| Multiple points | `reduceRegions` + `.getInfo()` | Method 2 |
| Polygon areas (AGEBs) | `reduceRegions` with mean reducer | Method 3 |
| Visual exploration | JavaScript Code Editor | Method 4 |
| Large exports | `ee.batch.Export.table.toDrive()` | Note in Method 3 |

---

## References

- Brown, C. F. et al. (2025). AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data. arXiv:2507.22291.
- Klemmer, K. et al. (2025). Earth Embeddings: Towards AI-centric representations of our planet.
- Gong, W. et al. (2026). Earth embeddings reveal diverse urban signals from space. arXiv:2604.03456.
- GEE Dataset catalog: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
