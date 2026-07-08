"""
Script 08: Modelo Completo (G) Optimizado con Late Fusion
Combina Regresión Ridge para procesar el embedding latente + LightGBM para integrar 
las capas estructurales, censales, urbanas (OSM), crimen y amenidades.

Requiere:
    ../data/stlouis_master_full.csv
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

Path("../data").mkdir(exist_ok=True)

# ── 1. Cargar dataset maestro ─────────────────────────────────────────────────
df = pd.read_csv("../data/stlouis_master_full.csv")
print(f"Dataset Maestro Cargado: {len(df)} propiedades")

# ── 2. Definir todas las capas de features ────────────────────────────────────
EMB_COLS  = [c for c in df.columns if c.startswith("emb_")]
NDVI_COLS = [c for c in df.columns if c.startswith("ndvi_")]
PROP_COLS = [c for c in ["sqft", "beds", "baths", "year_built", "lot_size"] if c in df.columns]

URBAN_COLS = [c for c in [
    "dist_main_road_m", "dist_park_m", "dist_transit_m", "dist_commercial_m", "road_density_300m",
    "dist_nearest_school_m", "schools_1km"
] if c in df.columns]

AMENITIES_COLS = [c for c in [
    "walk_score", "transit_score", "bike_score",
    "violent_crime_rate", "total_crime_rate", "violent_crime_ytd"
] if c in df.columns]

ACS_COLS  = [c for c in [
    "pop_total", "median_age", "median_hh_income", "pop_bachelors",
    "pop_masters", "pop_doctorate", "pop_no_schooling", "pop_labor_force",
    "pop_unemployed", "housing_units_total", "housing_units_occupied",
    "owner_occupied", "renter_occupied", "median_gross_rent", "median_home_value",
    "median_year_built", "commuters_total", "commute_car", "commute_transit",
    "pop_white", "pop_black", "pop_asian", "pct_owner_occupied", "pct_college_plus",
    "unemployment_rate", "pct_renter"
] if c in df.columns]

TARGET = "price_per_sqft"
y      = np.log1p(df[TARGET].values)
groups = df["zip"].fillna(0).astype(str).values

# ── 3. Validación Cruzada Geográfica (Fusión en 2 Etapas para Modelo G) ───────
print("\n" + "="*75)
print("ENTRENANDO MODELO G COMPLETO — LATE FUSION (RIDGE + LIGHTGBM)")
print("="*75)

gkf = GroupKFold(n_splits=5)
oof_predictions = np.zeros(len(y))
df["pred_price_satelital"] = 0.0

for tr, val in gkf.split(df, y, groups):
    # --- ETAPA 1: Ridge Regression para comprimir los Embeddings ---
    scaler = StandardScaler()
    X_emb_tr  = scaler.fit_transform(df.iloc[tr][EMB_COLS].fillna(df[EMB_COLS].median()))
    X_emb_val = scaler.transform(df.iloc[val][EMB_COLS].fillna(df[EMB_COLS].median()))
    
    ridge_sat = Ridge(alpha=10.0, random_state=42)
    ridge_sat.fit(X_emb_tr, y[tr])
    
    df.loc[df.index[tr], "pred_price_satelital"] = ridge_sat.predict(X_emb_tr)
    df.loc[df.index[val], "pred_price_satelital"] = ridge_sat.predict(X_emb_val)
    
    # --- ETAPA 2: LightGBM con TODAS las capas combinadas + la señal satelital ---
    ALL_TAB_FEATS = PROP_COLS + ACS_COLS + NDVI_COLS + URBAN_COLS + AMENITIES_COLS + ["pred_price_satelital"]
    
    X_tab_tr  = df.iloc[tr][ALL_TAB_FEATS].fillna(df[ALL_TAB_FEATS].median()).values
    X_tab_val = df.iloc[val][ALL_TAB_FEATS].fillna(df[ALL_TAB_FEATS].median()).values
    
    lgb_model = lgb.LGBMRegressor(
        objective="regression", metric="rmse", n_estimators=600,
        learning_rate=0.03, num_leaves=31, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=1.0, random_state=42, verbose=-1
    )
    
    lgb_model.fit(X_tab_tr, y[tr],
                  eval_set=[(X_tab_val, y[val])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
                  
    oof_predictions[val] = lgb_model.predict(X_tab_val)

# Métricas del Ensamble Multimodal G
yt, yp = np.expm1(y), np.expm1(oof_predictions)
r2_final = r2_score(yt, yp)
rmse_final = root_mean_squared_error(yt, yp)
medae_final = np.median(np.abs(yt - yp))

print(f"\nRendimiento Final Arquitectura Híbrida Modelo G (Todo Junto):")
print(f"  R² Score:   {r2_final:.4f}  <-- ¡Compara este número contra el 0.5135 del C!")
print(f"  RMSE:      ${rmse_final:.2f}")
print(f"  MedAE:     ${medae_final:.2f}")

# ── 4. Entrenamiento de Producción (Todo el Dataset) ──────────────────────────
print("\nEntrenando Modelo G Definitivo para Producción...")

scaler_prod = StandardScaler()
X_emb_all = scaler_prod.fit_transform(df[EMB_COLS].fillna(df[EMB_COLS].median()))
ridge_prod = Ridge(alpha=10.0, random_state=42)
ridge_prod.fit(X_emb_all, y)

df["pred_price_satelital"] = ridge_prod.predict(X_emb_all)

# Incluye absolutamente todas las variables mapeadas
ALL_TAB_FEATS_PROD = PROP_COLS + ACS_COLS + NDVI_COLS + URBAN_COLS + AMENITIES_COLS + ["pred_price_satelital"]
X_tab_all = df[ALL_TAB_FEATS_PROD].fillna(df[ALL_TAB_FEATS_PROD].median()).values

lgb_prod = lgb.LGBMRegressor(
    objective="regression", metric="rmse", n_estimators=500,
    learning_rate=0.03, num_leaves=31, min_child_samples=10,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
    reg_lambda=1.0, random_state=42, verbose=-1
)
lgb_prod.fit(X_tab_all, y)

# ── 5. Importancia de Variables en la Fusión G ────────────────────────────────
imp = pd.Series(lgb_prod.feature_importances_, index=ALL_TAB_FEATS_PROD).sort_values(ascending=False)
print("\nTop Features más importantes en el Ensamble G (Todo Junto):")
print(imp.head(15).to_string())

# ── 6. Exportar Artefacto Especializado ───────────────────────────────────────
joblib.dump({
    "model": lgb_prod,
    "feature_cols": ALL_TAB_FEATS_PROD, 
    "prop_cols": PROP_COLS,
    "acs_cols": ACS_COLS,
    "emb_cols": EMB_COLS,
    "medians": df[ALL_TAB_FEATS_PROD].median().to_dict(),
    "fusion_architecture": True,
    "sat_scaler": scaler_prod,
    "sat_ridge": ridge_prod
}, "../data/best_model_C.pkl")

print("\n✓ '../data/best_model_C.pkl' guardado exitosamente con arquitectura G-Fusion.")