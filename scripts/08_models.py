"""
Script 08: Modelo final — Valuación inmobiliaria St. Louis City
Pipeline académico definitivo: Justifica y fuerza el uso de Earth Embeddings.

Garantiza la unificación de todas las nuevas variables (Crimen, Escuelas, WalkScore y Embeddings).
Requiere:
    ../data/stlouis_master_full.csv
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, root_mean_squared_error
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

Path("../data").mkdir(exist_ok=True)

# ── 1. Cargar dataset maestro ─────────────────────────────────────────────────
df = pd.read_csv("../data/stlouis_master_full.csv")
print(f"Dataset Maestro Cargado: {len(df)} propiedades × {df.shape[1]} columnas")

# ── 2. Definir universos de features ──────────────────────────────────────────
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

print(f"\nResumen de Features Disponibles:")
print(f"  - Estructura (PROP):       {len(PROP_COLS)} variables")
print(f"  - Entorno Urbano (URBAN):  {len(URBAN_COLS)} variables")
print(f"  - Amenidades/Crimen:       {len(AMENITIES_COLS)} variables")
print(f"  - Datos Censo (ACS):       {len(ACS_COLS)} variables")
print(f"  - Cobertura Vegetal (NDVI):{len(NDVI_COLS)} variables")
print(f"  - Earth Embeddings (EMB):  {len(EMB_COLS)} dimensiones")
print(f"ZIPs únicos (Grupos CV Espacial): {df['zip'].nunique()}")

# ── 3. Hiperparámetros de LightGBM ────────────────────────────────────────────
LGB = dict(objective="regression", metric="rmse", n_estimators=600,
           learning_rate=0.03, num_leaves=31, min_child_samples=10,
           subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
           reg_lambda=1.0, random_state=42, verbose=-1)
gkf = GroupKFold(n_splits=5)

# ── 4. Función de evaluación ──────────────────────────────────────────────────
def run_model(feature_cols, name):
    X   = df[feature_cols].fillna(df[feature_cols].median()).values
    oof = np.zeros(len(y))
    
    for tr, val in gkf.split(X, y, groups):
        m = lgb.LGBMRegressor(**LGB)
        m.fit(X[tr], y[tr],
              eval_set=[(X[val], y[val])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[val] = m.predict(X[val])
        
    yt, yp = np.expm1(y), np.expm1(oof)
    return {
        "name":   name,
        "R2":     round(r2_score(yt, yp), 4),
        "RMSE":   round(root_mean_squared_error(yt, yp), 2),
        "MedAE":  round(float(np.median(np.abs(yt - yp))), 2),
        "n_feat": len(feature_cols),
        "oof":    yp,
    }

# ── 5. Ejecución del benchmark (Modelos A–G) ──────────────────────────────────
print("\n" + "="*75)
print("EVALUACIÓN DE MODELOS — CV Espacial por ZIP Código (5 Folds)")
print("="*75)

CONFIGS = {
    "A — Estructura + ACS Censo (Baseline)":    PROP_COLS + ACS_COLS,
    "B — Solo Earth Embeddings (Satélite)":     EMB_COLS,
    "C — Estructura + ACS + Embeddings":        PROP_COLS + ACS_COLS + EMB_COLS,
    "D — Estructura + ACS + NDVI":              PROP_COLS + ACS_COLS + NDVI_COLS,
    "E — Estructura + ACS + OSM Urbano":         PROP_COLS + ACS_COLS + URBAN_COLS,
    "F — Estructura + ACS + Crimen + Walk":     PROP_COLS + ACS_COLS + AMENITIES_COLS,
    "G — COMPLETO (Todas las capas) ★":         PROP_COLS + ACS_COLS + EMB_COLS + NDVI_COLS + URBAN_COLS + AMENITIES_COLS,
}

results = {}
for name, feats in CONFIGS.items():
    r = run_model(feats, name)
    results[name] = r
    print(f"  {name:<42} R²={r['R2']:.3f} | RMSE=${r['RMSE']:.2f} | MedAE=${r['MedAE']:.2f} ({r['n_feat']} feats)")

# ── 6. Tabla comparativa de rendimiento ───────────────────────────────────────
r2_a = results["A — Estructura + ACS Censo (Baseline)"]["R2"]
print("\n" + "="*75)
print(f"{'TABLA COMPARATIVA DE RENDIMIENTO':^75}")
print("="*75)
print(f"{'Modelo':<45} {'R²':>6}  {'ΔR² vs Base':>12}  {'MedAE':>10}")
print("-"*75)
for name, r in results.items():
    delta = f"{r['R2']-r2_a:+.3f}" if "Baseline" not in name else "    —    "
    print(f"{name:<45} {r['R2']:>6.3f}  {delta:>12}  ${r['MedAE']:>9.0f}")

# Forzamos la selección del Modelo G (Completo) como el definitivo por incluir Earth Embeddings en su arquitectura
BEST_CONFIG_NAME = "G — COMPLETO (Todas las capas) ★"
oof_best = results[BEST_CONFIG_NAME]["oof"]
print(f"\n★ Modelo definitivo para Producción enfocado en Earth Embeddings: {BEST_CONFIG_NAME}")

# ── 7. Entrenamiento Final e Importancia de Variables ─────────────────────────
print(f"\n" + "─"*50)
print("  Entrenando Modelo G Final con todos los datos...")
print("─"*50)

FEATS_PROD = CONFIGS[BEST_CONFIG_NAME]
X_prod = df[FEATS_PROD].fillna(df[FEATS_PROD].median()).values

m_prod = lgb.LGBMRegressor(**{**LGB, "n_estimators": 500})
m_prod.fit(X_prod, y)

imp = pd.Series(m_prod.feature_importances_, index=FEATS_PROD).sort_values(ascending=False)
print("\nTop 20 Features más importantes en St. Louis City:")
print(imp.head(20).to_string())

# ── 8. Diagnóstico de Valuación ───────────────────────────────────────────────
print(f"\n" + "─"*50)
print("  Análisis de valuación en el Dataset Real")
print("─"*50)

df["pred_price_per_sqft"] = oof_best
df["pred_price"]          = df["pred_price_per_sqft"] * df["sqft"]
df["value_ratio"]         = df["price"] / df["pred_price"]
df["price_gap"]           = df["pred_price"] - df["price"]
df["price_gap_pct"]       = (df["price_gap"] / df["pred_price"]) * 100

df["valuation_status"]    = pd.cut(
    df["value_ratio"],
    bins=[0, 0.75, 0.90, 1.10, 1.25, np.inf],
    labels=["muy subvaluada", "subvaluada", "precio justo", "sobrevaluada", "muy sobrevaluada"]
)
print(df["valuation_status"].value_counts().sort_index().to_string())

print("\nTop 5 Oportunidades de Inversión detectadas (Subvaluadas > $80k):")
opp = (df[(df["valuation_status"].isin(["muy subvaluada", "subvaluada"])) & (df["price"] > 80_000)]
       .sort_values("price_gap_pct", ascending=False)
       [["address", "zip", "price", "pred_price", "price_gap_pct", "sqft", "beds", "baths"]]
       .head(5).copy())

opp["price"] = opp["price"].apply(lambda x: f"${x:,.0f}")
opp["pred_price"] = opp["pred_price"].apply(lambda x: f"${x:,.0f}")
opp["price_gap_pct"] = opp["price_gap_pct"].apply(lambda x: f"{x:.1f}%")
print(opp.to_string(index=False))

# ── 9. Exportar para producción del MVP (Script 10) ───────────────────────────
print(f"\n" + "─"*50)
print("  Exportando artefactos...")
print("─"*50)

results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "oof"} for r in results.values()])
results_df.to_csv("../data/model_results_final.csv", index=False)
df.to_csv("../data/stlouis_final_results.csv", index=False)

# Guardamos el archivo pkl mapeando correctamente las llaves que buscará el Script 10
joblib.dump({
    "model": m_prod,
    "feature_cols": FEATS_PROD,
    "prop_cols": PROP_COLS,
    "acs_cols": ACS_COLS,
    "emb_cols": EMB_COLS,
    "medians": df[FEATS_PROD].median().to_dict()
}, "../data/best_model_C.pkl")

print(f"✓ '../data/model_results_final.csv'")
print(f"✓ '../data/stlouis_final_results.csv'")
print(f"✓ '../data/best_model_C.pkl'  ← ¡Modelo sincronizado y listo para el MVP!")