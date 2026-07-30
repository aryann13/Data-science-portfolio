"""
prediction/scripts/train_classifier.py
---------------------------------------
Trains an XGBoost Classifier (`xgb_trained.pkl`) for `is_delayed`.
Uses the exact same clean feature set as `save_preprocessors.py` and `train_regressor.py`.
"""

import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# ── Paths Setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")

# ── 1. Load Data ─────────────────────────────────────────────────────────────
primary_path = r"D:\indian-railways-predict-train-delay\ir_train.csv"
fallback_path = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "ir_train.csv"))

file_path = primary_path if os.path.exists(primary_path) else fallback_path
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not locate ir_train.csv at {file_path}")

print(f"Loading dataset from: {file_path} ...")
df = pd.read_csv(file_path)

# Extract Target
y = df['is_delayed']

# ── 2. Memory optimisation ───────────────────────────────────────────────────
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]) and col != 'is_delayed':
        if str(df[col].dtype).startswith('int'):
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
        else:
            df[col] = df[col].astype(np.float32)

# ── 3. Drop columns to match saved preprocessors ─────────────────────────────
cols_to_drop = [
    'journey_id', 'departure_date', 'delay_minutes', 'primary_delay_cause',
    'zone', 'is_overloaded', 'fog_risk_score', 'is_fog_risk', 'seat_utilisation_pct',
    'is_circular_route', 'train_number', 'scheduled_travel_hours', 'year',
    'is_peak_hour', 'is_special_train', 'is_delayed', 'is_monsoon_season', 
    'is_electrified', 'is_weekend'
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

# ── 4. Apply Preprocessors ───────────────────────────────────────────────────
print("Applying existing preprocessors...")
encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))

cat_cols = ['train_type', 'season', 'zone_abbr', 'source_station_category', 'destination_station_category', 'traction_type']
encoded_array = encoder.transform(df[cat_cols])
encoded_col_names = encoder.get_feature_names_out(cat_cols)
encoded_df = pd.DataFrame(encoded_array, columns=encoded_col_names, index=df.index)

df = df.drop(columns=cat_cols)
df = pd.concat([df, encoded_df], axis=1)

cols_to_scale = [
    'distance_km', 'num_scheduled_stops', 'psr_count', 'zone_fog_index',
    'zone_congestion_index', 'season_severity_score', 'loco_age_years',
    'coach_age_years', 'maintenance_score', 'late_incoming_rake',
    'route_historical_ontime_pct'
]
df[cols_to_scale] = scaler.transform(df[cols_to_scale])

# Reorder columns to exactly match what the model expects
X = df.reindex(columns=feature_columns, fill_value=0)

# ── 5. Train Classifier ──────────────────────────────────────────────────────
print("Training XGBoost Classifier (this may take a few minutes)...")
classifier = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
classifier.fit(X, y)

# ── 6. Save Model ────────────────────────────────────────────────────────────
model_save_path = os.path.join(MODELS_DIR, "xgb_trained.pkl")
joblib.dump(classifier, model_save_path)
print(f"[SUCCESS] Classifier saved to {model_save_path}")
