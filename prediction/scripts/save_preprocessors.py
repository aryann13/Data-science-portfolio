"""
prediction/scripts/save_preprocessors.py
----------------------------------------
Run this script to re-fit and save the encoder + scaler 
so that prediction/predictor.py can use them for predictions.

Command: python prediction/scripts/save_preprocessors.py
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Paths Setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# prediction/models/
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 1. Load raw data ─────────────────────────────────────────────────────────
primary_path = r"D:\indian-railways-predict-train-delay\ir_train.csv"
fallback_path = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "ir_train.csv"))

file_path = None
if os.path.exists(primary_path):
    file_path = primary_path
elif os.path.exists(fallback_path):
    file_path = fallback_path

if not file_path:
    raise FileNotFoundError(
        f"Could not locate ir_train.csv. Checked:\n"
        f"  1. {primary_path}\n"
        f"  2. {fallback_path}"
    )

print(f"Loading dataset from: {file_path} ...")
df = pd.read_csv(file_path)

# ── 2. Memory optimisation ────────────────────────────────────────────────────
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        if str(df[col].dtype).startswith('int'):
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
        else:
            df[col] = df[col].astype(np.float32)

# ── 3. Drop collinear, redundant & uninformative columns ─────────────────────
cols_to_drop = [
    'journey_id', 'departure_date', 'delay_minutes', 'primary_delay_cause',
    'zone', 'is_overloaded', 'fog_risk_score', 'is_fog_risk', 'seat_utilisation_pct',
    'is_circular_route', 'train_number', 'scheduled_travel_hours', 'year',
    'is_peak_hour', 'is_special_train', 'is_monsoon_season', 'is_electrified', 'is_weekend'
]
df = df.drop(columns=cols_to_drop, errors='ignore')

# ── 4. Fit + save OneHotEncoder ───────────────────────────────────────────────
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Categorical columns: {cat_cols}")

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype='int8')
encoder.fit(df[cat_cols])
encoder_save_path = os.path.join(MODELS_DIR, "encoder.pkl")
joblib.dump(encoder, encoder_save_path)
print(f"Saved: {encoder_save_path}")

# ── 5. Build X just like the notebook ────────────────────────────────────────
encoded_array = encoder.transform(df[cat_cols])
encoded_col_names = encoder.get_feature_names_out(cat_cols)
encoded_df = pd.DataFrame(encoded_array, columns=encoded_col_names, index=df.index)
df = df.drop(columns=cat_cols)
df = pd.concat([df, encoded_df], axis=1)

X = df.drop(columns=['is_delayed'])

# ── 6. Fit + save StandardScaler ─────────────────────────────────────────────
cols_to_scale = [
    'distance_km', 'num_scheduled_stops', 'psr_count', 'zone_fog_index',
    'zone_congestion_index', 'season_severity_score', 'loco_age_years',
    'coach_age_years', 'maintenance_score', 'late_incoming_rake',
    'route_historical_ontime_pct'
]
scaler = StandardScaler()
scaler.fit(X[cols_to_scale])
scaler_save_path = os.path.join(MODELS_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_save_path)
print(f"Saved: {scaler_save_path}")

# ── 7. Save feature column order so the API knows the exact column order ──────
feature_columns_save_path = os.path.join(MODELS_DIR, "feature_columns.pkl")
joblib.dump(list(X.columns), feature_columns_save_path)
print(f"Saved: {feature_columns_save_path}")

print("\nAll preprocessing objects saved successfully!")
