"""
prediction/predictor.py
-----------------------
Encapsulates all logic for loading ML artifacts, executing scaling/encoding,
and running delay predictions.
"""

import os
import joblib
import pandas as pd
import numpy as np
import shap
from datetime import datetime, timezone

# ── Paths & Loading ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load pre-trained assets
model = joblib.load(os.path.join(MODELS_DIR, "xgb_trained.pkl"))
regressor = joblib.load(os.path.join(MODELS_DIR, "xgb_regressor.pkl"))
explainer = shap.TreeExplainer(model)
encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))

# ── Config Constants ─────────────────────────────────────────────────────────
COLS_TO_SCALE = [
    'distance_km', 'num_scheduled_stops',
    'psr_count', 'zone_fog_index', 'zone_congestion_index',
    'season_severity_score', 'loco_age_years', 'coach_age_years',
    'maintenance_score', 'late_incoming_rake', 'route_historical_ontime_pct'
]

CAT_COLS = [
    'train_type', 'season', 'zone_abbr',
    'source_station_category', 'destination_station_category', 'traction_type'
]

TRAIN_CATALOG = {
    "12002": "Bhopal Shatabdi Express",
    "20172": "Vande Bharat Express (NDLS-RKMP)",
    "12951": "Mumbai Rajdhani",
    "12301": "Howrah Rajdhani",
    "12259": "Sealdah Duronto",
    "12627": "Karnataka Express",
    "22691": "Bengaluru Rajdhani",
    "12245": "Shatabdi Express",
    "12560": "Shatabdi Express (NR)",
}

# ── Helper Functions ─────────────────────────────────────────────────────────
def _build_factors(full_df: pd.DataFrame, delay_prob: float) -> list[dict]:
    """Generate delay factor breakdown using SHAP values from the model."""
    factors = []
    
    # Calculate SHAP values for this instance
    shap_values = explainer.shap_values(full_df)
    
    # Handle SHAP output format
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0] # Positive class
    else:
        if len(shap_values.shape) == 3:
            shap_vals = shap_values[0, :, 1]
        elif len(shap_values.shape) == 2:
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values

    # Pair features with their SHAP values
    feature_names = full_df.columns.tolist()
    feature_shap = list(zip(feature_names, shap_vals))
    
    # Sort by SHAP value descending (most positive impact on delay first)
    feature_shap.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 3 factors driving the delay
    top_factors = [f for f in feature_shap if f[1] > 0.05][:3]
    
    for feature, shap_val in top_factors:
        impact = "HIGH" if shap_val > 0.5 else ("MEDIUM" if shap_val > 0.2 else "LOW")
        
        # Make feature names readable
        readable_name = feature.replace("_", " ").title()
        
        factors.append({
            "name": readable_name,
            "description": f"{readable_name} raised the delay risk by {shap_val:.2f} SHAP points",
            "impact": impact
        })

    # If no negative factors found but still delayed, mention general conditions
    if not factors and delay_prob >= 50:
        factors.append({
            "name": "Complex Logistics",
            "description": "Multiple minor operational factors contributing to delay",
            "impact": "LOW"
        })
        
    # If no factors found and on time, add a positive note
    if not factors:
        factors.append({
            "name": "Clear Conditions",
            "description": "No significant delay factors detected on route",
            "impact": "LOW"
        })
        
    return factors

# ── Main Predict API ─────────────────────────────────────────────────────────
def predict_delay(journey_dict: dict) -> dict:
    """
    Accepts train journey details as a dictionary and returns predicted delay metrics.
    """
    # Create local copy so we do not mutate original dict
    input_dict = journey_dict.copy()
    
    # Pop train_number as it's not a model feature, just UI metadata
    train_number = input_dict.pop("train_number", "12002")

    # One-hot encode categorical features
    cat_input = {col: [input_dict[col]] for col in CAT_COLS}
    cat_df = pd.DataFrame(cat_input)

    encoded_array = encoder.transform(cat_df)
    encoded_col_names = encoder.get_feature_names_out(CAT_COLS)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_col_names)

    # Standardize numerical features
    num_cols = [col for col in input_dict if col not in CAT_COLS]
    num_df = pd.DataFrame({col: [input_dict[col]] for col in num_cols})

    # Concat and scale
    full_df = pd.concat([num_df, encoded_df], axis=1)
    full_df[COLS_TO_SCALE] = scaler.transform(full_df[COLS_TO_SCALE])
    
    # Reorder columns to exactly match the training data order
    full_df = full_df.reindex(columns=feature_columns, fill_value=0)

    # Predict probability and class
    prediction = model.predict(full_df)[0]
    probability = model.predict_proba(full_df)[0]
    delay_prob = round(float(probability[1]) * 100, 1)

    # Derive predicted delay minutes from the Regressor model
    if prediction == 1:
        predicted_delay_minutes = int(regressor.predict(full_df)[0])
        predicted_delay_minutes = max(0, predicted_delay_minutes)
    else:
        predicted_delay_minutes = 0

    # Risk level
    if delay_prob >= 70:
        risk_level = "HIGH"
    elif delay_prob >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Route progress (simulated for UI purposes)
    route_progress = min(95, max(10, int(50 + (delay_prob - 50) * 0.3)))

    # Train name lookup
    train_name = TRAIN_CATALOG.get(train_number, f"Train {train_number}")

    return {
        "train_number": train_number,
        "train_name": train_name,
        "status": "DELAYED" if prediction == 1 else "ON_TIME",
        "delay_probability": delay_prob,
        "predicted_delay_minutes": predicted_delay_minutes,
        "risk_level": risk_level,
        "route_progress_percentage": route_progress,
        "factors": _build_factors(full_df, delay_prob),
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
