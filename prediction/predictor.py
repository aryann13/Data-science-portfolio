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
from datetime import datetime, timezone

# ── Paths & Loading ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load pre-trained assets
model = joblib.load(os.path.join(MODELS_DIR, "xgb_trained.pkl"))
encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))

# ── Config Constants ─────────────────────────────────────────────────────────
COLS_TO_SCALE = [
    'distance_km', 'num_scheduled_stops', 'scheduled_travel_hours',
    'psr_count', 'zone_fog_index', 'zone_congestion_index',
    'season_severity_score', 'loco_age_years', 'coach_age_years',
    'maintenance_score', 'late_incoming_rake', 'route_historical_ontime_pct'
]

CAT_COLS = [
    'train_type', 'season', 'zone_abbr',
    'source_station_category', 'destination_station_category', 'traction_type'
]

TRAIN_CATALOG = {
    "12002": "Vande Bharat Express",
    "12951": "Mumbai Rajdhani",
    "12301": "Howrah Rajdhani",
    "12259": "Sealdah Duronto",
    "12627": "Karnataka Express",
    "22691": "Bengaluru Rajdhani",
    "12245": "Shatabdi Express",
    "12560": "Shatabdi Express (NR)",
}

# ── Helper Functions ─────────────────────────────────────────────────────────
def _build_factors(journey_dict: dict, delay_prob: float) -> list[dict]:
    """Generate delay factor breakdown based on actual real input values."""
    factors = []
    
    # Weather / Fog
    zone_fog_index = journey_dict.get("zone_fog_index", 0.0)
    is_fog_risk = journey_dict.get("is_fog_risk", 0)
    if zone_fog_index >= 0.6 or is_fog_risk:
        impact = "HIGH" if zone_fog_index >= 0.8 else "MEDIUM"
        factors.append({
            "name": "Weather & Visibility",
            "description": f"Fog conditions present (Index: {zone_fog_index:.2f}) - significant visibility reduction",
            "impact": impact
        })
        
    # Congestion
    zone_congestion_index = journey_dict.get("zone_congestion_index", 0.0)
    is_hdn_route = journey_dict.get("is_hdn_route", 0)
    if zone_congestion_index >= 0.6 or is_hdn_route:
        impact = "HIGH" if zone_congestion_index >= 0.8 else "MEDIUM"
        factors.append({
            "name": "Network Congestion",
            "description": f"Zone Congestion: {zone_congestion_index:.2f} (HDN Route) - heavy traffic on route",
            "impact": impact
        })
        
    # Route History
    route_historical_ontime_pct = journey_dict.get("route_historical_ontime_pct", 1.0)
    if route_historical_ontime_pct < 0.6:
        impact = "HIGH" if route_historical_ontime_pct < 0.4 else "MEDIUM"
        factors.append({
            "name": "Route Reliability",
            "description": f"Historical on-time percentage is low ({route_historical_ontime_pct*100:.1f}%)",
            "impact": impact
        })
        
    # Incoming rake
    late_incoming_rake = journey_dict.get("late_incoming_rake", 0)
    if late_incoming_rake == 1:
        factors.append({
            "name": "Incoming Rake Delay",
            "description": "Rake arrived late from its previous run causing cascading delay",
            "impact": "HIGH"
        })
        
    # Maintenance
    maintenance_score = journey_dict.get("maintenance_score", 100.0)
    if maintenance_score < 50:
        factors.append({
            "name": "Rake Maintenance",
            "description": f"Sub-optimal maintenance score ({maintenance_score}) increases risk",
            "impact": "MEDIUM"
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

    # Derive predicted delay minutes from probability (heuristic)
    predicted_delay_minutes = int(delay_prob * 0.6) if prediction == 1 else 0

    # Risk level
    if delay_prob >= 70:
        risk_level = "HIGH"
    elif delay_prob >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Route progress (simulated)
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
        "factors": _build_factors(journey_dict, delay_prob),
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
