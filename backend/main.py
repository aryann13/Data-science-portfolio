"""
backend/main.py
---------------
Indian Railway Delay Prediction API (YatriGaan Backend)
FastAPI application that serves prediction requests.
"""

import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure root directory is in python search path to support clean imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction.predictor import predict_delay

# ── Create the FastAPI app ───────────────────────────────────────────────────
app = FastAPI(
    title="YatriGaan — Delay Prediction API",
    description="Indian Railway Delay Prediction backend for the YatriGaan dashboard.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Define what INPUT the API expects ────────────────────────────────────────
class TrainJourney(BaseModel):
    """Clean feature schema matching updated Machine Learning pipeline."""
    # Metadata for UI
    train_number: str = "12002"
    
    # Temporal & Calendar (Ints/Booleans)
    month: int = 1
    day_of_week: int = 0
    departure_hour: int = 6
    is_night_departure: int = 0
    is_festival_season: int = 0
    
    # Route & Infrastructure (Booleans)
    track_doubled: int = 1
    is_hdn_route: int = 1
    
    # Train Logistics (Booleans)
    has_lhb_coaches: int = 1
    is_rake_shared: int = 0
    late_incoming_rake: int = 0

    # Numerical features (Scaled)
    distance_km: float = 707.0
    num_scheduled_stops: int = 8
    psr_count: int = 2
    zone_fog_index: float = 0.85
    zone_congestion_index: float = 0.50
    season_severity_score: float = 1.2
    loco_age_years: float = 10.0
    coach_age_years: float = 8.0
    maintenance_score: float = 90.0
    route_historical_ontime_pct: float = 0.90
    
    # Categorical features (Encoded)
    train_type: str = "Shatabdi Express"
    season: str = "Winter/Fog"
    zone_abbr: str = "NR"
    source_station_category: str = "A1"
    destination_station_category: str = "A"
    traction_type: str = "Electric (25kV AC)"

# ── Health Check ─────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "YatriGaan Delay Prediction API is running!",
        "usage": "Send a POST request to /predict with train journey details."
    }

# ── Prediction Endpoint ──────────────────────────────────────────────────────
@app.post("/predict")
def predict(journey: TrainJourney):
    """
    Accepts train journey details and returns a delay prediction
    in the YatriGaan API contract format.
    """
    return predict_delay(journey.dict())
