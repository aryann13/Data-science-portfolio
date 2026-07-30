import sys
import os

# Add root directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from prediction.predictor import predict_delay

test_payload = {
    "train_number": "12002",
    "train_type": "Shatabdi Express",
    "traction_type": "Electric (25kV AC)",
    "distance_km": 707.0,
    "loco_age_years": 10.0,
    "zone_abbr": "NR",
    "maintenance_score": 90.0,
    "route_historical_ontime_pct": 90.0,
    "late_incoming_rake": 0,
    "zone_fog_index": 0.85,
    "zone_congestion_index": 0.50,
    "season": "Winter/Fog",
    "month": 1,
    "departure_hour": 6,
    "day_of_week": 0,
    "is_night_departure": 0,
    "is_festival_season": 0,
    "num_scheduled_stops": 8,
    "has_lhb_coaches": 1,
    "is_rake_shared": 0,
    "coach_age_years": 8.0,
    "source_station_category": "A1",
    "destination_station_category": "A",
    "psr_count": 2,
    "track_doubled": 1,
    "is_hdn_route": 1,
    "season_severity_score": 1.2
}

print("Running test prediction for 12002 Bhopal Shatabdi Express...")
result = predict_delay(test_payload)
print("\n--- PREDICTION RESULTS ---")
print(f"Status: {result['status']}")
print(f"Delay Probability: {result['delay_probability']}%")
print(f"Predicted Delay Minutes: {result['predicted_delay_minutes']} min")
print(f"Risk Level: {result['risk_level']}")
print("Factors Identified:")
for f in result['factors']:
    print(f"  - {f['name']} ({f['impact']}): {f['description']}")

print("\n[SUCCESS] Pipeline Verification Passed!")
