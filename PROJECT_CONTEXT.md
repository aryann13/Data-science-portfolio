# YatriGaan — Indian Railway Delay Prediction System
## Complete Project Context Document

---

## 1. Project Overview

**YatriGaan** is a full-stack, end-to-end machine learning application that predicts Indian Railway train delay probability and estimated delay duration in real-time. It is built as a decoupled microservice architecture with a FastAPI backend serving an XGBoost classifier and a Streamlit frontend providing an interactive glassmorphism dashboard.

**GitHub:** https://github.com/aryann13/Data-science-portfolio  
**Live Backend API:** https://yatrigaan-backend.onrender.com  
**Tech Stack:** Python 3.12, XGBoost, Scikit-Learn, Pandas, NumPy, FastAPI, Uvicorn, Streamlit, Plotly, Joblib, Render (Cloud)

---

## 2. Project Directory Structure

```
train_Delay_prediction/
├── backend/
│   └── main.py                          # FastAPI application (API endpoint definitions)
├── frontend/
│   └── app.py                           # Streamlit dashboard (751 lines, glassmorphism UI)
├── prediction/
│   ├── predictor.py                     # Core ML inference logic (encoding, scaling, prediction)
│   ├── models/                          # Serialized ML artifacts
│   │   ├── xgb_trained.pkl              # Trained XGBoost classifier (~294 KB)
│   │   ├── encoder.pkl                  # Fitted OneHotEncoder (~3 KB)
│   │   ├── scaler.pkl                   # Fitted StandardScaler (~1.5 KB)
│   │   └── feature_columns.pkl          # Feature column ordering list (80 columns)
│   ├── scripts/
│   │   ├── feature_engineering.ipynb     # Feature engineering notebook
│   │   ├── train_delay_prediction.ipynb  # Model training notebook
│   │   ├── save_preprocessors.py        # Script to re-fit and serialize encoder/scaler
│   │   └── visualizations/
│   │       ├── target_distribution.png
│   │       └── top_correlations.png
│   └── yatrigaan_master_prompt.md        # Architectural requirements specification
├── data/
│   └── ir_train.csv                     # Raw dataset (337 MB, 1.5M rows, 45 columns) — gitignored
├── render.yaml                          # Render.com deployment config (IaC)
├── requirements.txt                     # Python dependencies
├── .gitignore
└── README.md
```

---

## 3. Dataset

- **File:** `ir_train.csv`
- **Size:** 337 MB, **1,500,000 rows**, **45 columns**
- **Source:** Indian Railway operational data
- **Target Variable:** `is_delayed` (binary: 0 = On Time, 1 = Delayed)
- **Original columns include:** journey_id, train_number, departure_date, train_type, traction_type, zone, zone_abbr, season, distance_km, num_scheduled_stops, scheduled_travel_hours, loco_age_years, coach_age_years, maintenance_score, zone_fog_index, zone_congestion_index, route_historical_ontime_pct, delay_minutes, primary_delay_cause, is_delayed, and many more.

---

## 4. Feature Engineering Pipeline

Performed in `feature_engineering.ipynb`. Steps:

### Step 1: Drop 10 columns
Columns removed for being identifiers, leakage features, zero-variance, or noise:
- `journey_id` (unique identifier — no predictive value)
- `departure_date` (temporal features extracted separately as year/month/day_of_week)
- `delay_minutes` (direct leakage — target-derived)
- `primary_delay_cause` (leakage — only known after delay occurs)
- `zone` (redundant with `zone_abbr`)
- `is_overloaded` (near-zero variance)
- `fog_risk_score` (redundant with `zone_fog_index`)
- `seat_utilisation_pct` (no predictive signal for delay)
- `is_circular_route` (near-zero variance)
- `train_number` (high cardinality — 5000+ unique values)

### Step 2: Remaining 34 raw features (+ 1 target)

**12 Numerical/Continuous features (StandardScaler applied):**
| Feature | Description | Range |
|---|---|---|
| `distance_km` | Route distance | 10 – 3000 km |
| `num_scheduled_stops` | Number of stops | 0 – 50+ |
| `scheduled_travel_hours` | Planned journey duration | 0.5 – 48 hrs |
| `psr_count` | Permanent Speed Restrictions on route | 0 – 20+ |
| `zone_fog_index` | Fog severity in the zone | 0.0 – 1.0 |
| `zone_congestion_index` | Track traffic congestion | 0.0 – 1.0 |
| `season_severity_score` | Weather severity score | -3.0 – 3.0 |
| `loco_age_years` | Age of the locomotive | 0 – 50 yrs |
| `coach_age_years` | Age of coaches | 0 – 50 yrs |
| `maintenance_score` | Rake maintenance quality | 0 – 100 |
| `late_incoming_rake` | Whether the rake arrived late from its previous trip | 0 or 1 |
| `route_historical_ontime_pct` | Historical on-time percentage for the route | 0.0 – 1.0 |

**16 Binary/Boolean features (used as-is, no scaling):**
| Feature | Description |
|---|---|
| `year` | Year of departure |
| `month` | Month (1-12) |
| `day_of_week` | Day of week (0=Mon, 6=Sun) |
| `departure_hour` | Hour of departure (0-23) |
| `is_weekend` | Is it Saturday/Sunday? |
| `is_night_departure` | Departure between 10PM-5AM? |
| `is_peak_hour` | Departure during rush hours? |
| `is_festival_season` | Is it a major Indian festival period? |
| `is_monsoon_season` | Is it monsoon season? |
| `track_doubled` | Is the route double-tracked? |
| `is_hdn_route` | Is it a High Density Network route? |
| `is_electrified` | Is the route electrified? |
| `has_lhb_coaches` | Does the train use modern LHB coaches? |
| `is_rake_shared` | Is the rake shared between train services? |
| `is_special_train` | Is this a special/holiday train? |
| `is_fog_risk` | Is there a fog risk flag? |

**6 Categorical features (OneHotEncoder applied):**
| Feature | Unique Values | Examples |
|---|---|---|
| `train_type` | 15 | Superfast Express, Rajdhani Express, Vande Bharat Express, Passenger Train, DEMU/MEMU, etc. |
| `season` | 6 | Winter/Fog, Summer, Monsoon, Pre-Monsoon, Post-Monsoon, Autumn |
| `zone_abbr` | 16 | NR, CR, WR, SR, ER, SCR, NWR, ECR, ECoR, NCR, NER, NFR, SECR, SER, SWR, WCR |
| `source_station_category` | 6 | A1, A, B, C, D, E |
| `destination_station_category` | 6 | A1, A, B, C, D, E |
| `traction_type` | 3 | Electric (25kV AC), Diesel, Dual |

### Step 3: One-Hot Encoding
The 6 categorical columns expand into **52 binary columns** after encoding.

### Step 4: Final feature matrix
- **Total feature columns after encoding:** 80
- **Target:** `is_delayed` (binary classification)
- The exact column order is saved as `feature_columns.pkl` and used at inference time via `df.reindex(columns=feature_columns, fill_value=0)` to guarantee column alignment.

---

## 5. Model Training

Performed in `train_delay_prediction.ipynb`.

- **Algorithm:** XGBoost Classifier (`xgboost.XGBClassifier`)
- **Train/Test Split:** Standard sklearn train_test_split
- **Preprocessing:** StandardScaler on 12 continuous features, OneHotEncoder on 6 categorical features
- **Hyperparameter tuning:** Performed (details in notebook)
- **Serialization:** Model saved via `joblib.dump()` as `xgb_trained.pkl`

### Preprocessing Artifact Generation
`save_preprocessors.py` independently re-fits the encoder and scaler on the full dataset and saves:
1. `encoder.pkl` — fitted `OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype='int8')`
2. `scaler.pkl` — fitted `StandardScaler` on the 12 numerical columns
3. `feature_columns.pkl` — ordered list of all 80 feature column names

---

## 6. Backend Architecture

**File:** `backend/main.py` (FastAPI)

### API Endpoints:
| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check — returns `{"message": "YatriGaan Delay Prediction API is running!"}` |
| `POST` | `/predict` | Accepts a JSON payload with 34 raw features and returns prediction results |

### Input Schema (Pydantic `TrainJourney`):
Accepts all 34 raw features as a JSON body with sensible defaults. Key fields:
- `train_number` (metadata, not a model feature)
- 12 numerical features (distance_km, zone_fog_index, etc.)
- 16 binary features (is_weekend, track_doubled, etc.)
- 6 categorical features as strings (train_type, season, zone_abbr, etc.)

### Prediction Pipeline (`prediction/predictor.py`):
1. **Pop** `train_number` (UI metadata, not a model feature)
2. **One-Hot Encode** the 6 categorical columns using the saved `encoder.pkl`
3. **Build** a full DataFrame by concatenating numerical + encoded columns
4. **Scale** the 12 numerical columns using the saved `scaler.pkl`
5. **Reindex** the DataFrame to match `feature_columns.pkl` (80 columns in exact training order, fill missing with 0)
6. **Predict** using `model.predict()` and `model.predict_proba()`
7. **Return** a JSON response with: status, delay_probability, predicted_delay_minutes, risk_level, route_progress_percentage, and delay factors breakdown

### Output Schema:
```json
{
  "train_number": "12002",
  "train_name": "Vande Bharat Express",
  "status": "DELAYED",
  "delay_probability": 98.7,
  "predicted_delay_minutes": 59,
  "risk_level": "HIGH",
  "route_progress_percentage": 64,
  "factors": [
    {"name": "Weather & Visibility", "description": "Fog conditions present (Index: 0.90)", "impact": "HIGH"},
    {"name": "Network Congestion", "description": "Zone Congestion: 0.90 (HDN Route)", "impact": "HIGH"}
  ],
  "last_updated": "2026-07-08T07:00:00+00:00"
}
```

### Delay Factor Analysis Logic:
The backend generates an explainability breakdown based on input thresholds:
- **Weather & Visibility:** Triggered when `zone_fog_index >= 0.6` or `is_fog_risk == 1`
- **Network Congestion:** Triggered when `zone_congestion_index >= 0.6` or `is_hdn_route == 1`
- **Route Reliability:** Triggered when `route_historical_ontime_pct < 0.6`
- **Incoming Rake Delay:** Triggered when `late_incoming_rake == 1`
- **Rake Maintenance:** Triggered when `maintenance_score < 50`

---

## 7. Frontend Architecture

**File:** `frontend/app.py` (Streamlit, 751 lines)

### Design System:
- **Theme:** Glassmorphism with Indian Railway color palette (Saffron #FF9933, Navy #003366, Blue gradients)
- **Typography:** Google Fonts — Poppins (headings), Inter (body)
- **Background:** CSS linear gradient `#F0F8FF → #E0EFFF → #ADD8E6`
- **Cards:** Frosted glass effect via `backdrop-filter: blur(12px)` and semi-transparent backgrounds

### UI Components:
1. **Header:** YatriGaan logo (🚆 emoji), title, and tagline
2. **Sidebar Input Panel** — 3 expandable sections with only the most impactful features:
   - 🚂 **Core Train Details:** Select Train, Train Type (15 options), Traction Type (3 options), Distance, Loco Age
   - 🛤️ **Operations & Route:** Railway Zone (16 zones), Maintenance Score, Route On-Time %, Late Incoming Rake
   - 🌧️ **Environment & Congestion:** Fog Index slider (0-1), Congestion Index slider (0-1)
3. **Digital Boarding Pass:** Train name, number, ON_TIME/DELAYED badge
4. **3 Metric Cards:** Delay Probability %, Predicted Delay (minutes), Risk Level (LOW/MEDIUM/HIGH)
5. **Plotly Gauge Chart:** Semi-circular speedometer showing delay probability (0-100)
6. **Route Progress Bar:** Animated train icon on a horizontal track with origin → destination
7. **Delay Factor Analysis Cards:** Color-coded cards (green/yellow/red) with impact badges

### Demo Mode vs Live API Mode:
- **Demo Mode (toggle ON):** Uses a mock heuristic formula for predictions — no backend required
- **Live API Mode (toggle OFF):** Sends real HTTP POST to the FastAPI backend at the configured `YATRIGAAN_API_URL` environment variable
- **Graceful Fallback:** If the API is unreachable, automatically falls back to mock data and shows a yellow warning banner

### Hardcoded Background Features:
To keep the UI clean, 23 low-impact features are hardcoded in the payload (year=2024, month=1, has_lhb_coaches=1, track_doubled=1, etc.) and sent to the API invisibly. Only the 11 most impactful features are exposed in the sidebar.

---

## 8. Deployment

### Backend (FastAPI on Render):
- **Platform:** Render.com (Free Tier)
- **Configuration:** `render.yaml` (Infrastructure as Code)
- **Runtime:** Python 3.12
- **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- **URL:** `https://yatrigaan-backend.onrender.com`
- Note: Free tier instances spin down after inactivity and take ~50 seconds for cold starts.

### Frontend (Streamlit Cloud — pending):
- **Platform:** Streamlit Community Cloud
- **Main file:** `frontend/app.py`
- **Environment Variable:** `YATRIGAAN_API_URL = "https://yatrigaan-backend.onrender.com/predict"`

---

## 9. Key Design Decisions

1. **Decoupled Architecture:** Backend and frontend are completely independent services. The frontend communicates with the backend via REST API. This allows independent scaling and deployment.
2. **Feature Column Alignment:** `feature_columns.pkl` stores the exact 80-column order from training. At inference, `df.reindex(columns=feature_columns, fill_value=0)` guarantees the model sees features in the correct order, even if a new categorical value is encountered.
3. **Graceful Degradation:** The frontend has a built-in mock prediction engine so the dashboard works even when the backend is offline.
4. **Simplified UI:** Only 11 high-impact features are shown to the user. The remaining 23 features are hardcoded with sensible defaults to avoid overwhelming the user with a 34-field form.

---

## 10. Verified Test Results

| Scenario | Fog Index | Congestion Index | Model Prediction | Risk Level |
|---|---|---|---|---|
| Disaster (Heavy fog + congestion) | 0.90 | 0.90 | **98.7% delay probability, 59 min** | HIGH |
| Default (Moderate conditions) | 0.40 | 0.60 | **37.2% delay probability, 0 min** | LOW |

---

## 11. Dependencies

```
fastapi, uvicorn, python-multipart, streamlit, requests, plotly, pandas, numpy, joblib, scikit-learn, xgboost
```
