# YatriGaan — Indian Railway Delay Prediction System

YatriGaan is a full-stack data science application that predicts Indian Railway delay probability and time based on historical route statistics, environment risks, temporal factors, and infrastructure constraints.

## Project Structure

The project has been segregated into clean, decoupled components:

```
train_Delay_prediction/
├── backend/
│   └── main.py                 # FastAPI application
├── frontend/
│   └── app.py                  # Streamlit application
├── prediction/
│   ├── predictor.py            # Prediction wrapper and features processing logic
│   ├── models/                 # Serialized model pickles
│   │   ├── xgb_trained.pkl     # Trained XGBoost model
│   │   ├── encoder.pkl         # Fit One-Hot encoder
│   │   ├── scaler.pkl          # Fit Standard Scaler
│   │   └── feature_columns.pkl # Feature ordering list
│   ├── scripts/                # Preprocessing and model training scripts
│   │   ├── save_preprocessors.py
│   │   ├── train_delay_prediction.ipynb
│   │   ├── feature_engineering.ipynb
│   │   └── visualizations/     # Model training plots
│   │       ├── target_distribution.png
│   │       └── top_correlations.png
│   └── yatrigaan_master_prompt.md  # Architectural requirements specification
├── data/
│   └── ir_train.csv            # Dataset (contains ir_train.csv)
├── requirements.txt            # Project dependencies
└── README.md                   # Setup and execution guide
```

---

## Installation & Setup

1. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### 1. Start the Backend API (FastAPI)
Run the backend server from the `train_Delay_prediction` root directory:
```bash
python -m uvicorn backend.main:app --reload
```
The API documentation will be available at `http://127.0.0.1:8000/docs`.

### 2. Start the Frontend (Streamlit)
Run the Streamlit dashboard in a separate terminal:
```bash
streamlit run frontend/app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## Model Re-Training & Preprocessing

If you need to fit and save the preprocessor (scaler/encoder) again based on your updated dataset:
1. Ensure the training CSV is located at either:
   - `D:\indian-railways-predict-train-delay\ir_train.csv` (default)
   - Or inside the project's root `data/ir_train.csv` (fallback)
2. Run the script:
   ```bash
   python prediction/scripts/save_preprocessors.py
   ```
   This will automatically save updated pickles directly into `prediction/models/`.
