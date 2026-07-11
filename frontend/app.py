"""
app.py — YatriGaan: Reliable Delay Oracle
==========================================
"""

import os
import time
import random
from datetime import datetime, timezone
from typing import Any

import requests
import streamlit as st
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
API_URL: str = os.getenv("YATRIGAAN_API_URL", "http://127.0.0.1:8000/predict")

TRAIN_OPTIONS: dict[str, str] = {
    "12002": "12002 — Vande Bharat Express",
    "12951": "12951 — Mumbai Rajdhani",
    "12301": "12301 — Howrah Rajdhani",
    "12259": "12259 — Sealdah Duronto",
    "12627": "12627 — Karnataka Express",
    "22691": "22691 — Bengaluru Rajdhani",
    "12245": "12245 — Shatabdi Express",
    "12560": "12560 — Shatabdi Express (NR)",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. CSS — Glassmorphism + Typography + Animations
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS: str = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

/* ── Global resets ─────────────────────────────────────────────────────── */
html, body {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Poppins', sans-serif !important;
    font-weight: 700 !important;
}

/* ── Page background ───────────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #F0F8FF 0%, #E0EFFF 40%, #ADD8E6 100%);
    min-height: 100vh;
}

/* ── Sidebar styling ───────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(0,0,0,0.1) !important;
}

/* ── Glassmorphism card (For text-only components) ─────────────────────── */
.glass-card {
    background: rgba(255, 255, 255, 0.55);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.45);
    border-radius: 20px;
    padding: 2rem 2.2rem;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0, 51, 102, 0.12);
    transition: transform 200ms cubic-bezier(0.4, 0, 0.2, 1),
                box-shadow 200ms cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0, 51, 102, 0.18);
}

/* Make Streamlit native containers look like glass cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.55) !important;
    backdrop-filter: blur(18px) !important;
    -webkit-backdrop-filter: blur(18px) !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-radius: 20px !important;
    box-shadow: 0 8px 32px rgba(0, 51, 102, 0.12) !important;
}

/* ── Header styling ────────────────────────────────────────────────────── */
.yatri-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}
.yatri-header h1 {
    color: #003366;
    font-size: 2.6rem !important;
    margin-bottom: 0 !important;
    letter-spacing: -0.02em;
}
.yatri-subtitle {
    color: #336699;
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    font-style: italic;
    opacity: 0.85;
    margin-top: 0.2rem;
}
.yatri-logo {
    font-size: 3rem;
    display: block;
    margin-bottom: 0.3rem;
}

/* ── Status badges ─────────────────────────────────────────────────────── */
.status-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 50px;
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.03em;
}
.badge-delayed {
    background: linear-gradient(135deg, #FF9933, #FF7733);
    color: #fff;
}
.badge-ontime {
    background: linear-gradient(135deg, #28a745, #218838);
    color: #fff;
}

/* ── Factor cards ──────────────────────────────────────────────────────── */
.factor-card {
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    border-left: 4px solid #ccc;
    transition: transform 150ms ease-in-out;
}
.factor-card:hover {
    transform: translateY(-1px);
}
.factor-high { border-left-color: #FF9933 !important; }
.factor-medium { border-left-color: #E6A817 !important; }
.factor-low { border-left-color: #6c8ebf !important; }

.factor-impact {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.15rem 0.55rem;
    border-radius: 4px;
    display: inline-block;
}
.impact-high { background: #FF993322; color: #CC6600; }
.impact-medium { background: #E6A81722; color: #B8860B; }
.impact-low { background: #00336615; color: #336699; }

/* ── Journey timeline ──────────────────────────────────────────────────── */
.timeline-track {
    position: relative;
    height: 6px;
    background: #c8dce8;
    border-radius: 3px;
    margin: 2rem 0 1.5rem 0;
}
.timeline-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #003366, #ADD8E6);
    transition: width 1s ease-in-out;
}
.timeline-train {
    position: absolute;
    top: -14px;
    font-size: 1.6rem;
    transition: left 1s ease-in-out;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
}
.timeline-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: #336699;
    font-weight: 500;
}

/* ── Prediction button ─────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #FF9933, #FF7733) !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.7rem 2rem !important;
    box-shadow: 0 4px 15px rgba(255, 153, 51, 0.35) !important;
    transition: transform 150ms ease, box-shadow 150ms ease !important;
}
.stButton > button * {
    color: white !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(255, 153, 51, 0.5) !important;
    border: none !important;
}

/* ── Warning banner ────────────────────────────────────────────────────── */
.fallback-banner {
    background: rgba(255, 153, 51, 0.15);
    border: 1px solid #FF9933;
    border-radius: 10px;
    padding: 0.7rem 1.2rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #995500;
    text-align: center;
    margin: 0.5rem 0 1rem 0;
}

/* ── Metric cards ──────────────────────────────────────────────────────── */
.metric-box {
    text-align: center;
    padding: 1rem;
}
.metric-value {
    font-family: 'Poppins', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #003366;
}
.metric-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: #6688aa;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 0.2rem;
}

/* ── Footer ────────────────────────────────────────────────────────────── */
.yatri-footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: #6688aa;
    opacity: 0.7;
}

/* ── Loading animation ─────────────────────────────────────────────────── */
@keyframes trainMove {
    0%   { transform: translateX(-100px); opacity: 0; }
    20%  { opacity: 1; }
    80%  { opacity: 1; }
    100% { transform: translateX(calc(100vw - 100px)); opacity: 0; }
}
.loading-train {
    font-size: 2.5rem;
    animation: trainMove 2.5s ease-in-out infinite;
    display: inline-block;
}
</style>
"""


def load_custom_css() -> None:
    """Inject all custom CSS into the Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. HEADER
# ─────────────────────────────────────────────────────────────────────────────
def render_header() -> None:
    """Render the YatriGaan logo, title, and subtitle."""
    st.markdown(
        """
        <div class="yatri-header">
            <span class="yatri-logo">🚆</span>
            <h1>YatriGaan</h1>
            <p class="yatri-subtitle">
                "Reliable Delay Oracle: Kripya Dhyaan Dijiye Predictions."
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. INPUT PANEL (Sidebar)
# ─────────────────────────────────────────────────────────────────────────────
def render_input_panel() -> dict:
    """
    Render the sidebar input form containing all 34 real model features.
    """
    with st.sidebar:
        st.markdown("### 🚄 Journey Parameters")
        
        # Demo / Live toggle
        demo_mode = st.toggle("🔌 Demo Mode", value=True,
                              help="ON = uses simulated data. OFF = calls the live FastAPI backend.")
        st.session_state["demo_mode"] = demo_mode

        if demo_mode:
            st.info("Demo Mode — no backend required")
        else:
            st.warning("🔴 Live API Mode")

        st.markdown("---")
        
        # 1. Core Train Details
        with st.expander("🚂 Core Train Details", expanded=True):
            train_number = st.selectbox(
                "Select Train",
                options=list(TRAIN_OPTIONS.keys()),
                format_func=lambda k: TRAIN_OPTIONS[k],
            )
            train_type = st.selectbox("Train Type", [
                "Superfast Express", "Rajdhani Express", "Shatabdi Express",
                "Vande Bharat Express", "Mail/Express", "Duronto Express",
                "Intercity Express", "Passenger Train", "DEMU/MEMU",
                "Garib Rath Express", "Jan Shatabdi Express",
                "Humsafar Express", "Tejas Express", "Gatimaan Express",
                "Sampark Kranti Express"
            ])
            traction_type = st.selectbox("Traction Type", ["Electric (25kV AC)", "Diesel", "Dual"])
            distance_km = st.number_input("Distance (km)", min_value=10.0, value=450.0)
            loco_age_years = st.number_input("Loco Age (Years)", 0.0, 50.0, 5.0)
            
        # 2. Operations & Route
        with st.expander("🛤️ Operations & Route", expanded=False):
            zone_abbr = st.selectbox("Railway Zone", [
                "NR", "CR", "WR", "SR", "ER", "SCR", "NWR",
                "ECR", "ECoR", "NCR", "NER", "NFR", "SECR",
                "SER", "SWR", "WCR"
            ])
            maintenance_score = st.slider("Maintenance Score (0-100)", 0.0, 100.0, 72.0, help="Lower score = poorly maintained rake")
            route_historical_ontime_pct = st.slider("Historical Route On-Time %", 0.0, 1.0, 0.78)
            late_incoming_rake = st.checkbox("Late Incoming Rake?", value=False, help="Did the train arrive late from its previous journey?")

        # 3. Environment & Congestion
        with st.expander("🌧️ Environment & Congestion", expanded=False):
            zone_fog_index = st.slider("Fog Index (0-1)", 0.0, 1.0, 0.4, help="1.0 = Blind fog, 0.0 = Clear skies")
            zone_congestion_index = st.slider("Congestion Index (0-1)", 0.0, 1.0, 0.6, help="1.0 = Maximum track traffic")

        st.markdown("---")
        predict_clicked = st.button("🔮 Predict Delay", use_container_width=True)

    payload: dict[str, Any] = {
        # Visible UI Inputs
        "train_number": train_number,
        "train_type": train_type,
        "traction_type": traction_type,
        "distance_km": distance_km,
        "loco_age_years": loco_age_years,
        "zone_abbr": zone_abbr,
        "maintenance_score": maintenance_score,
        "route_historical_ontime_pct": route_historical_ontime_pct,
        "late_incoming_rake": 1 if late_incoming_rake else 0,
        "zone_fog_index": zone_fog_index,
        "zone_congestion_index": zone_congestion_index,
        
        # Hardcoded Background Features (to keep UI clean)
        "year": 2024,
        "month": 1,
        "day_of_week": 0,
        "departure_hour": 8,
        "is_weekend": 0,
        "is_night_departure": 0,
        "is_peak_hour": 1,
        "is_festival_season": 0,
        "num_scheduled_stops": 8,
        "scheduled_travel_hours": distance_km / 60.0,  # rough estimate
        "is_special_train": 0,
        "has_lhb_coaches": 1,
        "is_rake_shared": 0,
        "coach_age_years": 8.0,
        "source_station_category": "A1",
        "destination_station_category": "A",
        "psr_count": 2,
        "track_doubled": 1,
        "is_electrified": 1,
        "is_hdn_route": 1,
        "season": "Winter/Fog",
        "season_severity_score": 1.2,
        "is_fog_risk": 1 if zone_fog_index >= 0.5 else 0,
        "is_monsoon_season": 0,
    }

    if predict_clicked:
        st.session_state["predict_requested"] = True

    return payload


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOADING STATE
# ─────────────────────────────────────────────────────────────────────────────
def render_loading_state() -> None:
    """Show the cultural loading animation."""
    loading_container = st.empty()
    loading_container.markdown(
        """
        <div class="glass-card" style="text-align:center; padding:2.5rem;" aria-live="polite">
            <div class="loading-train">🚄</div>
            <h3 style="color:#003366; margin-top:1rem;">
                Kripya Dhyaan Dijiye...
            </h3>
            <p style="color:#336699; font-family:'Inter',sans-serif;">
                Querying the Delay Oracle.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(1.5)
    loading_container.empty()


# ─────────────────────────────────────────────────────────────────────────────
# 5. API FETCH + MOCK DATA
# ─────────────────────────────────────────────────────────────────────────────
def _generate_mock_response(payload: dict) -> dict:
    """Generate a realistic mock response for Demo Mode."""
    fog = payload["zone_fog_index"]
    congestion = payload["zone_congestion_index"]
    rake = payload["late_incoming_rake"]
    ontime_pct = payload["route_historical_ontime_pct"]

    # Heuristic math based on real model features
    base_prob = (fog * 40) + (congestion * 30) + (15 if rake else 0) + ((1.0 - ontime_pct) * 20)
    base_prob = min(98, max(5, base_prob + random.uniform(-5, 5)))
    delay_prob = round(base_prob, 1)

    is_delayed = delay_prob >= 50
    predicted_minutes = int(delay_prob * 0.6) if is_delayed else 0

    if delay_prob >= 70:
        risk_level = "HIGH"
    elif delay_prob >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    factors = []
    if fog >= 0.6:
        factors.append({
            "name": "Weather & Visibility",
            "description": f"Zone Fog Index: {fog:.2f} — significant visibility reduction",
            "impact": "HIGH" if fog >= 0.8 else "MEDIUM"
        })
    if congestion >= 0.6:
        factors.append({
            "name": "Network Congestion",
            "description": f"Zone Congestion: {congestion:.2f} — heavy traffic on route",
            "impact": "HIGH" if congestion >= 0.8 else "MEDIUM"
        })
    if rake:
        factors.append({
            "name": "Incoming Rake Delay",
            "description": "Rake arrived late from its previous run",
            "impact": "HIGH"
        })
    if not factors:
        factors.append({
            "name": "Clear Conditions",
            "description": "No significant delay factors detected",
            "impact": "LOW"
        })

    train_name = TRAIN_OPTIONS.get(payload["train_number"], f"Train {payload['train_number']}")
    train_name = train_name.split(" — ")[-1] if " — " in train_name else train_name

    return {
        "train_number": payload["train_number"],
        "train_name": train_name,
        "status": "DELAYED" if is_delayed else "ON_TIME",
        "delay_probability": delay_prob,
        "predicted_delay_minutes": predicted_minutes,
        "risk_level": risk_level,
        "route_progress_percentage": min(95, max(10, int(50 + (delay_prob - 50) * 0.3))),
        "factors": factors,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


def fetch_prediction(payload: dict, demo_mode: bool = True) -> dict:
    """Send payload to FastAPI or return mock data."""
    if demo_mode:
        return _generate_mock_response(payload)

    try:
        resp = requests.post(API_URL, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        required_keys = {"status", "delay_probability", "factors"}
        if not required_keys.issubset(data.keys()):
            raise ValueError("Malformed API response.")
        st.session_state["api_fallback"] = False
        return data
    except Exception as exc:
        st.session_state["api_fallback"] = True
        return _generate_mock_response(payload)


# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOTLY GAUGE
# ─────────────────────────────────────────────────────────────────────────────
def render_gauge(probability: float) -> None:
    """Render a circular probability gauge using Plotly."""
    bar_color = "#FF9933" if probability >= 70 else "#E6A817" if probability >= 40 else "#28a745"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability,
        number={"suffix": "%", "font": {"size": 42, "family": "Poppins", "color": "#003366"}},
        title={"text": "Delay Probability", "font": {"size": 16, "family": "Inter", "color": "#336699"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#ccc",
                     "tickfont": {"family": "Inter", "size": 11}},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(40, 167, 69, 0.1)"},
                {"range": [40, 70], "color": "rgba(230, 168, 23, 0.1)"},
                {"range": [70, 100], "color": "rgba(255, 153, 51, 0.15)"},
            ],
            "threshold": {
                "line": {"color": "#003366", "width": 2},
                "thickness": 0.75,
                "value": probability,
            },
        },
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. JOURNEY TIMELINE
# ─────────────────────────────────────────────────────────────────────────────
def render_journey_timeline(progress_pct: float) -> None:
    """Render a horizontal journey progress bar with a train icon."""
    st.markdown(
        f"""
        <div style="padding: 0.5rem 0;">
            <div class="timeline-track">
                <div class="timeline-fill" style="width: {progress_pct}%;"></div>
                <span class="timeline-train" style="left: calc({progress_pct}% - 15px);">🚄</span>
            </div>
            <div class="timeline-labels">
                <span>🏁 Origin</span>
                <span>{progress_pct}% Complete</span>
                <span>📍 Destination</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. FACTORS BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
def render_factors_breakdown(factors: list[dict]) -> None:
    """Render the delay factor cards with impact-based styling."""
    st.markdown("#### 🔍 Delay Factor Analysis")
    for factor in factors:
        impact = factor.get("impact", "LOW").upper()
        css_class = f"factor-{impact.lower()}"
        impact_class = f"impact-{impact.lower()}"
        icon = {"HIGH": "⚠️", "MEDIUM": "🔶", "LOW": "✅"}.get(impact, "ℹ️")

        st.markdown(
            f"""
            <div class="factor-card {css_class}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong style="color:#003366; font-family:'Poppins',sans-serif; font-size:0.95rem;">
                        {factor.get('name', 'Unknown')}
                    </strong>
                    <span class="factor-impact {impact_class}">{icon} {impact}</span>
                </div>
                <p style="margin:0.4rem 0 0 0; color:#4a6a8a; font-size:0.88rem;">
                    {factor.get('description', '')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9. BOARDING PASS (Main Output)
# ─────────────────────────────────────────────────────────────────────────────
def render_boarding_pass(data: dict) -> None:
    """Compose all output components into the digital boarding pass."""
    status = data.get("status", "ON_TIME")
    is_delayed = status == "DELAYED"
    badge_class = "badge-delayed" if is_delayed else "badge-ontime"
    badge_text = "⚠️ DELAYED" if is_delayed else "✅ ON TIME"
    probability = data.get("delay_probability", 0)
    delay_min = data.get("predicted_delay_minutes", 0)
    risk = data.get("risk_level", "LOW")
    progress = data.get("route_progress_percentage", 50)
    train_name = data.get("train_name", "Unknown")
    train_number = data.get("train_number", "-----")
    last_updated = data.get("last_updated", "")

    if st.session_state.get("api_fallback", False):
        st.markdown(
            '<div class="fallback-banner">'
            '⚠️ Delay Oracle unreachable — showing a simulated prediction.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Header ──
    st.markdown(
        f"""
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                <div>
                    <h2 style="color:#003366; margin:0; font-size:1.5rem;">
                        🚆 {train_number} — {train_name}
                    </h2>
                    <p style="color:#6688aa; margin:0.2rem 0 0 0; font-size:0.82rem; font-family:'Inter',sans-serif;">
                        Digital Journey Ticket • YatriGaan Oracle
                    </p>
                </div>
                <span class="status-badge {badge_class}">{badge_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Metric cards ──
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="glass-card metric-box">
                <div class="metric-value" style="color:{'#FF9933' if probability >= 60 else '#003366'};">
                    {probability}%
                </div>
                <div class="metric-label">Delay Probability</div>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        st.markdown(
            f"""
            <div class="glass-card metric-box">
                <div class="metric-value">{delay_min}<span style="font-size:1rem;"> min</span></div>
                <div class="metric-label">Predicted Delay</div>
            </div>
            """, unsafe_allow_html=True)
    with col3:
        risk_color = {"HIGH": "#FF9933", "MEDIUM": "#E6A817", "LOW": "#28a745"}.get(risk, '#003366')
        risk_icon = {"HIGH": "⚠️", "MEDIUM": "🔶", "LOW": "✅"}.get(risk, '')
        st.markdown(
            f"""
            <div class="glass-card metric-box">
                <div class="metric-value" style="color:{risk_color};">{risk_icon} {risk}</div>
                <div class="metric-label">Risk Level</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Gauge + Timeline (Native Streamlit Containers) ──
    gauge_col, timeline_col = st.columns([1, 1])

    with gauge_col:
        with st.container(border=True):
            render_gauge(probability)

    with timeline_col:
        with st.container(border=True):
            st.markdown("#### 🛤️ Route Progress")
            render_journey_timeline(progress)
            st.markdown("<br>", unsafe_allow_html=True)

    # ── Factors ──
    render_factors_breakdown(data.get("factors", []))

    if last_updated:
        try:
            ts = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            formatted = ts.strftime("%d %b %Y, %H:%M:%S UTC")
        except:
            formatted = last_updated
        st.markdown(f'<p style="text-align:right; color:#8899aa; font-size:0.75rem;">Last updated: {formatted}</p>',
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="YatriGaan — Delay Oracle", page_icon="🚆", layout="wide", initial_sidebar_state="expanded")
    load_custom_css()
    render_header()

    if "prediction_data" not in st.session_state:
        st.session_state["prediction_data"] = None
    if "predict_requested" not in st.session_state:
        st.session_state["predict_requested"] = False
    if "api_fallback" not in st.session_state:
        st.session_state["api_fallback"] = False

    payload = render_input_panel()

    if st.session_state.get("predict_requested", False):
        st.session_state["predict_requested"] = False
        render_loading_state()
        data = fetch_prediction(payload, demo_mode=st.session_state.get("demo_mode", True))
        st.session_state["prediction_data"] = data

    if st.session_state.get("prediction_data"):
        render_boarding_pass(st.session_state["prediction_data"])
    else:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:3rem;">
                <span style="font-size:4rem;">🚆</span>
                <h3 style="color:#003366; margin-top:1rem;">Welcome to YatriGaan</h3>
                <p style="color:#336699; font-family:'Inter',sans-serif; max-width:500px; margin:0.5rem auto;">
                    Adjust the journey parameters in the sidebar,
                    and click <strong>Predict Delay</strong> to consult the Oracle.
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="yatri-footer">Predictions are indicative. Built with Streamlit + FastAPI + XGBoost.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
