# YatriGaan — Master Build Prompt (v2, Engineering-Complete)

> **Status:** Your original creative direction is preserved 100% — colors, copy, component ideas, and structure are untouched. Items marked **🆕** are additions closing gaps that would otherwise cause problems in a real build or a placement demo.

---

## Why This Revision Exists (Read This First)

Your original prompt nails the *aesthetic* brief but is silent on things that decide whether the app actually survives contact with a real backend, a judge's laptop, or a colorblind reviewer. Specifically, it was missing:

1. **API Contract** — you specified what the UI should *show*, never the exact JSON shape the backend sends. Without this, your mock data and real data will drift apart the day you connect FastAPI.
2. **Failure Handling** — "non-blocking HTTP POST" was mentioned, but there was no spec for what happens when the backend is down, slow, or returns garbage. A premium dashboard can't just show a Python traceback.
3. **Typography** — every color was specified down to the hex code, but no font. Type is half of what makes something look "premium SaaS" vs. "default Streamlit."
4. **State Management** — Streamlit reruns the entire script on every widget interaction. Without `st.session_state`, your boarding pass will reset or flicker the moment someone touches a slider after predicting.
5. **Function-Level Modularity** — "modular" was requested, but no function boundaries were defined, which is where most Streamlit apps turn into 400-line spaghetti anyway.
6. **Accessibility** — saffron-vs-green as the *only* risk signal fails for colorblind users; needs a text/icon backup.
7. **Config Management** — the API URL was hardcoded in the requirement. For a placement project, hardcoded URLs are a giveaway of inexperience; env-var config plus a Demo/Live toggle reads much better in a panel review.
8. **Definition of Done** — "fully executable" was the ask, but there was no `requirements.txt` or run instruction, so "executable" was implied, not guaranteed.

Everything below folds these in without touching your creative vision.

---

## 1. Role & Persona

Act as a **Lead Python Frontend Developer and UI/UX Designer**, specializing in high-fidelity, production-grade Streamlit dashboards for data-science/ML products. Treat this as a portfolio-grade placement project — code quality and polish both matter, not just visuals.

---

## 2. Product Identity

| Field | Value |
|---|---|
| App Name | **YatriGaan** (Passengers) |
| Subtitle | *"Reliable Delay Oracle: Kripya Dhyaan Dijiye Predictions."* |
| 🆕 Browser tab title | `YatriGaan — Delay Oracle` |
| 🆕 Favicon | 🚆 emoji or inline SVG train mark (see §4.3) |
| 🆕 `st.set_page_config` | `layout="wide"`, `initial_sidebar_state="expanded"` |

---

## 3. System Architecture

This frontend is a **lightweight consumer** in a decoupled microservice setup. It collects user inputs, sends them via HTTP POST to a FastAPI backend, and renders the returned JSON. It must run and demo correctly **even when the backend is offline**.

### 3.1 🆕 API Contract — Request Payload
```json
{
  "train_number": "12002",
  "fog_risk_score": 7,
  "congestion_index": 6,
  "incoming_rake_delay": true
}
```

### 3.2 🆕 API Contract — Response Payload
Both the mock function and the real FastAPI call must return **exactly** this shape, so swapping one for the other is a one-line change:

```json
{
  "train_number": "12002",
  "train_name": "Vande Bharat Express",
  "status": "DELAYED",
  "delay_probability": 78.5,
  "predicted_delay_minutes": 42,
  "risk_level": "HIGH",
  "route_progress_percentage": 63,
  "factors": [
    {
      "name": "Weather Factor",
      "description": "Significant Fog Risk on the route",
      "impact": "HIGH"
    },
    {
      "name": "Network Congestion",
      "description": "Nagpur Division Alert — heavy traffic",
      "impact": "MEDIUM"
    },
    {
      "name": "Incoming Rake Delay",
      "description": "Rake arrived 15 minutes late from previous run",
      "impact": "LOW"
    }
  ],
  "last_updated": "2026-07-08T14:32:00Z"
}
```
`status` ∈ `{ON_TIME, DELAYED}` · `risk_level`/`impact` ∈ `{LOW, MEDIUM, HIGH}`

### 3.3 🆕 Failure Modes & Resilience
- Wrap the POST call in `try/except` for `requests.exceptions.Timeout`, `ConnectionError`, and generic `RequestException`.
- Use an explicit `timeout=5` (seconds) on every request — never let the UI hang indefinitely.
- On failure, don't crash: fall back to mock data and show a small, on-brand warning banner — *"⚠️ Delay Oracle unreachable — showing a simulated prediction."* Style it in saffron, but as an info strip, not a full-page error.
- Validate the response JSON has the expected keys before rendering; if malformed, same fallback path.

### 3.4 🆕 Config Management
- Read the API URL from an environment variable with a sensible default:
  ```python
  API_URL = os.getenv("YATRIGAAN_API_URL", "http://127.0.0.1:8000/predict")
  ```
- Add a **sidebar toggle**: `🔌 Demo Mode` vs `🔴 Live API Mode` — defaults to Demo Mode so the app is fully self-contained and presentable with zero backend running. This is a strong signal in a placement demo that you understand real-world fallback design.

---

## 4. Visual Identity & Design System

*(Your original palette and glassmorphism direction — kept exactly as specified.)*

| Role | Color | Hex |
|---|---|---|
| Primary Brand | Indian Railways Navy Blue | `#003366` |
| Accent / Gradient Layer | Light Blue | `#ADD8E6` |
| Base Background | Airy Off-White | `#F0F8FF` |
| Alert / High-Risk / Severe Delay | Vande Bharat Saffron | `#FF9933` |

- Style: Glassmorphism SaaS dashboard — frosted-glass cards (`backdrop-filter: blur()`), soft shadows, subtle border glow, rounded corners. No default Streamlit gray widgets — all inputs restyled via injected CSS.
- Header includes a minimalist, professional train icon/logo.

### 4.1 🆕 Typography System
Inject a Google Fonts pairing via `@import` in the CSS block — this alone is a big lever for "premium" perception:
- **Headings:** `Poppins` (600/700 weight) — modern, geometric, fits a transit-tech brand.
- **Body / Data / Labels:** `Inter` or `Roboto` — high legibility for numbers and small print.

### 4.2 🆕 Motion & Micro-interaction Tokens
- Input focus: smooth `box-shadow` glow transition, `~150ms ease-in-out`.
- Card hover: subtle `translateY(-2px)` lift + shadow deepen, `~200ms cubic-bezier(0.4,0,0.2,1)`.
- Gauge sweep: animate from 0 to final value on render rather than snapping instantly.

### 4.3 🆕 Iconography Approach
- Keep the app **single-file and self-contained**: use an inline SVG or emoji (🚆/🚄) for the header logo and status icons rather than external image assets that could go missing.
- Pair every color-coded risk signal with a redundant icon/text label (see §8) — never rely on color alone.

---

## 5. Layout & Components

### 5.1 Hero Input Panel *(original, + validation)*
Clean, well-spaced central form or sidebar with styled inputs and smooth focus indicators:
- **Train Number** — selection input.
  - 🆕 Constrain to a 5-digit numeric pattern (or a dropdown of sample trains like `12002 – Vande Bharat Express`, `12951 – Mumbai Rajdhani`) so the payload is always well-formed.
- **Fog/Weather Risk Score** — Slider, 0–10.
- **Zone Congestion Index** — Slider, 0–10.
- **Incoming Rake Delay** — Checkbox.

### 5.2 Cultural Loading State *(original, + accessibility)*
On clicking **"Predict Delay"**, show a custom-styled loading sequence mimicking a railway platform announcement:
> *"Kripya Dhyaan Dijiye... Querying the Delay Oracle."*

Optionally, a smooth CSS animation of a sleek Vande Bharat-style train crossing the loading zone.

- 🆕 Wrap this message in an `aria-live="polite"` region — a fitting technical parallel to an actual audio announcement, and genuinely improves screen-reader behavior.

### 5.3 The Digital Boarding Pass (Output Container) *(original, + status redundancy)*
A high-fidelity glassmorphism "Digital Journey Ticket":
- **Header:** Train Number + Name (e.g., `12002 Vande Bharat Express`) with a state icon — 🆕 pair the On-Time/Delayed icon with a text badge (`✅ ON TIME` / `⚠️ DELAYED`), not color alone.
- **Interactive Probability Gauge:** Circular dial via `plotly.graph_objects` (recommended for zero extra install friction) or `streamlit-echarts`. Sweeps smoothly from green (low risk) to saffron (high risk) based on `delay_probability`.
- **Visual Journey Timeline:** Horizontal progress line with a minimalist train icon positioned at `route_progress_percentage`.
- **Delay Factors Breakdown:** Elegant list rendering the `factors` array — highlight `impact: "HIGH"` items in saffron, `MEDIUM` in a muted amber, `LOW` in neutral navy/gray.

---

## 6. 🆕 Code Architecture & Modularity

Structure `app.py` around these functions (single file is fine, but keep responsibilities separated):

```python
def load_custom_css() -> None: ...          # inject all CSS once
def render_header() -> None: ...            # logo, title, subtitle
def render_input_panel() -> dict: ...        # returns the payload dict
def render_loading_state() -> None: ...      # cultural loading animation
def fetch_prediction_from_fastapi(payload: dict, demo_mode: bool = True) -> dict: ...
def render_gauge(probability: float) -> None: ...
def render_journey_timeline(progress_pct: float) -> None: ...
def render_factors_breakdown(factors: list[dict]) -> None: ...
def render_boarding_pass(data: dict) -> None: ...   # composes the above
def main() -> None: ...
```

- Use `st.session_state` to persist `prediction_data` across reruns, so adjusting a slider *after* predicting doesn't wipe the boarding pass until "Predict Delay" is clicked again.
- Include type hints and short docstrings on every function — expected in a placement-grade codebase.
- Follow PEP8; keep CSS in a single triple-quoted string or a separate constant, not scattered across the file.

---

## 7. Microservice Integration Hook

```python
def fetch_prediction_from_fastapi(payload: dict, demo_mode: bool = True) -> dict:
    """
    Sends the prediction payload to the FastAPI backend.
    Falls back to mock data if demo_mode is True or the request fails.
    """
```
- Uses `requests.post()` against `API_URL` (see §3.4), `timeout=5`.
- Mock response handling matches the schema in §3.2 exactly, so the UI functions standalone today and can be pointed at the live endpoint with no rendering-code changes.
- On any exception, log it (e.g., `st.toast` or a subtle banner) and return mock data rather than raising.

---

## 8. 🆕 Accessibility & Responsiveness

- Every saffron "high risk" indicator gets a matching icon (⚠️) or text label — not color-only signaling.
- Maintain sufficient contrast for navy text on the light-blue/off-white gradient (test against WCAG AA where practical).
- Add basic responsive rules (media queries in the injected CSS) so `st.columns` layouts don't visually collapse awkwardly on a narrow projector/laptop screen during a demo.

---

## 9. 🆕 Deliverables & Definition of Done

- `app.py` — the complete, runnable script.
- `requirements.txt` — e.g. `streamlit`, `requests`, `plotly` (or `streamlit-echarts`).
- Run instructions: `streamlit run app.py`.
- App must launch and be fully demoable with **zero backend running** (Demo Mode default `True`).

---

## 10. Optional Enhancements *(not mandatory — only if time permits)*

- 🆕 **Prediction History panel** — sidebar list of the last few predictions made in-session (from `st.session_state`), nice for a live demo narrative.
- 🆕 **Bilingual toggle (EN/HI)** for labels, fitting the app's cultural framing.
- 🆕 **Footer** — small disclaimer + credits, e.g. *"Predictions are indicative, based on a demo ML model. Built with Streamlit + FastAPI."* — reads well to a placement panel.
- 🆕 **Dark mode toggle** — only if it doesn't dilute focus from the core glassmorphism identity.
