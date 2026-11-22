# 🦟 Urban Dengue Early Warning System
### AI-Powered Hyper-Local Outbreak Prediction & Municipal Early-Warning Engine
**EDU AI Engineering Hackathon 2026 — Phase 1: 24-Hour Remote Qualifier**  
**Challenge:** AI for Urban Dengue Outbreak  
**Target Metropolitan Areas:** Dhaka North (DNCC), Dhaka South (DSCC), Chattogram (CCC), Gazipur (GCC), Narayanganj (NCC) — 60 Urban Wards

---

## 📌 Executive Overview
In urban Bangladesh, dengue epidemics overwhelm hospitals every year. When city corporations deploy larviciding and fogging teams only after hospital admissions surge, transmission chains are already weeks ahead of containment.

The **Urban Dengue Early Warning System** operationalizes the biological **Vector Breeding Clock**:
- **7 to 14 Days:** Rainfall stagnation in drains and discarded receptacles incubates *Aedes aegypti* eggs into adult mosquitoes.
- **8 to 12 Days:** Extrinsic incubation period (viral replication inside the mosquito) followed by human clinical incubation.
- **2 to 4 Weeks:** The critical lag between heavy rainfall events and hospital emergency admission spikes.

By computing multi-week grouped weather lags and integrating hyper-local ward structural vulnerabilities (`water_logging_risk_index`, `population_density_sqkm`, `sanitation_score`), this system provides public health officials with an actionable **2-to-4-week early warning window** to eradicate mosquito larvae *before* outbreaks peak.

---

## 🚀 Quickstart: Reproducible Local Setup (≤ 3 Minutes)

This application runs 100% locally with zero cloud dependencies. Follow these steps to clone and launch in under 3 minutes:

### 1. Clone the Repository
```bash
git clone https://github.com/tokee219/Urban-Dengue-Early-Warning.git
cd Urban-Dengue-Early-Warning
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Interactive Dashboard
```bash
streamlit run frontend/app.py
```
The dashboard opens automatically in your browser at `http://localhost:8501`.

### 5. Run Automated Test Suite
Verify model inference, lag computations, and latency benchmarks:
```bash
python tests/test_pipeline.py
```

---

## 🎯 Key Capabilities & Domain Innovations

### 1. Multi-Week Grouped Weather Lag Pipeline
- Ingests weekly `rainfall_mm`, `avg_temp_c`, `min_temp_c`, `max_temp_c`, and `relative_humidity_pct`.
- Computes grouped lags ($t-1$ to $t-4$), 4-week rolling means, and autoregressive dengue lags per ward.
- Features are strictly backward-looking: contemporaneous dengue cases at week $t$ are excluded to prevent real-world data leakage.

### 2. Multi-Horizon Machine Learning Engine
- **2 Weeks Ahead ($t+14d$):** `HistGradientBoostingRegressor` (Test MAE: 1.07 vs Persistence: 1.87 — **42.7% error reduction**)
- **3 Weeks Ahead ($t+21d$):** `XGBoost Regressor` with ColumnTransformer (Test MAE: 1.15 vs Persistence: 2.06 — **44.2% error reduction**)
- **4 Weeks Ahead ($t+28d$):** `RandomForestRegressor` (Test MAE: 1.32 vs Persistence: 2.22 — **40.5% error reduction**)
- Calibrated outbreak risk tiers: **Low** (<4 cases), **Medium** (4–7.9 cases), **High** (8–13.9 cases), and **Critical** (≥14 cases).

### 3. Public Health "What-If" Scenario Simulator
- Public health directors can toggle **What-If Simulation** to model sudden monsoon rainfall surges (+20% to +200%) or temperature shifts.
- The pipeline re-evaluates multi-week lag impacts in real-time, forecasting downstream vector outbreaks weeks ahead.

### 4. High-Impact Domain Bonus Extensions (Implemented)
- ⭐ **Bonus 1: Interactive Ward-Level Risk Heatmap:** Dynamic OpenStreetMap spatial scatter visualization with ward centroids across Dhaka North, Dhaka South, Chattogram, Gazipur, and Narayanganj. Bubbles are color-coded by risk tier and sized by predicted case volume, accompanied by an actionable ward prioritization leaderboard.
- ⭐ **Bonus 2: Multi-Week Forecast Confidence Intervals:** 90% empirical prediction intervals calculated from out-of-sample validation residual quantiles ($p_{05}$ to $p_{95}$) rendered as uncertainty ribbons.
- ⭐ **Bonus 3: Chronological Validation & Model Performance Dashboard:** Interactive validation tab displaying chronological split rationale, multi-year rolling origin out-of-time evaluations (2019–2025), and ablation studies showing model error jumps by 96.7% if weather features are omitted.

---

## 💡 Compute & API Cost Efficiency Statement

- **Fast Local Inference:** Serialized `.joblib` and XGBoost JSON models load once via `@st.cache_resource` and execute predictions in **< 5 ms per ward** on CPU.
- **Zero LLM Token Costs:** The platform does not use costly, slow, or hallucinating LLM APIs. All driver sensitivities and recommendations are calculated deterministically via counterfactual algorithms.
- **Zero Paid External API Calls:** 100% self-contained surveillance records; requires no monthly recurring weather API subscriptions.
- **Precomputed Medians:** Feature medians are serialized in `ml/feature_medians.json` to deliver instant sensitivity driver breakdowns without runtime Monte Carlo overhead.

---

## 🤖 AI Disclosure & External API Statement
*(Required under Hackathon Scoring Criterion 8)*

- **AI Assistant Disclosure:**
  - **Google DeepMind Antigravity AI:** Used during development for architecture structuring, code refactoring, test pipeline generation, coordinate mapping, and technical documentation drafting.
- **External APIs:**
  - **None at Runtime.** The system requires **0 external paid APIs** and **0 LLM API calls**.
  - All geospatial map tiles render via free, open-source OpenStreetMap layers (`open-street-map`) in Plotly without requiring proprietary Mapbox tokens.

---

## 🎥 3-Minute Uncut Video Demo Walkthrough Guide
*(Required under Hackathon Scoring Criterion 7 — Uncut screen recording ≤ 3:00 mins with voiceover)*

A sample walkthrough sequence for recording the uncut video demo:
- **0:00 – 0:30 | Introduction & Local Startup:** Show terminal launching `streamlit run frontend/app.py` in ≤ 3 seconds. State problem statement: the 2-to-4-week Vector Breeding Clock in Dhaka & Chattogram.
- **0:30 – 1:15 | Ward Early Warning Outlook & Drivers:** Select `Dhaka North` -> `DNCC_Ward_01`. Point out the 2-week, 3-week, and 4-week forecast cards with 90% confidence intervals and risk badges. Show the historical rainfall vs case lag trajectory chart. Explain the weather driver sensitivity waterfall bar chart.
- **1:15 – 1:55 | What-If Public Health Simulator:** Toggle "Enable What-If Weather Shock". Increase rainfall slider to 1.5x (+50% monsoon surge) and show in real-time how predicted cases jump into High/Critical risk tiers 2-4 weeks down the line.
- **1:55 – 2:30 | City-Wide Spatial Risk Map & Leaderboard (Bonus):** Switch to Tab 2. Show the OpenStreetMap interactive heatmap of all 20 wards in Dhaka North. Switch horizons from 2 to 4 weeks. Highlight the sortable High-Risk Ward Prioritization Leaderboard.
- **2:30 – 3:00 | Validation Dashboard & Cost Efficiency (Bonus):** Switch to Tab 3. Highlight the chronological out-of-time split (2016-2022 train, 2023 val, 2024-2025 test), 42%+ MAE improvement over persistence, ablation proof, and conclude with zero-cost offline local execution.

---

## 📁 Repository Directory Structure

```
Urban-Dengue-Early-Warning/
├── ARCHITECTURE.md          # Comprehensive 1-2 page architecture and validation design
├── README.md                # Project documentation, quickstart, AI disclosure & demo script
├── requirements.txt         # Clean, pinned dependencies for reproducible setup
├── backend/                 # Backend modules and API extensions
├── data/
│   └── hackathon_preli_dataset.xlsx  # 10-year weekly dengue & weather dataset (60 wards)
├── frontend/
│   ├── app.py               # Streamlit interactive public health early-warning dashboard
│   ├── data_access.py       # Cached data loader, coordinate lookup, and window extraction
│   └── drivers.py           # Counterfactual weather sensitivity and driver decomposition
├── ml/
│   ├── dengue_2w_histgradientboosting.joblib # Trained 2-week forecast model
│   ├── dengue_3w_preprocessor.joblib         # ColumnTransformer preprocessor
│   ├── dengue_3w_xgboost_model.json          # Trained 3-week XGBoost model
│   ├── dengue_4w_random_forest.joblib        # Trained 4-week Random Forest model
│   ├── feature_config.json                   # 23-feature schema definition
│   ├── feature_medians.json                  # Precomputed feature medians for fast drivers
│   ├── model_metrics.json                    # Full chronological validation & ablation metrics
│   ├── predict.py                            # Production prediction interface & lag pipeline
│   ├── risk_thresholds.json                  # Calibrated risk category cutoffs
│   ├── ward_coords.json                      # Centroid coordinates for all 60 wards
│   └── ward_lookup.csv                       # Ward baseline vulnerability indices
├── notebooks/
│   └── 01_dataset_exploration.ipynb          # Exploratory analysis & model development
└── tests/
    └── test_pipeline.py     # Automated unit & integration tests (latency & lag verification)
```

---

## 👥 Team & Submission Information
- **Event:** EDU AI Engineering Hackathon 2026
- **Submission:** AI for Urban Dengue Outbreak Early Warning Platform
