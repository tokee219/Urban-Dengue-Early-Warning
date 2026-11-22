# Urban Dengue Early Warning System

Ward-level, 2-to-4-week dengue early-warning prototype for five Bangladeshi city corporations
(Dhaka North, Dhaka South, Chattogram, Gazipur, Narayanganj; 60 wards).

Built for the **EDU AI Engineering Hackathon 2026, Phase 1 (24-hour remote qualifier)**,
challenge *AI for Urban Dengue Outbreak*.

> **Status: decision-support prototype.** It is not clinically validated, not an official
> surveillance system, and not approved by any public-health authority. Risk tiers are
> model-defined statistical bands, not official alert thresholds. See [Limitations](#limitations).

---

## What it does

Pick a ward and a "current week". The system uses that ward's recent weather and case history to
predict dengue cases **2, 3 and 4 weeks ahead**, labels each forecast **Low, Medium, High or
Critical**, and shows which weather inputs the forecast is most sensitive to.

Why a lag-based model: rain does not cause infections overnight. Standing water, mosquito
development and virus incubation create a delay of roughly 2 to 4 weeks between rainfall and
hospital cases (see the hackathon brief). The model therefore uses multi-week, per-ward weather
lags instead of same-week weather.

## Quickstart

Requirements: Python 3.12 or 3.13 and an internet connection for the first install.

```bash
git clone https://github.com/tokee219/Urban-Dengue-Early-Warning.git
cd Urban-Dengue-Early-Warning

# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run frontend/app.py
```

The dashboard opens at <http://localhost:8501>. Installing the dependencies took about 20 seconds
in a clean environment during our checks.

**Dataset.** The dataset is provided by the hackathon organizers and is not stored in this
repository. On first launch the app tries to download it once from the public Google Sheet linked
in the brief and saves it as `data/hackathon_preli_dataset.xlsx`. If that download fails, open the
sheet, choose *File, Download, Microsoft Excel (.xlsx)*, and save the file under that exact name.

**Pinned versions matter.** `requirements.txt` pins exact versions on purpose. The saved model
files were trained with scikit-learn 1.6.1, and newer scikit-learn releases (for example 1.9)
cannot load them.

**Tests.**

```bash
python tests/test_pipeline.py
```

This checks the lag features, model inference at all three horizons, the risk tiers and the
inference time budget.

## Data

| Item | Value |
|---|---|
| Source | Hackathon organizers' dataset (link in the brief) |
| Size | 31,320 weekly rows, 12 columns, 60 wards, 5 cities |
| Period | 2016-01-04 to 2025-12-29 (about 522 weeks per ward) |
| Weather columns | `avg_temp_c`, `max_temp_c`, `min_temp_c`, `rainfall_mm`, `relative_humidity_pct` |
| Ward columns (fixed per ward) | `population_density_sqkm`, `water_logging_risk_index`, `sanitation_score` |
| Target | `dengue_cases` (weekly count per ward) |
| Quality | No missing values or duplicate rows; about 58% of weekly counts are zero |

**Provenance.** We have not independently verified whether these records are real reported
cases, simulated data, or generated data. In this project they are called *dataset records*, and
all results describe performance on this dataset only.

## How it works

```text
weekly weather + cases (per ward)
        |
        v
per-ward lag features  ---- rainfall lags 1-4 weeks, temperature and humidity lags 1-3 weeks,
        |                   dengue lags 1-4 weeks, 4-week rolling means, ward ID and
        |                   ward vulnerability values (23 model inputs in total)
        v
horizon-specific model (2, 3 or 4 weeks ahead), predictions clipped at zero
        |
        v
risk tier (Low / Medium / High / Critical)  ->  Streamlit dashboard
```

All lags and rolling means are computed within each `ward_id` and only look backwards. The model
uses weather up to the current week and dengue counts up to the previous week.

### Models

| Horizon | Model | Test MAE | Test RMSE | Persistence MAE | MAE reduction |
|---|---|---:|---:|---:|---:|
| 2 weeks | HistGradientBoosting | 1.07 | 2.09 | 1.87 | 42.5% |
| 3 weeks | XGBoost | 1.15 | 2.26 | 2.06 | 44.1% |
| 4 weeks | Random Forest | 1.32 | 2.50 | 2.22 | 40.5% |

MAE and RMSE are errors in cases per ward per week on the 2024-2025 test period. They are not
accuracy percentages. "Persistence" means predicting the most recent weekly count.

Model choice: Random Forest and XGBoost were compared at all three horizons on validation data;
at 2 weeks HistGradientBoosting and a small neural network (MLP) were also tried. The neural
network was clearly worse. At 2 weeks, HistGradientBoosting and XGBoost were tied within noise
(validation MAE 1.198 and 1.207), and the lower validation error was used to choose between them.
Models are serialized with `joblib` (the 3-week XGBoost model is stored as JSON plus a separate
preprocessor file), so predictions need no retraining.

### Validation strategy (chronological)

The split is by the date of the *target* week, so a model is never trained on weeks later than the
ones it is evaluated on.

| Set | Target dates | Rows per horizon |
|---|---|---:|
| Train | up to 2022-12-31 | about 21,500 |
| Validation | 2023 | 3,120 |
| Test | 2024-01-01 to 2025-12-29 | 6,300 |

Risk cutoffs were chosen on validation data only; the test set was used for reporting.

Further checks:

- **Rolling-origin evaluation** (train on all earlier years, test on 2019, 2020, ... 2025): the
  2-week model beats the persistence baseline in all seven years, with 40-55% lower MAE.
- **Seasonal baselines:** a seasonal-average rule (mean of three weeks one year earlier) scores
  MAE 1.44 on the test period, against 1.07 for the deployed 2-week model.
- **Feature-removal tests** (2-week model, validation): removing all weather features roughly
  doubles the error (+97%); removing the dengue-history features changes it by under 1%. The
  predictions are therefore driven mainly by weather and the ward's fixed profile, not by recent
  case counts. This is an unusual pattern for real surveillance data and is one reason we make no
  real-world claims.

### Risk tiers

Applied to the predicted number of weekly cases in a ward:

| Tier | Predicted weekly cases |
|---|---|
| Low | below 4 |
| Medium | 4 to below 8 |
| High | 8 to below 14 |
| Critical | 14 or more |

The High and Critical cutoffs (8 and 14) were chosen on 2023 validation data to balance catching
high-case weeks against false alarms, because regression models tend to under-predict peaks. The
Medium cutoff of 4 is the original, untuned band.

**What the tiers mean in practice (test period, 2-week to 4-week models):** a High-or-above alert
caught about 72-75% of the weeks that really had 10 or more cases, but only about 40-46% of such
alerts were correct. The test years were quiet (only 15 weeks with 20 or more cases), so results
for Critical are statistically fragile.

## Dashboard

Run with `streamlit run frontend/app.py`. Four tabs:

1. **Ward 2-4 week outlook.** Forecast cards for all three horizons with tier badges, a trend
   chart with an approximate range (5th to 95th percentile of 2023 validation errors; it is not a
   formal confidence interval), a rainfall chart, the weather-driver chart, and a *what-if*
   simulator that rescales rainfall, temperature or humidity to see how the model's forecast
   responds.
2. **City-wide spatial risk map.** Predicted risk for every ward of a city, plus a ranked table.
3. **Validation and benchmark metrics.** Test errors against baselines, the rolling-origin
   results and the feature-removal tests, read from `ml/model_metrics.json`.
4. **Vector control action protocol.** Example actions for each tier. This is illustrative
   text, not official guidance; check any real use with public-health authorities.

**Weather-driver method.** For a chosen forecast, each group of inputs (rainfall pattern,
temperature, humidity, recent cases, fixed ward profile) is replaced in turn by its dataset-wide
median value, and the change in the forecast is measured. This shows what the trained model is
sensitive to. It is not evidence of causation, and the effects do not have to add up to the
prediction.

**Map note.** The dataset has no coordinates. The positions and neighbourhood names in
`ml/ward_coords.json` are approximate, assigned for visualization only, and do not represent
official ward boundaries. Map tiles are loaded from OpenStreetMap by the browser, so the map needs
an internet connection.

**Demo mode.** There is no live weather feed. Every forecast uses records from the dataset for
the week you select.

## Compute and cost

- Models and data are loaded once per server run (`st.cache_resource`, `st.cache_data`), and city
  scoring is done in one vectorized pass instead of one call per ward.
- Measured prediction time for one ward, all three horizons including feature building: about
  15 ms on an ordinary CPU (single-row model calls take roughly 3-7 ms each).
- No LLM calls and no paid APIs. The only network use is the one-time dataset download (if the
  file is missing) and the map tiles in the browser.

## Repository structure

```text
Urban-Dengue-Early-Warning/
  README.md                      this file
  ARCHITECTURE.md                architecture, data flow, validation split, model rationale
  requirements.txt               pinned dependencies
  data/                          dataset goes here (see Data)
  ml/
    predict.py                   feature building, model loading, prediction, risk tiers
    demo_predict.py              minimal example of calling predict_ward
    feature_config.json          the 23 model inputs and model choices
    risk_thresholds.json         risk-tier cutoffs
    model_metrics.json           validation and test metrics shown in the dashboard
    feature_medians.json         medians used by the weather-driver analysis
    ward_lookup.csv              fixed ward values (density, water-logging, sanitation)
    ward_coords.json             approximate ward positions for the map
    dengue_2w_histgradientboosting.joblib
    dengue_3w_preprocessor.joblib, dengue_3w_xgboost_model.json
    dengue_4w_random_forest.joblib
  frontend/
    app.py                       Streamlit dashboard
    data_access.py               cached data loading and weekly windows
    drivers.py                   weather-driver sensitivity analysis
  notebooks/                     exploration and model development
  tests/test_pipeline.py         lag, inference, risk-tier and latency tests
```

## Limitations

- Dataset provenance and licence are unverified; results describe this dataset only.
- Not clinically validated, not an official surveillance system.
- The test period (2024-2025) was quiet, so alert precision and Critical-tier results are
  uncertain; about half or more of High alerts were false alarms.
- Ward vulnerability values never change over time, so the model cannot learn how changing them
  would affect dengue, and wards outside the 60 cannot be scored.
- The model has no month or week-of-year input, so rainfall may partly stand in for season.
  Driver results show model sensitivity, not cause and effect.
- Alert cutoffs were tuned on a single validation year (2023).
- No live weather source; predictions use dataset weather.

## AI assistants and external services

AI assistants used during development:

- **Claude (Anthropic):** guidance and code for exploratory analysis, feature engineering, model
  training, chronological validation, evaluation, risk-tier selection, model export, the
  inference module (`ml/predict.py`), and the dashboard build specification.
- **Google DeepMind Antigravity:** architecture structuring, code refactoring, test and
  dashboard code generation, ward-coordinate mapping and documentation drafting.

All AI-generated code and text were reviewed and tested by the team.

External services and data:

- **No external APIs or LLM calls at prediction time.**
- Hackathon dataset (Google Sheets link from the brief), downloaded once if not present locally.
- OpenStreetMap map tiles, loaded by the browser for the map tab (no API key).
- Open-source libraries listed in `requirements.txt`.

## Team

Debug duo
