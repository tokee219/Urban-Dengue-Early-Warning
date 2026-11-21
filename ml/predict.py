import json
import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

# Find the folder containing this Python file
ML_DIR = Path(__file__).resolve().parent


class PortableXGBPipeline:
    """Loads an XGBoost model saved as JSON plus a separate preprocessor,
    and offers .predict(X) like a normal scikit-learn pipeline."""

    def __init__(self, preprocessor_path, model_path):
        self.preprocessor = joblib.load(preprocessor_path)
        self.model = XGBRegressor()
        self.model.load_model(model_path)

    def predict(self, X):
        return self.model.predict(self.preprocessor.transform(X))


# Load model configuration
with open(ML_DIR / "feature_config.json", "r") as file:
    feature_config = json.load(file)

# Load risk thresholds
with open(ML_DIR / "risk_thresholds.json", "r") as file:
    risk_thresholds = json.load(file)

# Load the three trained models
models = {
    2: joblib.load(ML_DIR / "dengue_2w_histgradientboosting.joblib"),
    3: PortableXGBPipeline(
        ML_DIR / "dengue_3w_preprocessor.joblib",
        ML_DIR / "dengue_3w_xgboost_model.json",
    ),
    4: joblib.load(ML_DIR / "dengue_4w_random_forest.joblib"),
}

# Ward-level fixed values (one row per ward)
WARD_LOOKUP = pd.read_csv(ML_DIR / "ward_lookup.csv").set_index("ward_id")
FEATURES = feature_config["features"]


def classify_predicted_risk(predicted_cases):
    """Turn predicted weekly cases into Low / Medium / High / Critical."""
    if predicted_cases >= risk_thresholds["critical"]:
        return "Critical"
    elif predicted_cases >= risk_thresholds["high"]:
        return "High"
    elif predicted_cases >= risk_thresholds["medium"]:
        return "Medium"
    return "Low"


def build_features(ward_id, recent):
    """Build the 23 model inputs for ONE ward.

    recent: a table of the ward's latest 5 consecutive weekly records,
    oldest first, with columns:
        date, dengue_cases, rainfall_mm, avg_temp_c, relative_humidity_pct
    The LAST row is the current week (t). The dengue count in that last
    row is not used by the model (training used dengue counts up to t-1).
    """
    if ward_id not in WARD_LOOKUP.index:
        raise ValueError(f"Unknown ward_id: {ward_id}")

    needed = ["date", "dengue_cases", "rainfall_mm", "avg_temp_c", "relative_humidity_pct"]
    missing = [c for c in needed if c not in recent.columns]
    if missing:
        raise ValueError(f"Missing columns in recent data: {missing}")

    recent = recent.sort_values("date").reset_index(drop=True)
    if len(recent) != 5:
        raise ValueError("Provide exactly the latest 5 weekly records (t-4 ... t).")
    gaps = pd.to_datetime(recent["date"]).diff().dropna().dt.days
    if not (gaps == 7).all():
        raise ValueError("The 5 records must be consecutive weeks (7 days apart).")
    if recent[["rainfall_mm", "avg_temp_c", "relative_humidity_pct"]].isna().any().any():
        raise ValueError("Weather values must not be missing.")
    if recent["dengue_cases"].iloc[:4].isna().any():
        raise ValueError("Dengue counts for weeks t-4 ... t-1 must not be missing.")

    last = recent.iloc[-1]
    ward = WARD_LOOKUP.loc[ward_id]

    row = {
        "ward_id": ward_id,
        "population_density_sqkm": ward["population_density_sqkm"],
        "water_logging_risk_index": ward["water_logging_risk_index"],
        "sanitation_score": ward["sanitation_score"],
        "avg_temp_c": last["avg_temp_c"],
        "rainfall_mm": last["rainfall_mm"],
        "relative_humidity_pct": last["relative_humidity_pct"],
        "dengue_rolling_mean_4w": recent["dengue_cases"].iloc[-5:-1].mean(),
        "rainfall_rolling_mean_4w": recent["rainfall_mm"].iloc[-5:-1].mean(),
    }
    for k in range(1, 5):
        row[f"dengue_lag_{k}"] = recent["dengue_cases"].iloc[-1 - k]
        row[f"rainfall_lag_{k}"] = recent["rainfall_mm"].iloc[-1 - k]
    for k in range(1, 4):
        row[f"temp_lag_{k}"] = recent["avg_temp_c"].iloc[-1 - k]
        row[f"humidity_lag_{k}"] = recent["relative_humidity_pct"].iloc[-1 - k]

    return pd.DataFrame([row])[FEATURES]


def predict_ward(ward_id, recent):
    """Predicted weekly dengue cases and risk level for 2, 3 and 4 weeks ahead."""
    X = build_features(ward_id, recent)
    result = {}
    for horizon, model in models.items():
        cases = max(float(model.predict(X)[0]), 0.0)   # clip negatives at zero
        result[horizon] = {
            "predicted_cases": round(cases, 2),
            "risk": classify_predicted_risk(cases),
        }
    return result


if __name__ == "__main__":
    print("Models loaded successfully!")
    print("Available prediction horizons:", list(models.keys()))
    print("Number of input features:", len(FEATURES))
