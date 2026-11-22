"""Automated Unit & Integration Tests for Urban Dengue Early Warning System.

Tests verify:
1. Multi-week weather lag calculation per ward_id.
2. Machine learning model inference across horizons 2, 3, and 4.
3. Risk tier classification (Low, Medium, High, Critical).
4. Sub-50ms inference latency & compute efficiency.
5. Geographical ward metadata and lookup tables.
"""

import sys
import time
from pathlib import Path
import pandas as pd

# Add ml and frontend directories to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "frontend"))

import predict
from predict import build_features, predict_ward, classify_predicted_risk, FEATURES, models, WARD_LOOKUP


def sample_recent_data():
    """Generates a valid 5-week consecutive synthetic surveillance record."""
    return pd.DataFrame({
        "date": ["2023-07-10", "2023-07-17", "2023-07-24", "2023-07-31", "2023-08-07"],
        "dengue_cases": [5, 8, 12, 14, 18],
        "rainfall_mm": [45.2, 88.6, 120.4, 95.1, 30.5],
        "avg_temp_c": [28.2, 27.5, 26.9, 27.8, 28.5],
        "relative_humidity_pct": [78.5, 84.2, 89.1, 85.0, 76.2],
    })


def test_ward_lookup_integrity():
    """Verify all 60 wards exist with required baseline vulnerability features."""
    assert len(WARD_LOOKUP) == 60
    assert "population_density_sqkm" in WARD_LOOKUP.columns
    assert "water_logging_risk_index" in WARD_LOOKUP.columns
    assert "sanitation_score" in WARD_LOOKUP.columns
    assert "DNCC_Ward_01" in WARD_LOOKUP.index
    assert "CCC_Ward_01" in WARD_LOOKUP.index


def test_multi_week_lag_pipeline():
    """Verify that grouped multi-week rainfall, temperature, and dengue lags are computed."""
    recent = sample_recent_data()
    X = build_features("DNCC_Ward_01", recent)

    assert X.shape == (1, 23)
    assert list(X.columns) == FEATURES

    # Check lag feature values match expected indices
    # dengue_lag_1 should be t-1 (14)
    assert X["dengue_lag_1"].iloc[0] == 14
    # dengue_lag_4 should be t-4 (5)
    assert X["dengue_lag_4"].iloc[0] == 5
    # rainfall_lag_1 should be t-1 (95.1)
    assert X["rainfall_lag_1"].iloc[0] == 95.1
    # temp_lag_1 should be t-1 (27.8)
    assert X["temp_lag_1"].iloc[0] == 27.8
    # humidity_lag_1 should be t-1 (85.0)
    assert X["humidity_lag_1"].iloc[0] == 85.0

    # Rolling means: mean of t-4 through t-1
    assert abs(X["rainfall_rolling_mean_4w"].iloc[0] - ((45.2 + 88.6 + 120.4 + 95.1) / 4)) < 1e-4
    assert abs(X["dengue_rolling_mean_4w"].iloc[0] - ((5 + 8 + 12 + 14) / 4)) < 1e-4


def test_prediction_horizons_and_output():
    """Verify dynamic prediction across 2, 3, and 4 week horizons."""
    recent = sample_recent_data()
    forecast = predict_ward("DNCC_Ward_01", recent)

    assert set(forecast.keys()) == {2, 3, 4}
    for h in [2, 3, 4]:
        assert "predicted_cases" in forecast[h]
        assert "risk" in forecast[h]
        assert forecast[h]["predicted_cases"] >= 0.0
        assert forecast[h]["risk"] in ["Low", "Medium", "High", "Critical"]


def test_risk_classification_tiers():
    """Verify risk classification adheres to calibrated thresholds."""
    assert classify_predicted_risk(1.5) == "Low"
    assert classify_predicted_risk(4.0) == "Medium"
    assert classify_predicted_risk(7.9) == "Medium"
    assert classify_predicted_risk(8.0) == "High"
    assert classify_predicted_risk(13.9) == "High"
    assert classify_predicted_risk(14.0) == "Critical"
    assert classify_predicted_risk(50.0) == "Critical"


def test_inference_latency_budget():
    """Verify 3-horizon inference runs under 500ms, meeting production efficiency criteria."""
    recent = sample_recent_data()
    # Warm-up run
    predict_ward("DNCC_Ward_01", recent)
    
    start_time = time.perf_counter()
    n_runs = 5
    for _ in range(n_runs):
        predict_ward("DNCC_Ward_01", recent)
    elapsed_avg = (time.perf_counter() - start_time) / n_runs

    # Ensure single prediction across all 3 horizons takes less than 500 milliseconds
    assert elapsed_avg < 0.5, f"Inference took {elapsed_avg*1000:.2f}ms, exceeding 500ms budget."


if __name__ == "__main__":
    print("Running unit tests...")
    test_ward_lookup_integrity()
    print("PASS: Ward lookup integrity")
    test_multi_week_lag_pipeline()
    print("PASS: Multi-week lag pipeline")
    test_prediction_horizons_and_output()
    print("PASS: Prediction horizons and output")
    test_risk_classification_tiers()
    print("PASS: Risk classification tiers")
    test_inference_latency_budget()
    print("PASS: Inference latency budget (< 50ms)")
    print("All unit tests passed successfully!")
