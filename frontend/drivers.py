"""Weather-driver analysis: counterfactual (what-if) sensitivity of the model.

No external API calls and no LLM calls. Uses only the trained models.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_access import MODEL_INPUT_COLS

ROOT = Path(__file__).resolve().parents[1]

DRIVER_GROUPS = {
    "Rainfall pattern": [
        "rainfall_mm", "rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3",
        "rainfall_lag_4", "rainfall_rolling_mean_4w",
    ],
    "Temperature": ["avg_temp_c", "temp_lag_1", "temp_lag_2", "temp_lag_3"],
    "Humidity": [
        "relative_humidity_pct", "humidity_lag_1", "humidity_lag_2", "humidity_lag_3",
    ],
    "Recent dengue cases": [
        "dengue_lag_1", "dengue_lag_2", "dengue_lag_3", "dengue_lag_4",
        "dengue_rolling_mean_4w",
    ],
    "Ward baseline (fixed)": [
        "population_density_sqkm", "water_logging_risk_index", "sanitation_score",
    ],
}

DRIVER_NOTE = (
    "How to read this: each bar shows how much the forecast changes when that "
    "group of inputs is replaced by a typical (median) value. It is a what-if "
    "sensitivity of the trained model. The bars do not need to add up to the "
    "forecast, and they do not show cause and effect."
)


@st.cache_data(show_spinner=False)
def feature_medians(_predict, _df, n_windows: int = 1500, seed: int = 42) -> dict:
    """Median of every model input over many dataset windows.

    Loads from precomputed ml/feature_medians.json if available for instant startup.
    """
    precomputed = ROOT / "ml" / "feature_medians.json"
    if precomputed.exists():
        import json
        try:
            return json.loads(precomputed.read_text(encoding="utf-8"))
        except Exception:
            pass
    rng = np.random.default_rng(seed)
    wards = {w: g.reset_index(drop=True) for w, g in _df.groupby("ward_id")}
    ids = sorted(wards)
    rows = []
    for _ in range(n_windows):
        w = ids[int(rng.integers(len(ids)))]
        g = wards[w]
        end = int(rng.integers(4, len(g)))
        recent = g.iloc[end - 4:end + 1][MODEL_INPUT_COLS].reset_index(drop=True)
        rows.append(_predict.build_features(w, recent))
    X = pd.concat(rows, ignore_index=True).select_dtypes("number")
    return {c: float(X[c].median()) for c in X.columns}


@st.cache_data(show_spinner=False)
def driver_impacts(_predict, recent: pd.DataFrame, ward_id: str,
                   horizon: int, medians: dict):
    """Return (impacts table, base prediction) for one ward and horizon.

    impact = base prediction - prediction with that group set to its median.
    Positive means the group pushes the forecast up vs a typical week.
    """
    X = _predict.build_features(ward_id, recent)
    model = _predict.models[horizon]
    base = max(float(model.predict(X)[0]), 0.0)
    rows = []
    for label, cols in DRIVER_GROUPS.items():
        cols = [c for c in cols if c in X.columns and c in medians]
        X2 = X.copy()
        for c in cols:
            X2[c] = medians[c]
        new = max(float(model.predict(X2)[0]), 0.0)
        rows.append({"driver": label, "impact": base - new})
    return pd.DataFrame(rows), base


def driver_sentence(impacts: pd.DataFrame) -> str:
    top = impacts.loc[impacts["impact"].abs().idxmax()]
    if abs(top["impact"]) < 0.25:
        return ("For this week, no input group moves the forecast much "
                "compared with a typical week.")
    direction = "up" if top["impact"] > 0 else "down"
    return (
        f"For this week, the model's forecast is most sensitive to "
        f"**{top['driver'].lower()}**, which is associated with a forecast "
        f"pushed {direction} by about {abs(top['impact']):.1f} cases "
        f"compared with a typical week."
    )


def driver_figure(impacts: pd.DataFrame) -> go.Figure:
    d = impacts.sort_values("impact")
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in d["impact"]]
    fig = go.Figure(go.Bar(
        x=d["impact"], y=d["driver"], orientation="h", marker_color=colors,
        text=[f"{v:+.2f}" for v in d["impact"]], textposition="outside",
    ))
    fig.update_layout(
        height=300, margin=dict(l=10, r=30, t=10, b=10),
        xaxis_title="Change in forecast vs a typical week (cases)",
    )
    fig.add_vline(x=0, line_width=1, line_color="gray")
    return fig