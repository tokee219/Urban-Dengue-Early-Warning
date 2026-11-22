"""Data loading and caching for the dashboard.

Compute and cost note: no LLM calls and no external API calls at runtime.
The only network use is an optional one-time download of the dataset.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "ml"
DATA_PATH = ROOT / "data" / "hackathon_preli_dataset.xlsx"
SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1QFI0ACJRvJQkiQT3LfGHklM3g4PgaJWUt8xSVU07NIE/export?format=xlsx"
)
REQUIRED_COLS = [
    "date", "city", "ward_id", "population_density_sqkm",
    "water_logging_risk_index", "sanitation_score", "avg_temp_c",
    "max_temp_c", "min_temp_c", "rainfall_mm",
    "relative_humidity_pct", "dengue_cases",
]
MODEL_INPUT_COLS = [
    "date", "dengue_cases", "rainfall_mm", "avg_temp_c", "relative_humidity_pct",
]


class DataUnavailable(Exception):
    """Raised with a friendly message the UI can show."""


def _try_download() -> bool:
    """One-time download of the public sheet. Never raises."""
    try:
        import requests

        resp = requests.get(SHEET_URL, timeout=30)
        resp.raise_for_status()
        if not resp.content.startswith(b"PK"):  # real .xlsx files are zip files
            return False
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        DATA_PATH.write_bytes(resp.content)
        return True
    except Exception:
        return False


@st.cache_data(show_spinner="Loading dataset records...")
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists() and not _try_download():
        raise DataUnavailable(
            "The dataset file was not found and the automatic download failed.\n\n"
            "Manual steps: open the dataset Google Sheet, choose File > Download > "
            "Microsoft Excel (.xlsx), and save it as "
            "`data/hackathon_preli_dataset.xlsx` in the repo root. "
            "Then refresh this page."
        )
    try:
        df = pd.read_excel(DATA_PATH)
        df["date"] = pd.to_datetime(df["date"])
    except Exception as exc:
        raise DataUnavailable(f"The dataset file could not be read: {exc}") from exc
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise DataUnavailable(f"The dataset is missing columns: {missing}")
    return df.sort_values(["ward_id", "date"]).reset_index(drop=True)


@st.cache_resource(show_spinner="Loading models...")
def import_predict():
    """Import ml/predict.py once per server. Models load at import time."""
    if not (ML_DIR / "ward_lookup.csv").exists():
        raise DataUnavailable(
            "`ml/ward_lookup.csv` is missing. Copy it into the `ml/` folder "
            "and refresh this page."
        )
    if str(ML_DIR) not in sys.path:
        sys.path.insert(0, str(ML_DIR))
    try:
        import predict
    except Exception as exc:
        raise DataUnavailable(
            f"The prediction module could not be loaded: {exc}\n\n"
            "Check that `pip install -r requirements.txt` finished without errors."
        ) from exc
    return predict


@st.cache_data
def ward_table() -> pd.DataFrame:
    """ward_id, city, and static vulnerability metrics for all 60 wards."""
    return pd.read_csv(ML_DIR / "ward_lookup.csv")


@st.cache_data
def ward_coords() -> dict:
    """Geographical coordinates and friendly neighborhood names for all 60 wards."""
    import json
    path = ML_DIR / "ward_coords.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def cities() -> list:
    return sorted(ward_table()["city"].unique().tolist())


def wards_for_city(city: str) -> list:
    t = ward_table()
    return sorted(t.loc[t["city"] == city, "ward_id"].tolist())


@st.cache_data
def available_origins(_df: pd.DataFrame) -> list:
    """Selectable current weeks: from the 5th week of the data onward."""
    return sorted(_df["date"].unique())[4:]


def get_recent(df: pd.DataFrame, ward_id: str, origin, overrides: dict = None) -> pd.DataFrame:
    """The exact 5 weekly rows (oldest first) that predict_ward expects.
    
    Supports both historical dataset dates and current/live dates (e.g. today).
    If `origin` is a current/custom date, dynamically aligns the 5 weekly timestamps
    ending at `origin` with exact 7-day intervals.
    """
    from datetime import timedelta
    origin = pd.Timestamp(origin)
    w = df[(df["ward_id"] == ward_id) & (df["date"] <= origin)]
    if len(w) < 5:
        # If earlier than 5 weeks or beyond bounds, fall back to available records for ward
        w = df[df["ward_id"] == ward_id]
        if len(w) < 5:
            raise DataUnavailable(f"Ward {ward_id} has fewer than 5 records in database.")
    
    recent = w.tail(5)[MODEL_INPUT_COLS].copy().reset_index(drop=True)
    
    # Ensure exact 7-day consecutive timeline ending at origin
    target_dates = [origin - timedelta(days=7 * (4 - i)) for i in range(5)]
    recent["date"] = target_dates

    # Apply any live field overrides (e.g., current week weather or case inputs)
    if overrides:
        for col, val in overrides.items():
            if col in recent.columns and val is not None:
                recent.loc[recent.index[-1], col] = float(val)

    return recent


def get_history(df: pd.DataFrame, ward_id: str, origin, weeks: int = 16) -> pd.DataFrame:
    """Last `weeks` weekly rows of one ward, up to and including origin.
    
    Maintains timeline continuity when origin is set to today's current date.
    """
    from datetime import timedelta
    origin = pd.Timestamp(origin)
    max_data_date = df["date"].max()

    w = df[(df["ward_id"] == ward_id) & (df["date"] <= origin)]
    if len(w) < 5:
        w = df[df["ward_id"] == ward_id]
    
    hist = w.tail(weeks)[MODEL_INPUT_COLS + ["min_temp_c", "max_temp_c"]].copy().reset_index(drop=True)

    # If origin is beyond dataset date, adjust history timeline to end at origin
    if origin > max_data_date:
        n = len(hist)
        hist["date"] = [origin - timedelta(days=7 * (n - 1 - i)) for i in range(n)]

    return hist