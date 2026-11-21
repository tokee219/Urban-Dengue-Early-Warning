"""Dengue Early-Warning Dashboard (Streamlit). Run: streamlit run app.py"""
import altair as alt
import pandas as pd
import streamlit as st

from src import config as C
from src.features import prepare
from src.predict import baseline_of, explain, load_bundle, score

st.set_page_config(page_title="Dengue Early Warning", page_icon="🦟", layout="wide")
TIER_SCALE = alt.Scale(domain=C.TIERS, range=[C.TIER_COLORS[t] for t in C.TIERS])


# ---------- cached backend: features + model + scoring run once, not per click ----------
@st.cache_data(show_spinner="Building multi-week lag features...")
def get_features(_mtime: float) -> pd.DataFrame:
    return prepare()


@st.cache_resource(show_spinner="Loading model...")
def get_bundle(_mtime: float) -> dict:
    return load_bundle()


@st.cache_data(show_spinner="Scoring all wards...")
def get_scored(data_mtime: float, model_mtime: float) -> pd.DataFrame:
    return score(get_bundle(model_mtime), get_features(data_mtime))


def badge(tier: str) -> str:
    return (f"<span style='background:{C.TIER_COLORS[tier]};color:white;padding:4px 14px;"
            f"border-radius:14px;font-weight:700;font-size:1.3rem'>{tier}</span>")


ACTIONS = {
    "Critical": "Deploy larvicide + spraying teams now; alert nearest hospitals to prepare beds, saline and test kits.",
    "High": "Schedule drain-clearing and larviciding this week; start community awareness drive.",
    "Medium": "Monitor; inspect known water-logging hotspots.",
    "Low": "Routine surveillance.",
}

# ---------- guard rails ----------
if not C.DATA_PATH.exists():
    st.error(f"Dataset not found at `{C.DATA_PATH}`. Run `python scripts/download_data.py` "
             "or save the CSV there (see README).")
    st.stop()
if not C.MODEL_PATH.exists():
    st.warning(f"No trained model at `{C.MODEL_PATH}`. Run `python train_baseline.py` "
               "(or drop your teammate's bundle there).")
    st.stop()

d_m, m_m = C.DATA_PATH.stat().st_mtime, C.MODEL_PATH.stat().st_mtime
bundle, scored = get_bundle(m_m), get_scored(d_m, m_m)
feats = get_features(d_m)
ready = scored[scored["complete"]]

# ---------- sidebar ----------
st.sidebar.title("🦟 Dengue Early Warning")
weeks = sorted(ready["week"].unique(), reverse=True)
last_obs = scored.loc[~scored["is_forecast"], "week"].max()
week = st.sidebar.selectbox(
    "Target week", weeks,
    format_func=lambda w: f"{pd.Timestamp(w):%d %b %Y}" + ("  (forecast)" if pd.Timestamp(w) > last_obs else ""),
)
wards = sorted(ready["ward_id"].unique())
ward = st.sidebar.selectbox("Ward", wards)
st.sidebar.caption(f"Each prediction uses weather observed ≥ {C.MIN_LAG} weeks before the target week, "
                   f"giving a {C.MIN_LAG}-week early-warning window.")
st.sidebar.caption(f"Last observed week: {last_obs:%d %b %Y}")

week_df = ready[ready["week"] == week]
row = week_df[week_df["ward_id"] == ward].iloc[0]
prev = ready[(ready["ward_id"] == ward) & (ready["week"] == pd.Timestamp(week) - pd.Timedelta(weeks=1))]

st.title(f"Ward {ward} - {pd.Timestamp(week):%d %b %Y}")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Predicted weekly cases", f"{row.pred_cases:.1f}",
          None if prev.empty else f"{row.pred_cases - prev.iloc[0].pred_cases:+.1f} vs prev. week", delta_color="inverse")
k2.markdown("**Risk tier**")
k2.markdown(badge(row.tier), unsafe_allow_html=True)
groups_df, feat_df = explain(bundle, row, baseline_of(bundle, feats))
top = groups_df.iloc[0]
k3.metric("Main driver", top.group, f"{top.effect_cases:+.2f} cases", delta_color="inverse")
k4.metric("Wards at High/Critical", int((week_df["tier_idx"] >= 2).sum()), f"of {len(week_df)}", delta_color="off")
st.info(f"**Recommended action:** {ACTIONS[row.tier]}")

tab1, tab2, tab3 = st.tabs(["🗺️ City overview", "🔍 Why this ward?", "📏 Model validation"])

# ---------- tab 1: overview ----------
with tab1:
    counts = week_df["tier"].value_counts().reindex(C.TIERS).fillna(0).astype(int)
    for col, t in zip(st.columns(4), C.TIERS):
        col.markdown(f"<div style='border-left:6px solid {C.TIER_COLORS[t]};padding-left:10px'>"
                     f"<b>{t}</b><br><span style='font-size:1.6rem'>{counts[t]}</span> wards</div>",
                     unsafe_allow_html=True)
    tbl = (week_df.sort_values("pred_cases", ascending=False)
           [["ward_id", "tier", "pred_cases", "water_logging_risk_index", "population_density_sqkm", "sanitation_score"]
            if all(c in week_df for c in C.STATIC_COLS) else ["ward_id", "tier", "pred_cases"]]
           .rename(columns={"ward_id": "Ward", "tier": "Risk tier", "pred_cases": "Predicted cases"}))
    styler = tbl.style.format({"Predicted cases": "{:.1f}"})
    color = lambda v: f"background-color:{C.TIER_COLORS.get(v, '')};color:white;font-weight:600"
    styler = (styler.map if hasattr(styler, "map") else styler.applymap)(color, subset=["Risk tier"])
    st.dataframe(styler, hide_index=True, height=320)
    st.download_button("⬇️ Download ward risk list (CSV)", tbl.to_csv(index=False), f"ward_risk_{pd.Timestamp(week):%Y%m%d}.csv")

    st.subheader("Ward × week risk heatmap")
    n_weeks = st.slider("Weeks to show", 8, 52, 20)
    cutoff = pd.Timestamp(week) - pd.Timedelta(weeks=n_weeks)
    heat = ready[(ready["week"] >= cutoff) & (ready["week"] <= pd.Timestamp(week))]
    order = week_df.sort_values("pred_cases", ascending=False)["ward_id"].tolist()
    st.altair_chart(
        alt.Chart(heat).mark_rect().encode(
            x=alt.X("week:T", title=None), y=alt.Y("ward_id:N", sort=order, title="Ward"),
            color=alt.Color("tier:N", scale=TIER_SCALE, legend=alt.Legend(title="Risk")),
            tooltip=["ward_id", alt.Tooltip("week:T"), "tier", alt.Tooltip("pred_cases:Q", format=".1f")],
        ).properties(width="container", height=max(220, 14 * len(order))))

# ---------- tab 2: drivers ----------
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("What is pushing risk up or down?")
        st.caption("Change in predicted cases if that factor were at its typical value. Positive = raises risk.")
        st.altair_chart(
            alt.Chart(groups_df).mark_bar().encode(
                x=alt.X("effect_cases:Q", title="Effect on predicted cases"),
                y=alt.Y("group:N", sort="-x", title=None),
                color=alt.condition(alt.datum.effect_cases > 0, alt.value(C.TIER_COLORS["Critical"]), alt.value(C.TIER_COLORS["Low"])),
                tooltip=["group", alt.Tooltip("effect_cases:Q", format="+.2f")],
            ).properties(width="container", height=220))
        with st.expander("Per-feature detail"):
            st.dataframe(feat_df.rename(columns={"value": "this ward/week", "typical": "typical", "effect_cases": "effect (cases)"}),
                         hide_index=True)
    with c2:
        st.subheader("Rainfall that is driving this week")
        st.caption(f"Rain falling {min(C.RAIN_LAGS)}-{max(C.RAIN_LAGS)} weeks earlier feeds this prediction (breeding clock).")
        lag_df = pd.DataFrame({"weeks before target": [f"{k}w" for k in C.RAIN_LAGS],
                               "rainfall_mm": [row[f"rain_lag_{k}w"] for k in C.RAIN_LAGS]})
        st.altair_chart(
            alt.Chart(lag_df).mark_bar(color="#3b82f6").encode(
                x=alt.X("weeks before target:N", sort=None), y=alt.Y("rainfall_mm:Q", title="Rainfall (mm)"),
                tooltip=["weeks before target", alt.Tooltip("rainfall_mm:Q", format=".1f")],
            ).properties(width="container", height=220))

    st.subheader(f"Ward {ward}: predicted vs reported cases")
    hist = ready[ready["ward_id"] == ward].tail(52)
    base = alt.Chart(hist).encode(x=alt.X("week:T", title=None))
    layers = [base.mark_line(color="#d62839").encode(y=alt.Y("pred_cases:Q", title="Weekly cases"), tooltip=["week:T", alt.Tooltip("pred_cases:Q", format=".1f")])]
    if C.TARGET in hist:
        layers.append(base.mark_line(color="#555", strokeDash=[4, 3]).encode(y=f"{C.TARGET}:Q"))
    st.altair_chart(alt.layer(*layers).properties(width="container", height=240))
    st.caption("Red = model prediction, dashed grey = reported cases.")

# ---------- tab 3: validation ----------
with tab3:
    st.subheader("Chronological validation")
    st.caption(bundle.get("split", "Train on earlier weeks, test on later weeks (no shuffling)."))
    m = bundle.get("metrics", {})
    if m:
        for col, (k, v) in zip(st.columns(len(m)), m.items()):
            col.metric(k, f"{v:.2f}")
    st.caption(f"Model: {bundle.get('model_name', type(bundle['model']).__name__)} · trained {bundle.get('trained_at', 'n/a')}")
    val = bundle.get("validation")
    if val is not None and len(val):
        wk = val.groupby("week")[["actual", "predicted"]].sum().reset_index().melt("week", var_name="series", value_name="cases")
        st.altair_chart(
            alt.Chart(wk).mark_line().encode(x=alt.X("week:T", title=None), y=alt.Y("cases:Q", title="City-wide weekly cases"),
                                            color=alt.Color("series:N", title=None), tooltip=["week:T", "series", alt.Tooltip("cases:Q", format=".0f")]
                                            ).properties(width="container", height=280))
        err = (val.assign(abs_err=(val.actual - val.predicted).abs()).groupby("ward_id")["abs_err"].mean()
               .sort_values(ascending=False).head(10).reset_index().rename(columns={"abs_err": "MAE"}))
        st.markdown("**Wards with the largest hold-out error**")
        st.dataframe(err, hide_index=True)
    else:
        st.info("No hold-out predictions in the model bundle (optional `validation` DataFrame).")
