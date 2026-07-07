"""
F1 Race Outcome Predictor — Streamlit Dashboard

Run with:
    streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict.predictor import F1Predictor
from src.predict.prep import build_race_features, load_snapshot

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
)

TEAM_COLOURS = {
    "McLaren": "#FF8000",
    "Ferrari": "#DC0000",
    "Red Bull": "#3671C6",
    "Mercedes": "#27F4D2",
    "Aston Martin": "#358C75",
    "Alpine": "#FF87BC",
    "Williams": "#64C4FF",
    "Racing Bulls": "#6692FF",
    "Haas F1 Team": "#B6BABD",
    "Audi": "#E8002D",
    "Cadillac": "#003087",
}

KNOWN_RACES = [
    "Australian Grand Prix", "Chinese Grand Prix", "Japanese Grand Prix",
    "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Miami Grand Prix",
    "Emilia Romagna Grand Prix", "Monaco Grand Prix", "Spanish Grand Prix",
    "Canadian Grand Prix", "Austrian Grand Prix", "British Grand Prix",
    "Belgian Grand Prix", "Hungarian Grand Prix", "Dutch Grand Prix",
    "Italian Grand Prix", "Azerbaijan Grand Prix", "Singapore Grand Prix",
    "United States Grand Prix", "Mexico City Grand Prix", "São Paulo Grand Prix",
    "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix",
]

DRIVER_TEAMS = {
    "Antonelli": "Mercedes",    "Russell": "Mercedes",
    "Norris": "McLaren",        "Piastri": "McLaren",
    "Leclerc": "Ferrari",       "Hamilton": "Ferrari",
    "Verstappen": "Red Bull",   "Hadjar": "Red Bull",
    "Alonso": "Aston Martin",   "Stroll": "Aston Martin",
    "Gasly": "Alpine",          "Colapinto": "Alpine",
    "Sainz": "Williams",        "Albon": "Williams",
    "Lawson": "Racing Bulls",   "Lindblad": "Racing Bulls",
    "Bearman": "Haas F1 Team",  "Ocon": "Haas F1 Team",
    "Hulkenberg": "Audi",       "Bortoleto": "Audi",
    "Perez": "Cadillac",        "Bottas": "Cadillac",
}


@st.cache_resource(show_spinner="Loading model...")
def load_model() -> F1Predictor:
    return F1Predictor()


@st.cache_data(show_spinner=False)
def get_snapshot() -> dict:
    return load_snapshot()


@st.cache_data(ttl=3600, show_spinner=False)
def get_next_race() -> tuple[str, pd.Timestamp]:
    """Return (EventName, EventDate) for the next race not yet completed."""
    import fastf1
    from datetime import date
    from difflib import get_close_matches
    from src.config import CACHE_DIR
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    today = pd.Timestamp(date.today())
    for year in [today.year, today.year + 1]:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            upcoming = schedule[pd.to_datetime(schedule["EventDate"]) > today]
            if not upcoming.empty:
                row = upcoming.iloc[0]
                name = str(row["EventName"])
                if name not in KNOWN_RACES:
                    matches = get_close_matches(name, KNOWN_RACES, n=1, cutoff=0.6)
                    name = matches[0] if matches else KNOWN_RACES[0]
                return name, pd.Timestamp(row["EventDate"])
        except Exception:
            continue
    return "Canadian Grand Prix", pd.Timestamp("2026-06-14")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏎️ F1 Race Predictor")
    st.caption("Powered by LGBMRanker trained on 2018–2026 data")

    try:
        predictor = load_model()
        st.success(f"Model ready (trained through {predictor.training_cutoff})")
    except FileNotFoundError:
        st.error("Model not found. Run `python train.py` first.")
        st.stop()

    st.divider()
    next_race_name, next_race_date = get_next_race()
    default_idx = KNOWN_RACES.index(next_race_name) if next_race_name in KNOWN_RACES else 0
    race_name = st.selectbox("Select race", KNOWN_RACES, index=default_idx)
    race_date = st.date_input("Race date", value=next_race_date)

    st.subheader("Weather")
    rainfall = st.toggle("Rain expected", value=False)
    avg_track_temp = st.slider("Track temp (°C)", 15, 60, 35)
    min_humidity = st.slider("Min humidity (%)", 5, 100, 40)

# ── Main: Grid Input ──────────────────────────────────────────────────────────
st.title(f"Predict: {race_name}")

st.subheader("Starting Grid")
st.caption("Edit the table below — P1 at the top.")

default_grid = pd.DataFrame(
    [{"Position": i + 1, "Driver": drv, "Team": team}
     for i, (drv, team) in enumerate(DRIVER_TEAMS.items())]
)

grid_df = st.data_editor(
    default_grid,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "Position": st.column_config.NumberColumn(disabled=True, width="small"),
        "Driver": st.column_config.SelectboxColumn(
            options=list(DRIVER_TEAMS.keys()), width="medium"
        ),
        "Team": st.column_config.SelectboxColumn(
            options=list(TEAM_COLOURS.keys()), width="medium"
        ),
    },
)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict race outcome", type="primary", use_container_width=True):
    grid = [
        {"driver": row["Driver"], "team": row["Team"], "start": int(row["Position"])}
        for _, row in grid_df.iterrows()
    ]

    with st.spinner("Running model..."):
        try:
            race_features = build_race_features(
                race_name=race_name,
                race_date=str(race_date),
                grid=grid,
                rainfall=int(rainfall),
                avg_track_temp=float(avg_track_temp),
                min_humidity=float(min_humidity),
            )

            for col in predictor.feature_names:
                if col not in race_features.columns:
                    race_features[col] = 0.5

            result = predictor.predict_with_sim(race_features, n_sims=3000)
            result = result.merge(
                grid_df.rename(columns={"Driver": "driver", "Team": "team", "Position": "start"}),
                on=["driver", "team"],
                how="left",
            )
            # Plackett-Luce win probability as the confidence %
            result["confidence"] = result["p_win"] * 100
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # ── Podium ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Predicted Podium")

    top3 = result[result["pred_finish"] <= 3].sort_values("pred_finish")
    medals = ["🥇", "🥈", "🥉"]
    cols = st.columns(3)
    for col, (medal, (_, row)) in zip(cols, zip(medals, top3.iterrows())):
        team_colour = TEAM_COLOURS.get(row["team"], "#888888")
        col.markdown(
            f"""
            <div style="background:{team_colour}22;border-left:4px solid {team_colour};
                        padding:12px;border-radius:6px;text-align:center">
                <div style="font-size:2em">{medal}</div>
                <div style="font-size:1.4em;font-weight:bold">{row['driver']}</div>
                <div style="color:{team_colour};font-weight:600">{row['team']}</div>
                <div style="font-size:0.85em;color:#888">Started P{int(row['start_x'] if 'start_x' in row else row.get('start', '?'))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Full ranking ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Full Predicted Ranking")

    display = result[
        ["pred_finish", "driver", "team", "confidence", "p_podium", "p_points", "p_dnf"]
    ].copy()
    display.columns = ["Predicted Finish", "Driver", "Team", "Win %", "Podium %", "Points %", "DNF %"]
    for col in ["Win %", "Podium %", "Points %", "DNF %"]:
        scale = 1 if col == "Win %" else 100
        source = {"Win %": "confidence", "Podium %": "p_podium", "Points %": "p_points", "DNF %": "p_dnf"}[col]
        display[col] = (result[source] * scale).map(lambda x: f"{x:.1f}%")

    def _colour_team(val):
        c = TEAM_COLOURS.get(val, "#333")
        return f"color: {c}; font-weight: bold"

    st.dataframe(
        display.style.applymap(_colour_team, subset=["Team"]),
        use_container_width=True,
        hide_index=True,
    )
