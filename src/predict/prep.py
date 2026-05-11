import pickle
import pandas as pd
from src.config import SNAPSHOT_PATH
from src.features.engineer import set_categoricals


def load_snapshot() -> dict:
    with open(SNAPSHOT_PATH, "rb") as f:
        return pickle.load(f)


def build_race_features(
    race_name: str,
    race_date: str,
    grid: list[dict],
    rainfall: int = 0,
    avg_track_temp: float = 35.0,
    min_humidity: float = 40.0,
) -> pd.DataFrame:
    """
    Prepare a feature DataFrame for one upcoming race.

    grid: list of dicts with keys 'driver', 'team', 'start' (1-indexed).
    Rolling reliability and form are pulled from the snapshot saved at training time,
    so there's zero data leakage — the snapshot predates this race.
    """
    snap = load_snapshot()
    d_feats = snap["driver_features"]
    t_feats = snap["team_features"]
    circuit_map = snap["circuit_map"]
    driver_ages = snap["driver_ages"]

    race_ts = pd.Timestamp(race_date)
    rows = []
    for entry in grid:
        drv, team = entry["driver"], entry["team"]
        df_entry = d_feats.get(drv, {})
        tf_entry = t_feats.get(team, {})
        rows.append(
            {
                "GP name": race_name,
                "date": race_ts,
                "driver": drv,
                "team": team,
                "start": entry["start"],
                "year": race_ts.year,
                "rainfall": rainfall,
                "avg_track_temp": avg_track_temp,
                "min_humidity": min_humidity,
                "age_at_race": driver_ages.get(drv, 28),
                "driver_active": 1,
                "team_active": 1,
                "circuit name": circuit_map.get(race_name, race_name),
                "driver_reliability_ewm10": df_entry.get("reliability", 0.90),
                "team_reliability_ewm10": tf_entry.get("reliability", 0.90),
                "finish_ewm3": df_entry.get("finish_ewm3", 10.0),
                "driver_circuit_avg": df_entry.get("circuit_avgs", {}).get(
                    snap["circuit_map"].get(race_name, ""), 10.0
                ),
            }
        )

    df = pd.DataFrame(rows)
    return set_categoricals(df)
