import pickle

import numpy as np
import pandas as pd

from src.config import SNAPSHOT_PATH
from src.features.engineer import ELO_INIT, set_categoricals


def load_snapshot() -> dict:
    with open(SNAPSHOT_PATH, "rb") as f:
        return pickle.load(f)


def _quali_gap_for(start: int, snap: dict) -> float:
    """Grid-slot fallback gap% (2025+ races have no historical quali data)."""
    slot = snap.get("quali_slot_gap", {})
    val = slot.get(int(start), slot.get(str(int(start))))
    if val is None:
        return float(snap.get("quali_global_gap", 5.0))
    return float(val)


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
    Rolling reliability, form, Elo, team pace and quali-gap fallback are pulled from
    the snapshot saved at training time, so there's zero leakage — the snapshot
    predates this race.
    """
    snap = load_snapshot()
    d_feats = snap["driver_features"]
    t_feats = snap["team_features"]
    circuit_map = snap["circuit_map"]
    driver_ages = snap["driver_ages"]
    circuit = circuit_map.get(race_name, race_name)

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
                "circuit name": circuit,
                "driver_reliability_ewm10": df_entry.get("reliability", 0.90),
                "team_reliability_ewm10": tf_entry.get("reliability", 0.90),
                "finish_ewm3": df_entry.get("finish_ewm3", 10.0),
                "driver_circuit_avg": df_entry.get("circuit_avgs", {}).get(circuit, 10.0),
                "driver_elo": df_entry.get("elo", ELO_INIT),
                "team_pace_ewm5": tf_entry.get("pace_ewm5", 10.5),
                "quali_gap_pct": _quali_gap_for(entry["start"], snap),
            }
        )

    df = pd.DataFrame(rows)
    # teammate_quali_delta: driver's gap% minus their teammate's within this grid
    grp = df.groupby("team")["quali_gap_pct"]
    cnt = grp.transform("count")
    tot = grp.transform("sum")
    teammate_avg = np.where(cnt > 1, (tot - df["quali_gap_pct"]) / (cnt - 1), df["quali_gap_pct"])
    df["teammate_quali_delta"] = df["quali_gap_pct"] - teammate_avg

    return set_categoricals(df)
