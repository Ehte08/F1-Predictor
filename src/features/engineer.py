import functools

import numpy as np
import pandas as pd

from src.config import DATA_DIR, DROP_COLS

CAT_COLS_SPACE = ["GP name", "team", "driver", "circuit name"]
CAT_COLS_UNDERSCORE = ["GP_name", "team", "driver", "circuit_name"]

# Columns that are never part of the feature matrix X.
_NON_FEATURE_COLS = DROP_COLS + ["dob", "dnf", "driver_dnf", "team_dnf"]

# Features derived from historical *finish* positions. Excluded from the DNF
# classifier so it leans on reliability / pace / grid signals instead.
FINISH_DERIVED_FEATURES = ["finish_ewm3", "driver_circuit_avg", "team_pace_ewm5"]

# Elo constants
ELO_INIT = 1500.0
ELO_K = 28.0


def _sanitize_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace spaces in column names with underscores (LightGBM sanitizes internally)."""
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    EWM rolling windows (exponentially weighted — recent races weighted more):
    - driver_reliability_ewm10: 1 - EWM driver DNF rate, span=10
    - team_reliability_ewm10:   1 - EWM team DNF rate, span=10
    - finish_ewm3:              EWM driver finish, span=3

    All use shift(1) — current race excluded from its own lookback.
    """
    df = df.copy()
    df["driver_dnf"] = df["driver_dnf"].astype(int)
    df["team_dnf"] = df["team_dnf"].astype(int)

    df["driver_reliability_ewm10"] = (
        1
        - df.groupby("driver")["driver_dnf"].transform(
            lambda s: s.shift(1).ewm(span=10, min_periods=1).mean()
        )
    )
    df["team_reliability_ewm10"] = (
        1
        - df.groupby("team")["team_dnf"].transform(
            lambda s: s.shift(1).ewm(span=10, min_periods=1).mean()
        )
    )
    df["finish_ewm3"] = df.groupby("driver")["finish"].transform(
        lambda s: s.shift(1).ewm(span=3, min_periods=1).mean()
    )
    return df


def add_circuit_affinity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expanding historical average finish per (driver, circuit) — shift(1), no leakage.
    Falls back to finish_ewm3 on a driver's first visit to a circuit.
    """
    df = df.copy().sort_values(["driver", "circuit name", "date"])
    df["driver_circuit_avg"] = (
        df.groupby(["driver", "circuit name"])["finish"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    fallback = df.get("finish_ewm3", pd.Series(10.0, index=df.index))
    df["driver_circuit_avg"] = df["driver_circuit_avg"].fillna(fallback)
    return df


def add_elo_ratings(
    df: pd.DataFrame,
    k: float = ELO_K,
    init: float = ELO_INIT,
    return_ratings: bool = False,
):
    """
    Incremental per-race Elo over head-to-head finish comparisons.

    The value written for race N is each driver's rating BEFORE race N (leakage-safe:
    a race's own result never contributes to its own feature). After scoring a race,
    ratings are updated from every pairwise finish comparison, the per-driver delta
    averaged over the number of comparisons so K keeps its usual meaning.

    Returns df with a ``driver_elo`` column; if ``return_ratings`` also returns the
    final rating dict for the inference snapshot.
    """
    df = df.copy()
    ratings: dict[str, float] = {}
    elo_vals: dict = {}

    for _, g in df.sort_values("date").groupby("date", sort=True):
        drivers = g["driver"].to_numpy()
        finishes = g["finish"].to_numpy()
        pre = np.array([ratings.get(d, init) for d in drivers], dtype=float)

        for idx, d in zip(g.index, drivers):
            elo_vals[idx] = ratings.get(d, init)

        n = len(drivers)
        deltas = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                exp_i = 1.0 / (1.0 + 10 ** ((pre[j] - pre[i]) / 400.0))
                if finishes[i] < finishes[j]:
                    s_i = 1.0
                elif finishes[i] > finishes[j]:
                    s_i = 0.0
                else:
                    s_i = 0.5
                deltas[i] += k * (s_i - exp_i)
                deltas[j] += k * ((1.0 - s_i) - (1.0 - exp_i))
        denom = max(n - 1, 1)
        for d, delta in zip(drivers, deltas):
            ratings[d] = ratings.get(d, init) + delta / denom

    df["driver_elo"] = pd.Series(elo_vals).reindex(df.index)
    if return_ratings:
        return df, dict(ratings)
    return df


def add_team_pace(df: pd.DataFrame) -> pd.DataFrame:
    """team_pace_ewm5: EWM (span=5) of a team's mean finish per race, shift(1)."""
    df = df.copy()
    race_team = (
        df.groupby(["date", "team"])["finish"].mean().reset_index(name="_team_finish")
    )
    race_team = race_team.sort_values("date")
    race_team["team_pace_ewm5"] = race_team.groupby("team")["_team_finish"].transform(
        lambda s: s.shift(1).ewm(span=5, min_periods=1).mean()
    )
    return df.merge(
        race_team[["date", "team", "team_pace_ewm5"]], on=["date", "team"], how="left"
    )


# ── Qualifying features ───────────────────────────────────────────────────────

def _parse_laptime(t) -> float:
    """Parse an Ergast lap time ('M:SS.mmm' or 'SS.mmm') to seconds; NaN if invalid."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return np.nan
    s = str(t).strip()
    if not s or s in ("\\N", "nan"):
        return np.nan
    try:
        if ":" in s:
            mins, secs = s.split(":")
            return float(mins) * 60.0 + float(secs)
        return float(s)
    except (ValueError, TypeError):
        return np.nan


@functools.lru_cache(maxsize=1)
def _quali_lookup() -> dict:
    """(date_str, surname) -> best quali time in seconds. Empty dict if unavailable."""
    try:
        q = pd.read_csv(DATA_DIR / "qualifying.csv")
        races = pd.read_csv(DATA_DIR / "races.csv")[["raceId", "date"]]
        drivers = pd.read_csv(DATA_DIR / "drivers.csv")[["driverId", "surname"]]
        q = q.merge(races, on="raceId").merge(drivers, on="driverId")
        for c in ("q1", "q2", "q3"):
            q[c] = q[c].map(_parse_laptime)
        q["best"] = q[["q1", "q2", "q3"]].min(axis=1)
        q = q.dropna(subset=["best"])
        return {
            (str(r.date), str(r.surname)): float(r.best)
            for r in q.itertuples(index=False)
        }
    except Exception:
        return {}


def grid_gap_fallback(df: pd.DataFrame) -> tuple[dict, float]:
    """Median real quali gap% per grid slot, used to impute races without quali data."""
    if "_gap_real" not in df.columns:
        return {}, 5.0
    real = df.dropna(subset=["_gap_real"])
    if real.empty:
        return {}, 5.0
    slot_med = real.groupby("start")["_gap_real"].median().to_dict()
    return {int(k): float(v) for k, v in slot_med.items()}, float(real["_gap_real"].median())


def add_quali_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    quali_gap_pct:        driver's best quali time gap to pole, as a %.
    teammate_quali_delta: driver's gap% minus their teammate's gap% (same race/team).

    Real quali times come from f1_datasets/qualifying.csv (best of q1/q2/q3). Races
    without quali data (2025+ FastF1 races) fall back to a median gap fitted per grid
    slot. Never raises on missing data.
    """
    df = df.copy()
    lookup = _quali_lookup()
    dates = df["date"].dt.strftime("%Y-%m-%d")
    df["_quali_best"] = [
        lookup.get((d, drv), np.nan) for d, drv in zip(dates, df["driver"])
    ]
    df["_pole"] = df.groupby("date")["_quali_best"].transform("min")
    df["_gap_real"] = (df["_quali_best"] - df["_pole"]) / df["_pole"] * 100.0

    slot_med, global_med = grid_gap_fallback(df)
    fb = df["start"].map(lambda s: slot_med.get(int(s), global_med))
    df["quali_gap_pct"] = df["_gap_real"].fillna(fb)

    grp = df.groupby(["date", "team"])["quali_gap_pct"]
    cnt = grp.transform("count")
    tot = grp.transform("sum")
    teammate_avg = np.where(cnt > 1, (tot - df["quali_gap_pct"]) / (cnt - 1), df["quali_gap_pct"])
    df["teammate_quali_delta"] = df["quali_gap_pct"] - teammate_avg

    df.drop(columns=["_quali_best", "_pole", "_gap_real"], inplace=True)
    return df


# ── Categoricals / relevance ──────────────────────────────────────────────────

def set_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CAT_COLS_UNDERSCORE:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def finish_to_relevance(y: pd.Series | np.ndarray, max_pos: int = 20) -> np.ndarray:
    """Convert finish positions (1=best) to relevance scores (higher=better) for LGBMRanker."""
    return max_pos - np.asarray(y) + 1


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature-engineering pass. Returns a processed DataFrame sorted by
    ["date", "driver"], column names sanitized, categoricals cast, with a ``dnf``
    label column retained (excluded from X by prepare_features).
    """
    df = df.sort_values(["driver", "date"]).copy()
    if "driver_dnf" in df.columns and "team_dnf" in df.columns:
        df["dnf"] = ((df["driver_dnf"] == 1) | (df["team_dnf"] == 1)).astype(int)
    else:
        df["dnf"] = 0

    df = add_rolling_features(df)
    df = add_circuit_affinity(df)
    df = add_elo_ratings(df)
    df = add_team_pace(df)
    df = add_quali_features(df)

    # Column-specific fills before the catch-all so magnitudes stay sensible.
    df["team_pace_ewm5"] = df["team_pace_ewm5"].fillna(10.5)
    df["driver_elo"] = df["driver_elo"].fillna(ELO_INIT)
    med_gap = df["quali_gap_pct"].median()
    df["quali_gap_pct"] = df["quali_gap_pct"].fillna(med_gap if pd.notna(med_gap) else 5.0)
    df["teammate_quali_delta"] = df["teammate_quali_delta"].fillna(0.0)
    df = df.fillna(0.5)

    df = df.sort_values(["date", "driver"]).reset_index(drop=True)
    df = _sanitize_col_names(df)
    df = set_categoricals(df)
    return df


def prepare_features(
    df: pd.DataFrame, return_meta: bool = False
):
    """
    Engineer features and split into (X, y).

    With ``return_meta=True`` also returns a metadata DataFrame aligned row-for-row
    with X (columns: driver, team, date, finish, dnf) so callers can filter DNF rows
    or recompute ranker groups without re-running the pipeline.
    """
    proc = engineer(df)
    y = proc["finish"] if "finish" in proc.columns else None
    X = proc.drop(columns=[c for c in _NON_FEATURE_COLS if c in proc.columns])
    if return_meta:
        meta = proc[[c for c in ["driver", "team", "date", "finish", "dnf"] if c in proc.columns]].copy()
        return X, y, meta
    return X, y


def build_feature_snapshot(df: pd.DataFrame) -> dict:
    """
    Extract the latest per-driver / per-team features from a processed DataFrame.
    Saved with the model so inference can populate features for unseen races without
    re-running the full pipeline. Includes Elo ratings, team pace and quali fallback maps.
    """
    df_feat = df.sort_values(["driver", "date"]).copy()
    df_feat["dnf"] = ((df_feat["driver_dnf"] == 1) | (df_feat["team_dnf"] == 1)).astype(int)
    df_feat = add_rolling_features(df_feat)
    df_feat = add_circuit_affinity(df_feat)
    df_feat, final_elo = add_elo_ratings(df_feat, return_ratings=True)
    df_feat = add_team_pace(df_feat)

    # Quali fallback map (grid slot -> gap%), fitted on real quali rows.
    tmp = df.sort_values(["driver", "date"]).copy()
    lookup = _quali_lookup()
    dates = tmp["date"].dt.strftime("%Y-%m-%d")
    best = [lookup.get((d, drv), np.nan) for d, drv in zip(dates, tmp["driver"])]
    tmp["_quali_best"] = best
    tmp["_pole"] = tmp.groupby("date")["_quali_best"].transform("min")
    tmp["_gap_real"] = (tmp["_quali_best"] - tmp["_pole"]) / tmp["_pole"] * 100.0
    slot_med, global_med = grid_gap_fallback(tmp)

    latest = (
        df_feat.dropna(subset=["driver_reliability_ewm10", "finish_ewm3"])
        .drop_duplicates(subset="driver", keep="last")
    )

    driver_features = {
        row["driver"]: {
            "reliability": float(row["driver_reliability_ewm10"]),
            "finish_ewm3": float(row["finish_ewm3"]),
            "elo": float(final_elo.get(row["driver"], ELO_INIT)),
            "circuit_avgs": {},
        }
        for _, row in latest.iterrows()
    }

    circ_latest = (
        df_feat.dropna(subset=["driver_circuit_avg"])
        .drop_duplicates(subset=["driver", "circuit name"], keep="last")
    )
    for _, row in circ_latest.iterrows():
        drv = row["driver"]
        if drv in driver_features:
            driver_features[drv]["circuit_avgs"][row["circuit name"]] = float(
                row["driver_circuit_avg"]
            )

    team_latest = df_feat.sort_values("date").drop_duplicates(subset="team", keep="last")
    team_features = {
        row["team"]: {
            "reliability": float(row.get("team_reliability_ewm10", 0.9)),
            "pace_ewm5": float(row["team_pace_ewm5"])
            if pd.notna(row.get("team_pace_ewm5")) else 10.5,
        }
        for _, row in team_latest.iterrows()
    }

    circuit_map = (
        df[["GP name", "circuit name"]]
        .dropna()
        .drop_duplicates(subset="GP name", keep="last")
        .set_index("GP name")["circuit name"]
        .to_dict()
    )

    driver_ages = latest.set_index("driver")["age_at_race"].astype(float).to_dict()

    return {
        "driver_features": driver_features,
        "team_features": team_features,
        "circuit_map": circuit_map,
        "driver_ages": driver_ages,
        "quali_slot_gap": {int(k): float(v) for k, v in slot_med.items()},
        "quali_global_gap": float(global_med),
        "training_cutoff": str(df["date"].max().date()),
    }
