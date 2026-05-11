import numpy as np
import pandas as pd
from src.config import DROP_COLS

CAT_COLS_SPACE = ["GP name", "team", "driver", "circuit name"]
CAT_COLS_UNDERSCORE = ["GP_name", "team", "driver", "circuit_name"]


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
    df.drop(columns=["driver_dnf", "team_dnf"], inplace=True)
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


def set_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CAT_COLS_UNDERSCORE:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def finish_to_relevance(y: pd.Series | np.ndarray, max_pos: int = 20) -> np.ndarray:
    """Convert finish positions (1=best) to relevance scores (higher=better) for LGBMRanker."""
    return max_pos - np.asarray(y) + 1


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Full feature preparation pipeline:
    EWM rolling → circuit affinity → fill NaNs → re-sort by date → sanitize names → cast categoricals → split X/y
    Rows in X are always in ["date", "driver"] order so LGBMRanker group sizes stay aligned.
    """
    df = df.sort_values(["driver", "date"]).copy()  # driver+date order for rolling windows
    df = add_rolling_features(df)
    df = add_circuit_affinity(df)
    df = df.fillna(0.5)
    df = df.sort_values(["date", "driver"]).reset_index(drop=True)  # re-sort for ranker groups
    df = _sanitize_col_names(df)
    df = set_categoricals(df)
    y = df["finish"] if "finish" in df.columns else None
    X = df.drop(columns=[c for c in DROP_COLS + ["dob"] if c in df.columns])
    return X, y


def build_feature_snapshot(df: pd.DataFrame) -> dict:
    """
    Extracts the latest per-driver and per-team rolling features from a processed DataFrame.
    Saved alongside the model so inference can populate features for unseen races without
    re-running the full pipeline.
    """
    # Add rolling features to get the latest values
    df_feat = add_rolling_features(df.sort_values(["driver", "date"]))
    df_feat = add_circuit_affinity(df_feat)

    latest = (
        df_feat
        .dropna(subset=["driver_reliability_ewm10", "finish_ewm3"])
        .drop_duplicates(subset="driver", keep="last")
    )

    driver_features = {
        row["driver"]: {
            "reliability": row["driver_reliability_ewm10"],
            "finish_ewm3": row["finish_ewm3"],
            "circuit_avgs": {},  # populated below
        }
        for _, row in latest.iterrows()
    }

    # Per (driver, circuit) latest average
    circ_latest = (
        df_feat
        .dropna(subset=["driver_circuit_avg"])
        .drop_duplicates(subset=["driver", "circuit name"], keep="last")
    )
    for _, row in circ_latest.iterrows():
        drv = row["driver"]
        if drv in driver_features:
            driver_features[drv]["circuit_avgs"][row["circuit name"]] = row["driver_circuit_avg"]

    team_latest = df_feat.sort_values("date").drop_duplicates(subset="team", keep="last")
    team_features = {
        row["team"]: {"reliability": row.get("team_reliability_ewm10", 0.9)}
        for _, row in team_latest.iterrows()
    }

    circuit_map = (
        df[["GP name", "circuit name"]]
        .dropna()
        .drop_duplicates(subset="GP name", keep="last")
        .set_index("GP name")["circuit name"]
        .to_dict()
    )

    driver_ages = latest.set_index("driver")["age_at_race"].to_dict()

    return {
        "driver_features": driver_features,
        "team_features": team_features,
        "circuit_map": circuit_map,
        "driver_ages": driver_ages,
        "training_cutoff": str(df["date"].max().date()),
    }
