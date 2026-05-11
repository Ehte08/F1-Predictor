"""
compare.py — Baseline vs Improved feature set comparison.

New features added in the improved model:
  1. qualify_delta_s      — qualifying gap to pole position (seconds)
  2. driver_circuit_avg   — historical avg finish at this specific circuit (leakage-free)
  3. driver_champ_points  — driver championship points before this race
  4. driver_champ_pos     — driver championship position before this race
  5. team_champ_points    — constructor championship points before this race
  6. EWM rolling windows  — span=3 form / span=10 reliability (vs flat windows)

Both models use identical train/test splits and identical hyperparameters.
Only the features differ.

Usage:
    python compare.py
"""
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from src.config import DATA_DIR
from src.data.cleaner import TEAM_ALIASES, build_base_dataframe
from src.features.engineer import finish_to_relevance, set_categoricals
from src.models.evaluate import summarize
from src.models.train import BEST_PARAMS, time_based_split

DROP_COLS = ["finish", "date", "dob"]

# ── Qualifying delta ──────────────────────────────────────────────────────────

def _parse_laptime(t) -> float | None:
    if pd.isna(t) or str(t).strip() in ("\\N", "", "nan"):
        return None
    m = re.match(r"(\d+):(\d+\.\d+)", str(t))
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def load_qualifying_delta() -> pd.DataFrame:
    """Returns (date, driver, qualify_delta_s) — gap to pole in seconds."""
    qual = pd.read_csv(DATA_DIR / "qualifying.csv")
    races = pd.read_csv(DATA_DIR / "races.csv", usecols=["raceId", "date"])
    drivers = pd.read_csv(DATA_DIR / "drivers.csv", usecols=["driverId", "surname"])

    for col in ["q3", "q2", "q1"]:
        qual[f"{col}_s"] = qual[col].apply(_parse_laptime)

    # Best time per driver: q3 > q2 > q1
    qual["best_s"] = (
        qual["q3_s"].combine_first(qual["q2_s"]).combine_first(qual["q1_s"])
    )
    pole = qual.groupby("raceId")["best_s"].min().rename("pole_s")
    qual = qual.join(pole, on="raceId")
    qual["qualify_delta_s"] = qual["best_s"] - qual["pole_s"]

    qual = qual.merge(races, on="raceId").merge(drivers, on="driverId")
    qual["date"] = pd.to_datetime(qual["date"])
    qual.rename(columns={"surname": "driver"}, inplace=True)
    return qual[["date", "driver", "qualify_delta_s"]].dropna()


# ── Championship standings ────────────────────────────────────────────────────

def load_championship_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      driver_champ: (date, driver, driver_champ_points, driver_champ_pos)
      team_champ:   (date, team,   team_champ_points)
    Values are from the PREVIOUS race — zero data leakage.
    """
    races = pd.read_csv(DATA_DIR / "races.csv", usecols=["raceId", "date"])
    drivers = pd.read_csv(DATA_DIR / "drivers.csv", usecols=["driverId", "surname"])
    constructors = pd.read_csv(DATA_DIR / "constructors.csv", usecols=["constructorId", "name"])
    ds = pd.read_csv(DATA_DIR / "driver_standings.csv")
    cs = pd.read_csv(DATA_DIR / "constructor_standings.csv")

    race_dates = races.set_index("raceId")["date"].apply(pd.Timestamp)
    drv_map = drivers.set_index("driverId")["surname"]
    team_map = constructors.set_index("constructorId")["name"].replace(TEAM_ALIASES)

    # Driver standings
    ds["date"] = ds["raceId"].map(race_dates)
    ds["driver"] = ds["driverId"].map(drv_map)
    ds = ds[["date", "driver", "points", "position"]].dropna()
    ds = ds.sort_values(["driver", "date"])
    ds["driver_champ_points"] = ds.groupby("driver")["points"].shift(1).fillna(0.0)
    ds["driver_champ_pos"] = ds.groupby("driver")["position"].shift(1).fillna(20.0)
    driver_champ = ds[["date", "driver", "driver_champ_points", "driver_champ_pos"]].drop_duplicates()

    # Constructor standings
    cs["date"] = cs["raceId"].map(race_dates)
    cs["team"] = cs["constructorId"].map(team_map)
    cs = cs[["date", "team", "points"]].dropna()
    cs = cs.sort_values(["team", "date"])
    cs["team_champ_points"] = cs.groupby("team")["points"].shift(1).fillna(0.0)
    team_champ = cs[["date", "team", "team_champ_points"]].drop_duplicates()

    return driver_champ, team_champ


# ── Feature preparation ───────────────────────────────────────────────────────

def _add_flat_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline: flat rolling windows (original approach)."""
    df = df.copy()
    df["driver_dnf"] = df["driver_dnf"].astype(int)
    df["team_dnf"] = df["team_dnf"].astype(int)

    df["driver_reliability_rolling10"] = (
        1 - df.groupby("driver")["driver_dnf"]
            .transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    )
    df["team_reliability_rolling10"] = (
        1 - df.groupby("team")["team_dnf"]
            .transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    )
    df["finish_rolling3"] = (
        df.groupby("driver")["finish"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    df.drop(columns=["driver_dnf", "team_dnf"], inplace=True, errors="ignore")
    return df


def _add_ewm_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Improved: exponentially-weighted windows (recent races weighted more)."""
    df = df.copy()
    df["driver_dnf"] = df["driver_dnf"].astype(int)
    df["team_dnf"] = df["team_dnf"].astype(int)

    df["driver_reliability_ewm10"] = (
        1 - df.groupby("driver")["driver_dnf"]
            .transform(lambda s: s.shift(1).ewm(span=10, min_periods=1).mean())
    )
    df["team_reliability_ewm10"] = (
        1 - df.groupby("team")["team_dnf"]
            .transform(lambda s: s.shift(1).ewm(span=10, min_periods=1).mean())
    )
    df["finish_ewm3"] = (
        df.groupby("driver")["finish"]
            .transform(lambda s: s.shift(1).ewm(span=3, min_periods=1).mean())
    )
    df.drop(columns=["driver_dnf", "team_dnf"], inplace=True, errors="ignore")
    return df


def _add_circuit_affinity(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding historical average finish per (driver, circuit) — shift(1), no leakage."""
    df = df.copy().sort_values(["driver", "circuit name", "date"])
    df["driver_circuit_avg"] = (
        df.groupby(["driver", "circuit name"])["finish"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    # First visit to a circuit: fall back to overall rolling form
    fallback = df.get("finish_ewm3", df.get("finish_rolling3", pd.Series(10.0, index=df.index)))
    df["driver_circuit_avg"] = df["driver_circuit_avg"].fillna(fallback)
    return df


def _sanitize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Replace spaces in column names with underscores so LightGBM names match DataFrame names."""
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


def prepare_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.sort_values(["driver", "date"]).copy()
    df = _add_flat_rolling(df)
    df = df.fillna(0.5)
    df = _sanitize_cols(df)
    # After sanitizing, categorical column names now use underscores
    for col in ["GP_name", "circuit_name", "driver", "team"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    X = df.drop(columns=[c for c in ["finish", "date", "dob"] if c in df.columns])
    return X, df["finish"]


def prepare_improved(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.sort_values(["driver", "date"]).copy()
    df = _add_ewm_rolling(df)
    df = _add_circuit_affinity(df)
    df = df.fillna(0.5)
    df = _sanitize_cols(df)
    for col in ["GP_name", "circuit_name", "driver", "team"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    X = df.drop(columns=[c for c in ["finish", "date", "dob"] if c in df.columns])
    return X, df["finish"]


# ── Train + evaluate ──────────────────────────────────────────────────────────

def train_and_eval(
    full_df: pd.DataFrame,
    train_dates: np.ndarray,
    test_dates: np.ndarray,
    prepare_fn,
    label: str,
) -> dict:
    """
    Prepare features on the FULL dataset (consistent categorical encodings),
    then split by date for train/test. Metadata columns (_date, _driver, _finish)
    are embedded in X temporarily for splitting, then stripped before fitting.
    """
    X_full, y_full = prepare_fn(full_df)
    X_full = X_full.reset_index(drop=True)
    y_full = y_full.reset_index(drop=True)

    # Recover date/driver in the same row order as prepare_fn produced
    df_sorted = full_df.sort_values(["driver", "date"]).reset_index(drop=True)
    X_full["_date"] = df_sorted["date"].values
    X_full["_driver"] = df_sorted["driver"].values
    X_full["_finish"] = y_full.values

    META = ["_date", "_driver", "_finish"]

    train_rows = X_full[X_full["_date"].isin(train_dates)].sort_values("_date").reset_index(drop=True)
    test_rows  = X_full[X_full["_date"].isin(test_dates)].reset_index(drop=True)

    feat_cols = [c for c in X_full.columns if c not in META]

    X_tr = train_rows[feat_cols]
    y_tr = train_rows["_finish"]
    groups_tr = train_rows.groupby("_date", sort=True).size().tolist()

    X_te = test_rows[feat_cols]

    y_tr_rel = finish_to_relevance(y_tr, max_pos=int(y_full.max()))
    model = LGBMRanker(objective="lambdarank", metric="ndcg", verbosity=-1, **BEST_PARAMS)
    model.fit(X_tr, y_tr_rel, group=groups_tr)

    pred_df = test_rows[["_date", "_driver", "_finish"]].rename(
        columns={"_date": "date", "_driver": "driver", "_finish": "finish"}
    ).copy()
    pred_df["pred_score"] = model.predict(X_te[model.feature_name_])
    pred_df["pred_finish"] = (
        pred_df.groupby("date")["pred_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    metrics = summarize(pred_df)
    metrics["label"] = label
    metrics["n_features"] = len(feat_cols)
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading base data...")
    df_base = build_base_dataframe()
    print(f"  {len(df_base):,} rows | {df_base['date'].nunique()} races\n")

    # Build improved dataset by merging new feature sources
    print("Building new features...")
    qual = load_qualifying_delta()
    driver_champ, team_champ = load_championship_features()

    df_imp = (
        df_base
        .merge(qual, on=["date", "driver"], how="left")
        .merge(driver_champ, on=["date", "driver"], how="left")
        .merge(team_champ, on=["date", "team"], how="left")
    )

    cov_qual = df_imp["qualify_delta_s"].notna().mean()
    cov_champ = df_imp["driver_champ_points"].notna().mean()
    print(f"  qualify_delta_s coverage:   {cov_qual:.1%}")
    print(f"  driver_champ_points coverage: {cov_champ:.1%}")

    # Same time-based split for both models — ensures fair comparison
    all_dates = np.sort(df_base["date"].unique())
    cutoff = int(len(all_dates) * 0.80)
    train_dates = all_dates[:cutoff]
    test_dates = all_dates[cutoff:]

    print(f"\n  Train: {len(train_dates)} races  ({train_dates[0]}  →  {train_dates[-1]})")
    print(f"  Test:  {len(test_dates)} races  ({test_dates[0]}  →  {test_dates[-1]})\n")

    print("Training baseline model (flat rolling windows, no new features)...")
    base = train_and_eval(df_base, train_dates, test_dates, prepare_baseline, "Baseline")

    print("Training improved model (EWM + circuit affinity + qualifying + standings)...")
    imp = train_and_eval(df_imp, train_dates, test_dates, prepare_improved, "Improved")

    # ── Results table ─────────────────────────────────────────────────────────
    W = 62
    print("\n" + "═" * W)
    print(f"  {'Metric':<26} {'Baseline':>10} {'Improved':>10} {'Δ':>10}")
    print("─" * W)

    rows = [
        ("Mean Spearman",   "mean_spearman"),
        ("Median Spearman", "median_spearman"),
        ("NDCG@3 (podium)", "ndcg_at_3"),
        ("NDCG@10",         "ndcg_at_10"),
        ("NDCG@20",         "ndcg_at_20"),
    ]
    for label, key in rows:
        b, i = base[key], imp[key]
        delta = i - b
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<26} {b:>10.4f} {i:>10.4f} {sign}{delta:>9.4f}")

    print("─" * W)
    print(f"  {'Features used':<26} {base['n_features']:>10} {imp['n_features']:>10}")
    print(f"  {'Test races':<26} {len(test_dates):>10} {len(test_dates):>10}")
    print("═" * W)

    pct = (imp["mean_spearman"] - base["mean_spearman"]) / abs(base["mean_spearman"]) * 100
    print(f"\n  Mean Spearman lift: {pct:+.1f}%")
    print(f"  NDCG@3 lift:        {(imp['ndcg_at_3'] - base['ndcg_at_3']):+.4f}")


if __name__ == "__main__":
    main()
