"""
compare_v2.py — Three model improvements tested against the current improved baseline.

Models compared (all use identical train/test splits and feature set):
  0. Improved baseline  — EWM + circuit affinity + qualifying delta + standings (from compare.py)
  1. Optuna tuned       — Same features, hyperparameters found via 30-trial Optuna search
  2. DNF classifier     — Adds P(DNF) from an XGBoost classifier as an extra ranker feature
  3. Ensemble           — Blends LGBMRanker and XGBoost Regressor predictions by averaged rank
  4. Full stack         — All three combined: Optuna params + DNF feature + Ensemble

Usage:
    python compare_v2.py
"""
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from xgboost import XGBClassifier, XGBRegressor

from src.config import DATA_DIR
from src.data.cleaner import TEAM_ALIASES, build_base_dataframe
from src.features.engineer import (
    add_circuit_affinity,
    add_rolling_features,
    finish_to_relevance,
)
from src.models.evaluate import summarize
from src.models.train import BEST_PARAMS

# ── Reuse feature builders from compare.py ────────────────────────────────────

def _parse_laptime(t) -> float | None:
    if pd.isna(t) or str(t).strip() in ("\\N", "", "nan"):
        return None
    m = re.match(r"(\d+):(\d+\.\d+)", str(t))
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def load_qualifying_delta() -> pd.DataFrame:
    qual = pd.read_csv(DATA_DIR / "qualifying.csv")
    races = pd.read_csv(DATA_DIR / "races.csv", usecols=["raceId", "date"])
    drivers = pd.read_csv(DATA_DIR / "drivers.csv", usecols=["driverId", "surname"])
    for col in ["q3", "q2", "q1"]:
        qual[f"{col}_s"] = qual[col].apply(_parse_laptime)
    qual["best_s"] = qual["q3_s"].combine_first(qual["q2_s"]).combine_first(qual["q1_s"])
    pole = qual.groupby("raceId")["best_s"].min().rename("pole_s")
    qual = qual.join(pole, on="raceId")
    qual["qualify_delta_s"] = qual["best_s"] - qual["pole_s"]
    qual = qual.merge(races, on="raceId").merge(drivers, on="driverId")
    qual["date"] = pd.to_datetime(qual["date"])
    qual.rename(columns={"surname": "driver"}, inplace=True)
    return qual[["date", "driver", "qualify_delta_s"]].dropna()


def load_championship_features():
    races = pd.read_csv(DATA_DIR / "races.csv", usecols=["raceId", "date"])
    drivers = pd.read_csv(DATA_DIR / "drivers.csv", usecols=["driverId", "surname"])
    constructors = pd.read_csv(DATA_DIR / "constructors.csv", usecols=["constructorId", "name"])
    ds = pd.read_csv(DATA_DIR / "driver_standings.csv")
    cs = pd.read_csv(DATA_DIR / "constructor_standings.csv")

    race_dates = races.set_index("raceId")["date"].apply(pd.Timestamp)
    drv_map = drivers.set_index("driverId")["surname"]
    team_map = constructors.set_index("constructorId")["name"].replace(TEAM_ALIASES)

    ds["date"] = ds["raceId"].map(race_dates)
    ds["driver"] = ds["driverId"].map(drv_map)
    ds = ds[["date", "driver", "points", "position"]].dropna().sort_values(["driver", "date"])
    ds["driver_champ_points"] = ds.groupby("driver")["points"].shift(1).fillna(0.0)
    ds["driver_champ_pos"] = ds.groupby("driver")["position"].shift(1).fillna(20.0)
    driver_champ = ds[["date", "driver", "driver_champ_points", "driver_champ_pos"]].drop_duplicates()

    cs["date"] = cs["raceId"].map(race_dates)
    cs["team"] = cs["constructorId"].map(team_map)
    cs = cs[["date", "team", "points"]].dropna().sort_values(["team", "date"])
    cs["team_champ_points"] = cs.groupby("team")["points"].shift(1).fillna(0.0)
    team_champ = cs[["date", "team", "team_champ_points"]].drop_duplicates()
    return driver_champ, team_champ


# ── Core feature preparation ──────────────────────────────────────────────────

CAT_COLS = ["GP_name", "circuit_name", "driver", "team"]
META_COLS = ["_date", "_driver", "_finish", "_driver_dnf"]


def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


def prepare_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies rolling features + circuit affinity to df, retains driver_dnf as
    '_driver_dnf' for DNF classifier training, then sanitizes column names.
    Returns a flat DataFrame with all features + metadata columns.
    """
    df = df.sort_values(["driver", "date"]).copy()

    # Save DNF target before rolling drops it
    dnf_target = df["driver_dnf"].astype(int).values

    df = add_rolling_features(df)   # computes EWM features, drops driver_dnf/team_dnf
    df = add_circuit_affinity(df)
    df = df.fillna(0.5)
    df = _sanitize(df)

    # Re-attach metadata (same row order after sort)
    df = df.reset_index(drop=True)
    df["_date"] = pd.to_datetime(df["date"]) if "date" in df.columns else np.nan
    df["_driver"] = df["driver"].values
    df["_finish"] = df["finish"].values
    df["_driver_dnf"] = dnf_target

    return df


def split_Xy(df: pd.DataFrame, extra_drop: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    drop = ["finish", "date", "dob"] + META_COLS + (extra_drop or [])
    X = df.drop(columns=[c for c in drop if c in df.columns])
    y = df["_finish"]
    return X, y


def set_cats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


# ── DNF classifier ────────────────────────────────────────────────────────────

DNF_FEATURES = [
    "driver_reliability_ewm10",
    "team_reliability_ewm10",
    "rainfall",
    "start",
    "age_at_race",
    "driver_active",
    "team_active",
    "avg_track_temp",
    "min_humidity",
]


def train_dnf_classifier(train_df: pd.DataFrame) -> XGBClassifier:
    """
    Trains an XGBoost binary classifier to predict P(DNF) per driver-race.
    Only uses pre-race features (rolling reliability, weather, grid position).
    """
    feats = [f for f in DNF_FEATURES if f in train_df.columns]
    X = train_df[feats].fillna(0.5)
    y = train_df["_driver_dnf"]
    clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(X, y)
    return clf


def add_dnf_prob(df: pd.DataFrame, clf: XGBClassifier) -> pd.DataFrame:
    feats = [f for f in DNF_FEATURES if f in df.columns]
    df = df.copy()
    df["p_dnf"] = clf.predict_proba(df[feats].fillna(0.5))[:, 1]
    return df


# ── Shared train+eval helper ──────────────────────────────────────────────────

def _fit_ranker(X_tr, y_tr, groups_tr, params: dict) -> LGBMRanker:
    y_rel = finish_to_relevance(y_tr, max_pos=int(y_tr.max()))
    model = LGBMRanker(objective="lambdarank", metric="ndcg", verbosity=-1, **params)
    model.fit(X_tr, y_rel, group=groups_tr)
    return model


def _rank_scores(pred_df: pd.DataFrame, score_col: str) -> pd.Series:
    return pred_df.groupby("_date")[score_col].rank(ascending=False, method="first").astype(int)


def _evaluate(pred_df: pd.DataFrame) -> dict:
    pred_df = pred_df.rename(columns={"_date": "date", "_driver": "driver", "_finish": "finish"})
    return summarize(pred_df.reset_index(drop=True))


# ── Model 0: Improved baseline ────────────────────────────────────────────────

def run_improved(full_prep, train_dates, test_dates, params=BEST_PARAMS) -> dict:
    train = full_prep[full_prep["_date"].isin(train_dates)].sort_values("_date").reset_index(drop=True)
    test  = full_prep[full_prep["_date"].isin(test_dates)].reset_index(drop=True)

    X_tr, y_tr = split_Xy(set_cats(train))
    X_te, _    = split_Xy(set_cats(test))
    groups_tr  = train.groupby("_date", sort=True).size().tolist()

    model = _fit_ranker(X_tr, y_tr, groups_tr, params)
    feat_cols = model.feature_name_

    test["pred_score"] = model.predict(X_te[feat_cols])
    test["pred_finish"] = _rank_scores(test, "pred_score")
    return _evaluate(test[["_date", "_driver", "_finish", "pred_score", "pred_finish"]])


# ── Model 1: Optuna tuned ─────────────────────────────────────────────────────

def run_optuna(full_prep, train_dates, test_dates, n_trials: int = 30) -> tuple[dict, dict]:
    """Returns (metrics, best_params)."""

    # Use the first 80% of train dates for Optuna fitting, last 20% for validation
    t_dates = np.sort(train_dates)
    val_cut = int(len(t_dates) * 0.80)
    opt_train_dates = t_dates[:val_cut]
    opt_val_dates   = t_dates[val_cut:]

    opt_train = full_prep[full_prep["_date"].isin(opt_train_dates)].sort_values("_date").reset_index(drop=True)
    opt_val   = full_prep[full_prep["_date"].isin(opt_val_dates)].reset_index(drop=True)

    # Pre-fit categoricals on the combined opt set for consistency
    opt_all = pd.concat([opt_train, opt_val])
    for col in CAT_COLS:
        if col in opt_all.columns:
            cats = pd.CategoricalDtype(categories=sorted(opt_all[col].dropna().astype(str).unique()))
            opt_train[col] = opt_train[col].astype(str).astype(cats)
            opt_val[col]   = opt_val[col].astype(str).astype(cats)

    X_opt_tr, y_opt_tr = split_Xy(opt_train)
    X_opt_va, _        = split_Xy(opt_val)
    groups_opt_tr = opt_train.groupby("_date", sort=True).size().tolist()
    groups_opt_va = opt_val.groupby("_date", sort=True).size().tolist()
    y_opt_tr_rel  = finish_to_relevance(y_opt_tr, max_pos=20)
    y_opt_va_rel  = finish_to_relevance(opt_val["_finish"], max_pos=20)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "min_data_in_leaf":  trial.suggest_int("min_data_in_leaf", 5, 80),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":      1,
        }
        m = LGBMRanker(objective="lambdarank", metric="ndcg", verbosity=-1, **params)
        m.fit(X_opt_tr, y_opt_tr_rel, group=groups_opt_tr)

        scores = m.predict(X_opt_va[m.feature_name_])
        # Compute mean NDCG@10 over validation races
        offset, ndcgs = 0, []
        for gs in groups_opt_va:
            s, e = offset, offset + gs
            ndcg = ndcg_score(
                y_opt_va_rel[s:e].reshape(1, -1),
                scores[s:e].reshape(1, -1),
                k=10,
            )
            ndcgs.append(ndcg)
            offset = e
        return float(np.mean(ndcgs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = {**study.best_params, "bagging_freq": 1}

    metrics = run_improved(full_prep, train_dates, test_dates, params=best_params)
    return metrics, best_params


# ── Model 2: DNF classifier feature ──────────────────────────────────────────

def run_dnf_classifier(full_prep, train_dates, test_dates, params=BEST_PARAMS) -> dict:
    train = full_prep[full_prep["_date"].isin(train_dates)].sort_values("_date").reset_index(drop=True)
    test  = full_prep[full_prep["_date"].isin(test_dates)].reset_index(drop=True)

    # Train DNF classifier on train split only (no leakage)
    dnf_clf = train_dnf_classifier(train)

    train = add_dnf_prob(train, dnf_clf)
    test  = add_dnf_prob(test, dnf_clf)

    X_tr, y_tr = split_Xy(set_cats(train))
    X_te, _    = split_Xy(set_cats(test))
    groups_tr  = train.groupby("_date", sort=True).size().tolist()

    model = _fit_ranker(X_tr, y_tr, groups_tr, params)
    test["pred_score"] = model.predict(X_te[model.feature_name_])
    test["pred_finish"] = _rank_scores(test, "pred_score")
    return _evaluate(test[["_date", "_driver", "_finish", "pred_score", "pred_finish"]])


# ── Model 3: Ensemble (LGBMRanker + XGBoost Regressor) ───────────────────────

def run_ensemble(full_prep, train_dates, test_dates, params=BEST_PARAMS) -> dict:
    train = full_prep[full_prep["_date"].isin(train_dates)].sort_values("_date").reset_index(drop=True)
    test  = full_prep[full_prep["_date"].isin(test_dates)].reset_index(drop=True)

    X_tr, y_tr = split_Xy(set_cats(train))
    X_te, _    = split_Xy(set_cats(test))
    groups_tr  = train.groupby("_date", sort=True).size().tolist()

    # LGBMRanker
    ranker = _fit_ranker(X_tr, y_tr, groups_tr, params)

    # XGBoost Regressor — predicts finish position (1-20)
    X_tr_xgb = X_tr.drop(columns=[c for c in CAT_COLS if c in X_tr.columns])
    X_te_xgb = X_te.drop(columns=[c for c in CAT_COLS if c in X_te.columns])
    xgb = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=0,
    )
    xgb.fit(X_tr_xgb, y_tr)

    # Blend by averaging per-race ranks
    test = test.copy()
    test["lgbm_score"]  = ranker.predict(X_te[ranker.feature_name_])
    test["xgb_pred"]    = xgb.predict(X_te_xgb)

    # Rank within each race: LGBMRanker (higher=better) and XGBoost (lower=better)
    test["lgbm_rank"] = test.groupby("_date")["lgbm_score"].rank(ascending=False)
    test["xgb_rank"]  = test.groupby("_date")["xgb_pred"].rank(ascending=True)
    test["avg_rank"]  = (test["lgbm_rank"] + test["xgb_rank"]) / 2
    test["pred_finish"] = (
        test.groupby("_date")["avg_rank"].rank(method="first").astype(int)
    )
    # Give pred_score for summarize() — use negative avg_rank so higher = better
    test["pred_score"] = -test["avg_rank"]
    return _evaluate(test[["_date", "_driver", "_finish", "pred_finish", "pred_score"]])


# ── Model 4: Full stack ───────────────────────────────────────────────────────

def run_full_stack(full_prep, train_dates, test_dates, optuna_params: dict) -> dict:
    """Optuna-tuned params + DNF classifier feature + Ensemble."""
    train = full_prep[full_prep["_date"].isin(train_dates)].sort_values("_date").reset_index(drop=True)
    test  = full_prep[full_prep["_date"].isin(test_dates)].reset_index(drop=True)

    dnf_clf = train_dnf_classifier(train)
    train = add_dnf_prob(train, dnf_clf)
    test  = add_dnf_prob(test, dnf_clf)

    X_tr, y_tr = split_Xy(set_cats(train))
    X_te, _    = split_Xy(set_cats(test))
    groups_tr  = train.groupby("_date", sort=True).size().tolist()

    ranker = _fit_ranker(X_tr, y_tr, groups_tr, optuna_params)

    X_tr_xgb = X_tr.drop(columns=[c for c in CAT_COLS if c in X_tr.columns])
    X_te_xgb = X_te.drop(columns=[c for c in CAT_COLS if c in X_te.columns])
    xgb = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5,
                       subsample=0.8, colsample_bytree=0.8, verbosity=0)
    xgb.fit(X_tr_xgb, y_tr)

    test = test.copy()
    test["lgbm_score"] = ranker.predict(X_te[ranker.feature_name_])
    test["xgb_pred"]   = xgb.predict(X_te_xgb)
    test["lgbm_rank"]  = test.groupby("_date")["lgbm_score"].rank(ascending=False)
    test["xgb_rank"]   = test.groupby("_date")["xgb_pred"].rank(ascending=True)
    test["avg_rank"]   = (test["lgbm_rank"] + test["xgb_rank"]) / 2
    test["pred_finish"] = test.groupby("_date")["avg_rank"].rank(method="first").astype(int)
    test["pred_score"]  = -test["avg_rank"]
    return _evaluate(test[["_date", "_driver", "_finish", "pred_finish", "pred_score"]])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data and building features...")
    df_base = build_base_dataframe()
    qual    = load_qualifying_delta()
    d_champ, t_champ = load_championship_features()

    df_imp = (
        df_base
        .merge(qual, on=["date", "driver"], how="left")
        .merge(d_champ, on=["date", "driver"], how="left")
        .merge(t_champ, on=["date", "team"], how="left")
    )

    print("Preparing features (EWM + circuit affinity + qualifying + standings)...")
    full_prep = prepare_full(df_imp)

    # Pre-set categoricals from full universe
    for col in CAT_COLS:
        if col in full_prep.columns:
            cats = pd.CategoricalDtype(
                categories=sorted(full_prep[col].dropna().astype(str).unique())
            )
            full_prep[col] = full_prep[col].astype(str).astype(cats)

    all_dates  = np.sort(full_prep["_date"].unique())
    cutoff     = int(len(all_dates) * 0.80)
    train_dates = all_dates[:cutoff]
    test_dates  = all_dates[cutoff:]

    print(f"  Train: {len(train_dates)} races | Test: {len(test_dates)} races\n")

    results = {}

    t0 = time.time()
    print("[0/4] Improved baseline...")
    results["Improved baseline"] = run_improved(full_prep, train_dates, test_dates)
    print(f"      done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("[1/4] Optuna tuning (30 trials)...")
    results["Optuna tuned"], best_params = run_optuna(full_prep, train_dates, test_dates, n_trials=30)
    print(f"      done ({time.time()-t0:.1f}s) | best params: {best_params}")

    t0 = time.time()
    print("[2/4] DNF classifier feature...")
    results["+ DNF classifier"] = run_dnf_classifier(full_prep, train_dates, test_dates)
    print(f"      done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("[3/4] Ensemble (LGBMRanker + XGBoost)...")
    results["+ Ensemble"] = run_ensemble(full_prep, train_dates, test_dates)
    print(f"      done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("[4/4] Full stack (Optuna + DNF + Ensemble)...")
    results["Full stack"] = run_full_stack(full_prep, train_dates, test_dates, best_params)
    print(f"      done ({time.time()-t0:.1f}s)")

    # ── Results table ─────────────────────────────────────────────────────────
    W = 78
    metrics = ["mean_spearman", "median_spearman", "ndcg_at_3", "ndcg_at_10", "ndcg_at_20"]
    labels  = ["Mean Spearman", "Median Spearman", "NDCG@3", "NDCG@10", "NDCG@20"]
    models  = list(results.keys())
    base_key = "Improved baseline"

    print("\n" + "═" * W)
    header = f"  {'Metric':<18}" + "".join(f"{m:>12}" for m in models)
    print(header)
    print("─" * W)

    for label, key in zip(labels, metrics):
        row = f"  {label:<18}"
        for mname in models:
            v = results[mname][key]
            b = results[base_key][key]
            delta = v - b
            sign = "+" if delta > 0 else ""
            marker = " ✓" if delta > 0.001 else (" ✗" if delta < -0.001 else "  ")
            if mname == base_key:
                row += f"  {v:>8.4f}  "
            else:
                row += f"  {v:>6.4f}{marker}"
        print(row)

    print("─" * W)
    row = f"  {'Δ Mean Spearman':<18}"
    for mname in models:
        if mname == base_key:
            row += f"  {'—':>10}  "
        else:
            d = results[mname]["mean_spearman"] - results[base_key]["mean_spearman"]
            row += f"  {d:>+7.4f}   "
    print(row)
    row = f"  {'Δ NDCG@3':<18}"
    for mname in models:
        if mname == base_key:
            row += f"  {'—':>10}  "
        else:
            d = results[mname]["ndcg_at_3"] - results[base_key]["ndcg_at_3"]
            row += f"  {d:>+7.4f}   "
    print(row)
    print("═" * W)


if __name__ == "__main__":
    main()
