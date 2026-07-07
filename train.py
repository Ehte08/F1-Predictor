"""
Train the F1 race-outcome predictor and save all artifacts.

Orchestrates: fetch new races → clean → time-split evaluation (ranker + DNF) →
refit ranker + DNF + calibrate tau on ALL data → save pickle bundle → feature
snapshot (incl. Elo + team pace + quali fallback) → browser_model.json export.

Usage:
    python train.py
    python train.py --test-frac 0.15 --no-mlflow
    python train.py --save-path models/lgbm_ranker.pkl
"""
import argparse
import pickle
import warnings
from pathlib import Path

import mlflow
import pandas as pd

from src.artifacts import export_browser_model, model_version
from src.config import MODEL_PATH, SNAPSHOT_PATH, TEST_FRAC
from src.data.cleaner import build_base_dataframe
from src.data.fetcher import update_race_cache
from src.features.engineer import build_feature_snapshot
from src.models.dnf import evaluate_dnf
from src.models.evaluate import summarize
from src.models.pipeline import engineer_full, train_upto
from src.models.train import time_based_split

warnings.filterwarnings("ignore")


def main(
    save_path: Path = MODEL_PATH,
    test_frac: float = TEST_FRAC,
    log_mlflow: bool = True,
) -> dict:
    print("Checking for new race data...")
    n_new = update_race_cache(verbose=True)
    if n_new:
        print(f"  Fetched {n_new} new race(s).\n")

    print("Loading and cleaning data...")
    df = build_base_dataframe()
    print(f"  {len(df):,} rows | {df['date'].nunique()} races | {df['driver'].nunique()} drivers")

    eng = engineer_full(df)

    # ── Time-split evaluation (ranker order + DNF AUC) ──
    train_df, test_df = time_based_split(df, test_frac=test_frac)
    cutoff = pd.Timestamp(test_df["date"].min())
    print(
        f"  Train < {cutoff.date()} | Test: {test_df['date'].nunique()} races "
        f"({test_df['date'].min().date()} → {test_df['date'].max().date()})"
    )

    print("\nFitting ranker + DNF on train split...")
    bundle = train_upto(eng, cutoff)
    model, feature_names = bundle["ranker"], bundle["feats"]

    # Score the held-out races from the already-engineered matrix (leakage-safe).
    test_mask = (eng["meta"]["date"] >= cutoff).to_numpy()
    pred_df = eng["meta"].loc[test_mask].copy()
    pred_df["pred_score"] = model.predict(eng["X"].loc[test_mask, feature_names])
    pred_df["pred_finish"] = (
        pred_df.groupby("date")["pred_score"].rank(ascending=False, method="first").astype(int)
    )
    metrics = summarize(pred_df)
    dnf_metrics = evaluate_dnf(df, test_frac=test_frac)
    metrics.update(dnf_metrics)
    metrics["tau"] = bundle["tau"]

    print("\n── Test-set metrics ─────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<22} {v:.4f}")
    print("─────────────────────────────────────────────")

    if log_mlflow:
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "min_year": 2018,
                    "test_frac": test_frac,
                    "train_races": train_df["date"].nunique(),
                    "test_races": test_df["date"].nunique(),
                    "n_features": len(feature_names),
                }
            )
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            print("  MLflow run logged.")

    # ── Refit on ALL data before saving ──
    print("\nRefitting ranker + DNF + tau on full dataset...")
    final_cutoff = pd.Timestamp(df["date"].max()) + pd.Timedelta(days=1)
    final = train_upto(eng, final_cutoff)
    training_cutoff = str(df["date"].max().date())

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "model": final["ranker"],
                "dnf_model": final["dnf_model"],
                "feature_names": final["feats"],
                "dnf_feature_names": final["dnf_feats"],
                "tau": final["tau"],
                "training_cutoff": training_cutoff,
            },
            f,
        )
    print(f"  Model bundle saved → {save_path}")

    snap = build_feature_snapshot(df)
    with open(SNAPSHOT_PATH, "wb") as f:
        pickle.dump(snap, f)
    print(f"  Feature snapshot saved → {SNAPSHOT_PATH}")

    export_browser_model(
        version=model_version(),
        trained_through=training_cutoff,
        ranker=final["ranker"],
        dnf_model=final["dnf_model"],
        feats=final["feats"],
        X_all=eng["X"],
        tau=final["tau"],
        snapshot=snap,
        circuit_map=snap["circuit_map"],
        driver_ages=snap["driver_ages"],
    )
    print("  Browser model exported → data/site/model/browser_model.json")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the F1 race outcome predictor.")
    parser.add_argument("--save-path", default=str(MODEL_PATH))
    parser.add_argument("--test-frac", type=float, default=TEST_FRAC)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    main(
        save_path=Path(args.save_path),
        test_frac=args.test_frac,
        log_mlflow=not args.no_mlflow,
    )
