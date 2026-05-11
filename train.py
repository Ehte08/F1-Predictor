"""
Train the F1 race outcome predictor and save the model artifact.

Usage:
    python train.py
    python train.py --test-frac 0.15 --no-mlflow
    python train.py --save-path models/lgbm_ranker.pkl
"""
import argparse
import pickle
from pathlib import Path

import mlflow
import pandas as pd

from src.config import MODEL_PATH, SNAPSHOT_PATH, TEST_FRAC
from src.data.cleaner import build_base_dataframe
from src.data.fetcher import update_race_cache
from src.features.engineer import build_feature_snapshot, prepare_features
from src.models.evaluate import summarize
from src.models.train import fit, predict_on_df, time_based_split


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

    train_df, test_df = time_based_split(df, test_frac=test_frac)
    print(
        f"  Train: {train_df['date'].nunique()} races "
        f"({train_df['date'].min().date()} → {train_df['date'].max().date()})"
    )
    print(
        f"  Test:  {test_df['date'].nunique()} races "
        f"({test_df['date'].min().date()} → {test_df['date'].max().date()})"
    )

    print("\nFitting LGBMRanker on train split...")
    model, feature_names = fit(train_df)

    print("Evaluating on held-out test races...")
    X_test, _ = prepare_features(test_df)
    # prepare_features re-sorts by ["date", "driver"] — mirror that sort for the pred_df
    pred_df = test_df.sort_values(["date", "driver"]).reset_index(drop=True).copy()
    pred_df["pred_score"] = model.predict(X_test[feature_names])
    pred_df["pred_finish"] = (
        pred_df.groupby("date")["pred_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    metrics = summarize(pred_df)

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
            mlflow.log_metrics(metrics)
            print("  MLflow run logged.")

    # Refit on all data before saving so predictions use the full history
    print("\nRefitting on full dataset before saving...")
    model, feature_names = fit(df)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_names": feature_names,
                "training_cutoff": str(df["date"].max().date()),
            },
            f,
        )
    print(f"  Model saved → {save_path}")

    # Save feature snapshot for inference without rerunning the pipeline
    snap = build_feature_snapshot(df)
    with open(SNAPSHOT_PATH, "wb") as f:
        pickle.dump(snap, f)
    print(f"  Feature snapshot saved → {SNAPSHOT_PATH}")

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
