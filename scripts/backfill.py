"""
Walk-forward backfill from 2025 onward — the honest retroactive archive.

For every race dated >= 2025-01-01 in the dataset: train the ranker + DNF model + tau
on all races strictly BEFORE it, predict it from the actual grid + actual weather,
compute SHAP, run the Plackett-Luce simulation, score it against the real result, and
write one artifact JSON per race under data/site/. Then refit on all data and export
index.json + model/browser_model.json.

Run:  python scripts/backfill.py [--n-sims 10000] [--from 2025-01-01]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from src.artifacts import compute_rounds, model_version, rebuild_index  # noqa: E402
from src.data.cleaner import build_base_dataframe  # noqa: E402
from src.site_build import (  # noqa: E402
    backfill_race,
    build_eng,
    finalize_model,
    summarize_metrics,
)


def run(n_sims: int = 10_000, from_date: str = "2025-01-01") -> dict:
    version = model_version()
    print(f"Model version: {version}")
    print("Loading data...")
    df = build_base_dataframe()
    rounds = compute_rounds(df)

    print("Engineering features (once)...")
    eng = build_eng(df)

    target_dates = sorted(
        d for d in eng["meta"]["date"].unique() if pd.Timestamp(d) >= pd.Timestamp(from_date)
    )
    print(f"{len(target_dates)} races to backfill (>= {from_date}).\n")

    rows = []
    for d in target_dates:
        slug, m = backfill_race(eng, df, rounds, d, version, n_sims=n_sims)
        rows.append(m)
        print(
            f"  {slug:<38} spearman={m['spearman']:.3f} ndcg3={m['ndcg3']:.3f} "
            f"winner={'Y' if m['winner_correct'] else 'n'} podium={m['podium_hits']}/3"
        )

    print("\nRefitting on all data for the shipped model...")
    finalize_model(df, eng, version)
    rebuild_index(version)

    summary = summarize_metrics(rows)
    print("\n── Walk-forward metrics ─────────────────────")
    for k, v in summary.items():
        print(f"  {k:<22} {v:.4f}")
    print("─────────────────────────────────────────────")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Walk-forward backfill from 2025.")
    ap.add_argument("--n-sims", type=int, default=10_000)
    ap.add_argument("--from", dest="from_date", default="2025-01-01")
    args = ap.parse_args()
    run(n_sims=args.n_sims, from_date=args.from_date)
