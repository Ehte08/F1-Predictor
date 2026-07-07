"""
Post-race update: fetch newly completed races, retrain everything, (re)generate the
affected race artifacts with actuals + metrics, and refresh index.json +
browser_model.json + the saved model pickles.

Idempotent: only regenerates race artifacts that are missing or still lack actuals
(e.g. a previously-locked prediction whose race has now run). Safe to re-run.

Run:  python scripts/update_after_race.py [--n-sims 10000]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from src.artifacts import RACES_DIR, SITE_DIR, compute_rounds, model_version, race_slug, rebuild_index  # noqa: E402
from src.data.cleaner import build_base_dataframe  # noqa: E402
from src.data.fetcher import update_race_cache  # noqa: E402
from src.site_build import backfill_race, build_eng, finalize_model  # noqa: E402


def _existing_next_race() -> dict | None:
    idx = SITE_DIR / "index.json"
    if not idx.exists():
        return None
    with open(idx) as f:
        return json.load(f).get("next_race")


def _artifact_needs_actuals(slug: str) -> bool:
    """True if the artifact is missing or has no recorded actual result yet."""
    path = RACES_DIR / f"{slug}.json"
    if not path.exists():
        return True
    with open(path) as f:
        return json.load(f).get("actual") is None


def run(n_sims: int = 10_000, from_date: str = "2025-01-01") -> dict:
    version = model_version()
    print("Fetching new race data...")
    n_new = update_race_cache(verbose=True)
    print(f"  {n_new} new race(s) fetched.\n")

    df = build_base_dataframe()
    rounds = compute_rounds(df)
    eng = build_eng(df)

    # Races that have real results but whose artifact is missing / still un-resolved.
    to_generate = []
    for d in sorted(eng["meta"]["date"].unique()):
        d = pd.Timestamp(d)
        if d < pd.Timestamp(from_date):
            continue
        df_r = df[df["date"] == d]
        year = int(df_r["year"].iloc[0])
        rnd = rounds[(year, d.strftime("%Y-%m-%d"))]
        slug = race_slug(year, rnd, str(df_r["GP name"].iloc[0]))
        if _artifact_needs_actuals(slug):
            to_generate.append(d)

    print(f"{len(to_generate)} race artifact(s) to (re)generate with actuals.")
    for d in to_generate:
        slug, m = backfill_race(eng, df, rounds, d, version, n_sims=n_sims)
        print(f"  {slug}: spearman={m['spearman']:.3f} winner={'Y' if m['winner_correct'] else 'n'}")

    print("\nRefitting shipped model on all data...")
    finalize_model(df, eng, version)

    # Preserve the pending next-race prediction if it hasn't been run yet.
    next_race = _existing_next_race()
    if next_race and not _artifact_needs_actuals(next_race.get("slug", "")):
        next_race = None  # that race now has actuals; clear it
    rebuild_index(version, next_race=next_race)
    print("Updated index.json + browser_model.json + model pickles.")
    return {"n_new": n_new, "n_regenerated": len(to_generate)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Post-race update of all artifacts.")
    ap.add_argument("--n-sims", type=int, default=10_000)
    ap.add_argument("--from", dest="from_date", default="2025-01-01")
    args = ap.parse_args()
    run(n_sims=args.n_sims, from_date=args.from_date)
