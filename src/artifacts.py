"""
Site-artifact schema, metrics and IO. Emits the exact JSON contract other agents
consume under data/site/: index.json, races/{slug}.json, model/browser_model.json.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

from src.config import ROOT

SITE_DIR = ROOT / "data" / "site"
RACES_DIR = SITE_DIR / "races"
MODEL_DIR_SITE = SITE_DIR / "model"


def model_version(seq: int = 1) -> str:
    return f"{date.today():%Y.%m.%d}-{seq}"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def kebab(name: str) -> str:
    s = name.lower().replace("ã", "a").replace("é", "e").replace("ô", "o")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def compute_rounds(df: pd.DataFrame) -> dict:
    """(year, 'YYYY-MM-DD') -> round number (calendar order within the year)."""
    rounds = {}
    for year, g in df[["year", "date"]].drop_duplicates().groupby("year"):
        for i, d in enumerate(sorted(g["date"].unique()), start=1):
            rounds[(int(year), pd.Timestamp(d).strftime("%Y-%m-%d"))] = i
    return rounds


def race_slug(year: int, rnd: int, race_name: str) -> str:
    return f"{year}-{rnd:02d}-{kebab(race_name)}"


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(race_df: pd.DataFrame, max_pos: int = 20) -> dict:
    """
    race_df needs columns: finish, pred_finish, pred_score (one row per driver).
    Returns spearman, ndcg@3/10/20, winner_correct, podium_hits, mean_abs_delta.
    """
    finish = race_df["finish"].to_numpy(dtype=float)
    pred_finish = race_df["pred_finish"].to_numpy(dtype=float)
    pred_score = race_df["pred_score"].to_numpy(dtype=float)

    corr, _ = spearmanr(finish, pred_finish)
    rel = (max_pos - finish + 1).clip(min=0).reshape(1, -1)
    scores = pred_score.reshape(1, -1)
    n = len(finish)
    ndcg = {k: float(ndcg_score(rel, scores, k=min(k, n))) for k in (3, 10, 20)}

    pred_winner = race_df.loc[race_df["pred_finish"] == 1, "driver"]
    actual_winner = race_df.loc[race_df["finish"] == race_df["finish"].min(), "driver"]
    winner_correct = bool(len(pred_winner) and pred_winner.iloc[0] in set(actual_winner))

    pred_podium = set(race_df.nsmallest(3, "pred_finish")["driver"])
    actual_podium = set(race_df.nsmallest(3, "finish")["driver"])
    podium_hits = int(len(pred_podium & actual_podium))

    mean_abs_delta = float(np.mean(np.abs(pred_finish - finish)))

    return {
        "spearman": float(corr) if not np.isnan(corr) else 0.0,
        "ndcg3": ndcg[3],
        "ndcg10": ndcg[10],
        "ndcg20": ndcg[20],
        "winner_correct": winner_correct,
        "podium_hits": podium_hits,
        "mean_abs_delta": mean_abs_delta,
    }


def winner_logloss(p_win: np.ndarray, winner_idx: int, n_sims: int = 10000) -> float:
    eps = 1.0 / (n_sims * 10.0)
    return float(-np.log(max(float(p_win[winner_idx]), eps)))


# ── Artifact assembly ─────────────────────────────────────────────────────────

def _round_floats(v, nd=6):
    if isinstance(v, float):
        return round(v, nd)
    if isinstance(v, list):
        return [_round_floats(x, nd) for x in v]
    if isinstance(v, dict):
        return {k: _round_floats(x, nd) for k, x in v.items()}
    return v


def build_race_artifact(
    *,
    race_name: str,
    race_date: str,
    year: int,
    rnd: int,
    circuit: str,
    version: str,
    weather: dict,
    grid: list[dict],
    predictions_df: pd.DataFrame,
    shap: dict,
    actual: list[dict] | None,
    metrics: dict | None,
    locked: bool,
) -> dict:
    preds = []
    for _, r in predictions_df.iterrows():
        preds.append(
            {
                "driver": str(r["driver"]),
                "team": str(r["team"]),
                "start": int(r["start"]),
                "pred_finish": int(r["pred_finish"]),
                "pred_score": round(float(r["pred_score"]), 6),
                "p_win": round(float(r["p_win"]), 6),
                "p_podium": round(float(r["p_podium"]), 6),
                "p_points": round(float(r["p_points"]), 6),
                "p_dnf": round(float(r["p_dnf"]), 6),
                "position_probs": [round(float(x), 6) for x in r["position_probs"]],
            }
        )
    return {
        "race_name": race_name,
        "race_date": race_date,
        "year": int(year),
        "round": int(rnd),
        "circuit": circuit,
        "model_version": version,
        "generated_at": now_iso(),
        "locked": bool(locked),
        "weather": weather,
        "grid": grid,
        "predictions": preds,
        "shap": _round_floats(shap),
        "actual": actual,
        "metrics": _round_floats(metrics) if metrics else None,
    }


def write_race_artifact(slug: str, artifact: dict) -> Path:
    RACES_DIR.mkdir(parents=True, exist_ok=True)
    path = RACES_DIR / f"{slug}.json"
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    return path


def rebuild_index(version: str, next_race: dict | None = None) -> Path:
    """Scan races/*.json and (re)write index.json."""
    RACES_DIR.mkdir(parents=True, exist_ok=True)
    races, track_record = [], []
    for p in sorted(RACES_DIR.glob("*.json")):
        with open(p) as f:
            a = json.load(f)
        races.append(
            {
                "slug": p.stem,
                "race_name": a["race_name"],
                "race_date": a["race_date"],
                "year": a["year"],
                "round": a["round"],
                "has_actual": a.get("actual") is not None,
                "locked": a.get("locked", False),
            }
        )
        if a.get("metrics"):
            m = a["metrics"]
            track_record.append(
                {
                    "slug": p.stem,
                    "race_name": a["race_name"],
                    "race_date": a["race_date"],
                    "spearman": m["spearman"],
                    "ndcg3": m["ndcg3"],
                    "winner_correct": m["winner_correct"],
                    "podium_hits": m["podium_hits"],
                    "mean_abs_delta": m["mean_abs_delta"],
                }
            )
    races.sort(key=lambda r: r["race_date"])
    track_record.sort(key=lambda r: r["race_date"])

    index = {
        "updated_at": now_iso(),
        "model_version": version,
        "races": races,
        "next_race": next_race,
        "track_record": track_record,
    }
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    path = SITE_DIR / "index.json"
    with open(path, "w") as f:
        json.dump(index, f, indent=2, default=str)
    return path


def export_browser_model(
    *,
    version: str,
    trained_through: str,
    ranker,
    dnf_model,
    feats: list[str],
    X_all: pd.DataFrame,
    tau: float,
    snapshot: dict,
    circuit_map: dict,
    driver_ages: dict,
) -> Path:
    """Write a self-contained model dump for in-browser scoring."""
    cat_cols = [c for c in ["GP_name", "team", "driver", "circuit_name"] if c in X_all.columns]
    categorical_features = {
        c: [str(x) for x in X_all[c].cat.categories.tolist()] for c in cat_cols
    }

    driver_features = {
        d: {
            "reliability": round(v["reliability"], 6),
            "finish_ewm3": round(v["finish_ewm3"], 4),
            "elo": round(v["elo"], 2),
            "circuit_avgs": {k: round(a, 4) for k, a in v["circuit_avgs"].items()},
        }
        for d, v in snapshot["driver_features"].items()
    }
    team_features = {
        t: {"reliability": round(v["reliability"], 6), "pace_ewm5": round(v["pace_ewm5"], 4)}
        for t, v in snapshot["team_features"].items()
    }

    payload = {
        "model_version": version,
        "trained_through": trained_through,
        "feature_names": feats,
        "categorical_features": categorical_features,
        "booster": ranker.booster_.dump_model(),
        "dnf_booster": dnf_model.booster_.dump_model(),
        "pl": {"tau": round(float(tau), 4), "n_sims_recommended": 2000},
        "snapshot": {
            "driver_features": driver_features,
            "team_features": team_features,
            "driver_ages": {d: round(float(a), 3) for d, a in driver_ages.items()},
            "circuit_map": circuit_map,
            "quali_slot_gap": snapshot.get("quali_slot_gap", {}),
            "quali_global_gap": snapshot.get("quali_global_gap", 5.0),
        },
    }
    MODEL_DIR_SITE.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR_SITE / "browser_model.json"
    with open(path, "w") as f:
        json.dump(payload, f, default=str)
    return path
