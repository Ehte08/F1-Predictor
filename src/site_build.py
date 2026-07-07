"""
Shared site-artifact builders used by scripts/backfill.py, update_after_race.py and
predict_next.py. Wraps the walk-forward model pipeline + artifact schema so each script
stays a thin CLI.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from src.artifacts import (
    build_race_artifact,
    compute_metrics,
    export_browser_model,
    race_slug,
    winner_logloss,
    write_race_artifact,
)
from src.config import MODEL_PATH, SNAPSHOT_PATH
from src.features.engineer import build_feature_snapshot
from src.models.pipeline import engineer_full, predict_race, shap_for_race, train_upto


def build_eng(df: pd.DataFrame) -> dict:
    return engineer_full(df)


def _weather(df_r: pd.DataFrame, source: str) -> dict:
    r0 = df_r.iloc[0]
    return {
        "rainfall": int(r0.get("rainfall", 0)),
        "avg_track_temp": float(r0.get("avg_track_temp", 35.0)),
        "min_humidity": float(r0.get("min_humidity", 40.0)),
        "source": source,
    }


def backfill_race(
    eng: dict,
    df: pd.DataFrame,
    rounds: dict,
    date: pd.Timestamp,
    version: str,
    n_sims: int = 10_000,
    locked: bool = True,
    weather_source: str = "historical",
) -> tuple[str, dict]:
    """
    Train on races strictly before ``date``, predict that race from its actual grid +
    weather in the data, score against the real result, and write the artifact.
    Returns (slug, metrics_with_logloss).
    """
    date = pd.Timestamp(date)
    df_r = df[df["date"] == date]
    year = int(df_r["year"].iloc[0])
    race_name = str(df_r["GP name"].iloc[0])
    circuit = str(df_r["circuit name"].iloc[0])
    rnd = rounds[(year, date.strftime("%Y-%m-%d"))]
    slug = race_slug(year, rnd, race_name)

    bundle = train_upto(eng, date)

    X, meta, feats = eng["X"], eng["meta"], eng["feats"]
    mask = (meta["date"] == date).to_numpy()
    m_sub = meta.loc[mask]
    drivers = m_sub["driver"].astype(str).tolist()
    teams = m_sub["team"].astype(str).tolist()

    by_driver = df_r.set_index("driver")
    starts = [int(by_driver.loc[d, "start"]) for d in drivers]

    preds = predict_race(bundle, X.loc[mask], drivers, teams, starts, n_sims=n_sims)
    shap = shap_for_race(bundle["ranker"], X.loc[mask], feats, drivers)

    finish_map = by_driver["finish"].to_dict()
    preds["finish"] = preds["driver"].map(finish_map)
    metrics = compute_metrics(preds)

    actual_winner = min(finish_map, key=finish_map.get)
    widx = preds.index[preds["driver"] == actual_winner]
    metrics["winner_logloss"] = (
        winner_logloss(preds["p_win"].to_numpy(), int(widx[0]), n_sims) if len(widx) else float("nan")
    )

    grid = [
        {"driver": str(r["driver"]), "team": str(r["team"]), "start": int(r["start"])}
        for _, r in df_r.sort_values("start").iterrows()
    ]
    actual = [
        {
            "driver": str(r["driver"]),
            "team": str(r["team"]),
            "finish": int(r["finish"]),
            "status": "dnf" if (r["driver_dnf"] == 1 or r["team_dnf"] == 1) else "finished",
        }
        for _, r in df_r.sort_values("finish").iterrows()
    ]

    artifact = build_race_artifact(
        race_name=race_name,
        race_date=date.strftime("%Y-%m-%d"),
        year=year,
        rnd=rnd,
        circuit=circuit,
        version=version,
        weather=_weather(df_r, weather_source),
        grid=grid,
        predictions_df=preds,
        shap=shap,
        actual=actual,
        metrics={k: v for k, v in metrics.items() if k != "winner_logloss"},
        locked=locked,
    )
    write_race_artifact(slug, artifact)
    return slug, metrics


def predict_upcoming(
    eng: dict,
    df: pd.DataFrame,
    rounds: dict,
    *,
    race_name: str,
    race_date: str,
    year: int,
    rnd: int,
    circuit: str,
    grid: list[dict],
    weather: dict,
    version: str,
    n_sims: int = 10_000,
) -> str:
    """
    Predict a future race (no actuals yet) from a supplied grid + weather forecast.
    Builds inference features from the training snapshot; writes a locked artifact.
    """
    from src.predict.prep import build_race_features

    bundle = train_upto(eng, pd.Timestamp(df["date"].max()) + pd.Timedelta(days=1))
    feats = bundle["feats"]

    race_features = build_race_features(
        race_name=race_name,
        race_date=race_date,
        grid=grid,
        rainfall=int(weather.get("rainfall", 0)),
        avg_track_temp=float(weather.get("avg_track_temp", 35.0)),
        min_humidity=float(weather.get("min_humidity", 40.0)),
    )
    # build_race_features uses space-named GP/circuit cols; fill any missing model
    # feature columns (e.g. GP_name/circuit_name) with a neutral value.
    for col in feats:
        if col not in race_features.columns:
            race_features[col] = 0.5

    drivers = [g["driver"] for g in grid]
    teams = [g["team"] for g in grid]
    starts = [int(g["start"]) for g in grid]
    preds = predict_race(bundle, race_features, drivers, teams, starts, n_sims=n_sims)
    shap = shap_for_race(bundle["ranker"], race_features, feats, drivers)

    grid_out = [
        {"driver": g["driver"], "team": g["team"], "start": int(g["start"])}
        for g in sorted(grid, key=lambda x: x["start"])
    ]
    slug = race_slug(year, rnd, race_name)
    artifact = build_race_artifact(
        race_name=race_name,
        race_date=race_date,
        year=year,
        rnd=rnd,
        circuit=circuit,
        version=version,
        weather=weather,
        grid=grid_out,
        predictions_df=preds,
        shap=shap,
        actual=None,
        metrics=None,
        locked=True,
    )
    write_race_artifact(slug, artifact)
    return slug


def finalize_model(df: pd.DataFrame, eng: dict, version: str) -> dict:
    """Refit ranker+DNF+tau on all data; save pickle bundle, snapshot, browser model."""
    final = train_upto(eng, pd.Timestamp(df["date"].max()) + pd.Timedelta(days=1))
    training_cutoff = str(df["date"].max().date())
    snap = build_feature_snapshot(df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
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
    with open(SNAPSHOT_PATH, "wb") as f:
        pickle.dump(snap, f)

    export_browser_model(
        version=version,
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
    return {"tau": final["tau"], "training_cutoff": training_cutoff}


def summarize_metrics(rows: list[dict]) -> dict:
    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    return {
        "n_races": len(rows),
        "mean_spearman": float(np.nanmean(col("spearman"))),
        "mean_ndcg3": float(np.nanmean(col("ndcg3"))),
        "winner_accuracy": float(np.mean([bool(r["winner_correct"]) for r in rows])),
        "mean_podium_hits": float(np.mean(col("podium_hits"))),
        "mean_abs_delta": float(np.nanmean(col("mean_abs_delta"))),
        "mean_winner_logloss": float(np.nanmean(col("winner_logloss"))),
    }
