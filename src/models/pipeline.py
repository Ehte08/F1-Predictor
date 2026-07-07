"""
Walk-forward training / prediction / simulation orchestration.

Engineers the full dataset once, then trains the ranker + DNF classifier on any
"strictly-before-cutoff" window, calibrates the Plackett-Luce tau, and produces a
per-driver prediction table with simulation probabilities and SHAP explanations.
Shared by scripts/backfill.py, scripts/update_after_race.py and scripts/predict_next.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engineer import prepare_features, set_categoricals
from src.models.dnf import build_classifier, dnf_feature_names
from src.models.simulate import DEFAULT_TAU, calibrate_tau, simulate_race
from src.models.train import build_ranker, finish_to_relevance


def engineer_full(df: pd.DataFrame) -> dict:
    """Engineer the whole dataset once. Returns X, meta, feature name lists."""
    X, y, meta = prepare_features(df, return_meta=True)
    feats = list(X.columns)
    return {
        "X": X,
        "y": y,
        "meta": meta,          # driver, team, date, finish, dnf  (row-aligned with X)
        "feats": feats,
        "dnf_feats": dnf_feature_names(feats),
    }


def train_upto(
    eng: dict,
    cutoff: pd.Timestamp,
    finishers_only: bool = True,
    tau_val_races: int = 12,
    tau_n_sims: int = 1500,
) -> dict:
    """
    Train ranker + DNF model on all rows strictly before ``cutoff`` and calibrate tau
    on the most-recent ``tau_val_races`` races before the cutoff.
    """
    X, meta, feats, dnf_feats = eng["X"], eng["meta"], eng["feats"], eng["dnf_feats"]
    y = eng["y"]

    train = (meta["date"] < cutoff).to_numpy()

    # Ranker (finishers only by default)
    if finishers_only:
        rmask = train & (meta["dnf"] == 0).to_numpy()
    else:
        rmask = train
    Xr = X.loc[rmask]
    yr = y.loc[rmask]
    groups = meta.loc[rmask].groupby("date", sort=True).size().tolist()
    ranker = build_ranker()
    ranker.fit(Xr, finish_to_relevance(yr, max_pos=int(yr.max())), group=groups)

    # DNF classifier (full field before cutoff)
    dnf_model = build_classifier()
    dnf_model.fit(X.loc[train, dnf_feats], meta.loc[train, "dnf"].astype(int).to_numpy())

    # Calibrate tau on a HELD-OUT window: scores for the validation races come from a
    # ranker that never saw them, so tau reflects genuine predictive spread (avoids the
    # over-confidence collapse you get scoring races the model trained on).
    train_dates = np.sort(meta.loc[train, "date"].unique())
    tau = DEFAULT_TAU
    if len(train_dates) > tau_val_races + 5:
        val_dates = train_dates[-tau_val_races:]
        val_start = val_dates[0]
        cal_train = (meta["date"] < val_start).to_numpy()
        cal_rmask = cal_train & (meta["dnf"] == 0).to_numpy() if finishers_only else cal_train
        cal_ranker = build_ranker()
        cal_groups = meta.loc[cal_rmask].groupby("date", sort=True).size().tolist()
        cal_ranker.fit(
            X.loc[cal_rmask],
            finish_to_relevance(y.loc[cal_rmask], max_pos=int(y.loc[cal_rmask].max())),
            group=cal_groups,
        )
        val_races = []
        for d in val_dates:
            m = (meta["date"] == d).to_numpy()
            scores = cal_ranker.predict(X.loc[m, feats])
            winner_local = int(np.argmin(meta.loc[m, "finish"].to_numpy()))
            val_races.append(
                {"scores": scores, "p_dnf": np.zeros(len(scores)), "winner_idx": winner_local}
            )
        tau = calibrate_tau(val_races, n_sims=tau_n_sims)

    return {
        "ranker": ranker,
        "dnf_model": dnf_model,
        "feats": feats,
        "dnf_feats": dnf_feats,
        "tau": tau,
    }


def _rank_desc(scores: np.ndarray) -> np.ndarray:
    """1-based ranks (highest score = 1), stable ties by first occurrence."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def predict_race(
    bundle: dict,
    X_race: pd.DataFrame,
    drivers: list[str],
    teams: list[str],
    starts: list[int],
    n_sims: int = 10_000,
) -> pd.DataFrame:
    """
    Score one race and run the Plackett-Luce simulation.
    Returns a DataFrame (row per driver) with pred_score, p_dnf and sim probabilities.
    pred_finish ranks drivers by expected simulated finishing position.
    """
    feats, dnf_feats = bundle["feats"], bundle["dnf_feats"]
    Xc = set_categoricals(X_race.copy())
    scores = bundle["ranker"].predict(Xc[feats])
    p_dnf = bundle["dnf_model"].predict_proba(Xc[dnf_feats])[:, 1]

    sim = simulate_race(scores, p_dnf, tau=bundle["tau"], n_sims=n_sims)
    n = len(scores)
    positions = np.arange(1, n + 1)
    expected_pos = (sim["position_probs"] * positions[None, :]).sum(axis=1)
    pred_finish = _rank_desc(-expected_pos)  # lower expected pos = better = rank 1

    out = pd.DataFrame(
        {
            "driver": drivers,
            "team": teams,
            "start": starts,
            "pred_score": scores,
            "p_win": sim["p_win"],
            "p_podium": sim["p_podium"],
            "p_points": sim["p_points"],
            "p_dnf": sim["p_dnf"],
            "pred_finish": pred_finish,
            "expected_pos": expected_pos,
        }
    )
    out["position_probs"] = list(sim["position_probs"])
    return out.sort_values("pred_finish").reset_index(drop=True)


def shap_for_race(ranker, X_race: pd.DataFrame, feats: list[str], drivers: list[str], top_k: int = 8) -> dict:
    """
    Per-driver top-k features by |SHAP| for the ranker. Returns
    {"base_value": float, "drivers": {driver: [{feature, value, shap}, ...]}}.
    """
    import shap

    Xc = set_categoricals(X_race.copy())[feats]
    explainer = shap.TreeExplainer(ranker)
    sv = explainer.shap_values(Xc)
    sv = np.asarray(sv)
    if sv.ndim == 3:  # some versions return [classes, rows, feats]
        sv = sv[0]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.ravel(base)[0])
    else:
        base = float(base)

    out_drivers = {}
    for i, drv in enumerate(drivers):
        row = sv[i]
        idx = np.argsort(-np.abs(row))[:top_k]
        items = []
        for j in idx:
            val = Xc.iloc[i, j]
            if hasattr(val, "item"):
                val = val.item()
            items.append(
                {"feature": feats[j], "value": _jsonify(val), "shap": float(row[j])}
            )
        out_drivers[drv] = items
    return {"base_value": base, "drivers": out_drivers}


def _jsonify(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if pd.isna(v) if np.isscalar(v) else False:
        return None
    return str(v) if not isinstance(v, (int, float, bool)) else v
