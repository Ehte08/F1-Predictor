"""
Two-stage DNF model: an LGBMClassifier that predicts P(DNF) per driver-race.

Label = driver_dnf | team_dnf. Uses the ranker feature set minus finish-derived
columns (finish_ewm3, driver_circuit_avg, team_pace_ewm5) so it leans on
reliability / pace / grid / quali signals rather than raw past finishing position.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from src.features.engineer import (
    FINISH_DERIVED_FEATURES,
    prepare_features,
    set_categoricals,
)

DNF_PARAMS: dict = {
    "n_estimators": 300,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
}


def dnf_feature_names(all_features: list[str]) -> list[str]:
    """Ranker features minus finish-derived leakage."""
    return [f for f in all_features if f not in FINISH_DERIVED_FEATURES]


def build_classifier(**overrides) -> LGBMClassifier:
    params = {**DNF_PARAMS, **overrides}
    return LGBMClassifier(
        objective="binary",
        importance_type="gain",
        verbosity=-1,
        **params,
    )


def fit_dnf(df: pd.DataFrame, **overrides) -> tuple[LGBMClassifier, list[str]]:
    """Fit the DNF classifier on the full engineered feature matrix."""
    X, _, meta = prepare_features(df, return_meta=True)
    feats = dnf_feature_names(list(X.columns))
    y = meta["dnf"].astype(int).to_numpy()
    model = build_classifier(**overrides)
    model.fit(X[feats], y)
    return model, feats


def predict_dnf(model: LGBMClassifier, feats: list[str], df: pd.DataFrame) -> np.ndarray:
    """P(DNF) for each row of an already-engineered feature DataFrame."""
    X = set_categoricals(df.copy())
    return model.predict_proba(X[feats])[:, 1]


def evaluate_dnf(df: pd.DataFrame, test_frac: float = 0.20, **overrides) -> dict:
    """Time-split fit/evaluate; report ROC-AUC on the held-out recent races."""
    X, _, meta = prepare_features(df, return_meta=True)
    feats = dnf_feature_names(list(X.columns))
    y = meta["dnf"].astype(int).to_numpy()

    dates = np.sort(meta["date"].unique())
    cutoff = dates[int(len(dates) * (1 - test_frac))]
    train_mask = (meta["date"] < cutoff).to_numpy()
    test_mask = ~train_mask

    model = build_classifier(**overrides)
    model.fit(X.loc[train_mask, feats], y[train_mask])
    proba = model.predict_proba(X.loc[test_mask, feats])[:, 1]

    auc = (
        float(roc_auc_score(y[test_mask], proba))
        if len(np.unique(y[test_mask])) > 1
        else float("nan")
    )
    return {
        "dnf_auc": auc,
        "test_dnf_rate": float(y[test_mask].mean()),
        "n_test": int(test_mask.sum()),
    }
