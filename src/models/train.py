import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from src.features.engineer import finish_to_relevance, prepare_features

# Best hyperparameters found via 30-trial Optuna search (time-series CV, NDCG@10 objective)
# Improved baseline: Mean Spearman=0.676, NDCG@3=0.891 → tuned: 0.688, 0.907
BEST_PARAMS: dict = {
    "n_estimators": 534,
    "learning_rate": 0.01593,
    "num_leaves": 62,
    "min_data_in_leaf": 5,
    "feature_fraction": 0.6029,
    "bagging_fraction": 0.6104,
    "bagging_freq": 1,
}


def build_ranker(**overrides) -> LGBMRanker:
    params = {**BEST_PARAMS, **overrides}
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        importance_type="gain",
        verbosity=-1,
        **params,
    )


def time_based_split(
    df: pd.DataFrame, test_frac: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split races chronologically: earlier races for train, most recent for test."""
    all_dates = np.sort(df["date"].unique())
    cutoff = int(len(all_dates) * (1 - test_frac))
    train_dates, test_dates = all_dates[:cutoff], all_dates[cutoff:]
    return df[df["date"].isin(train_dates)].copy(), df[df["date"].isin(test_dates)].copy()


def fit(
    df: pd.DataFrame, finishers_only: bool = True, **model_overrides
) -> tuple[LGBMRanker, list[str]]:
    """
    Prepare features and fit LGBMRanker on the supplied DataFrame.

    When ``finishers_only`` (default) DNF rows are dropped from the ranker's training
    set and groups are recomputed — the ranker then learns the finishing order among
    cars that make the flag, and P(DNF) is handled separately by the DNF classifier /
    Plackett-Luce layer. Set ``finishers_only=False`` for the legacy full-field path.

    Returns the fitted model and its feature name list.
    """
    X, y, meta = prepare_features(df, return_meta=True)

    if finishers_only:
        keep = (meta["dnf"] == 0).to_numpy()
        X = X.loc[keep].reset_index(drop=True)
        y = y.loc[keep].reset_index(drop=True)
        groups = meta.loc[keep].groupby("date", sort=True).size().tolist()
    else:
        # meta is already in ["date", "driver"] order, matching X
        groups = meta.groupby("date", sort=True).size().tolist()

    y_rel = finish_to_relevance(y, max_pos=int(y.max()))
    model = build_ranker(**model_overrides)
    model.fit(X, y_rel, group=groups)
    return model, model.feature_name_


def predict_on_df(model: LGBMRanker, feature_names: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Run model inference on a prepared DataFrame.
    Adds 'pred_score' and 'pred_finish' columns; sorts by predicted position.
    """
    from src.features.engineer import set_categoricals

    df = df.copy()
    df = set_categoricals(df)

    X = df[feature_names]
    df["pred_score"] = model.predict(X)
    df["pred_finish"] = (
        df.groupby("date")["pred_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    return df
