import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score


def spearman_per_race(pred_df: pd.DataFrame) -> np.ndarray:
    scores = []
    for _, g in pred_df.groupby("date"):
        corr, _ = spearmanr(g["finish"].values, g["pred_finish"].values)
        scores.append(corr)
    return np.array(scores)


def ndcg_per_race(pred_df: pd.DataFrame, k_list=(3, 10, 20), max_pos: int = 20) -> dict[int, float]:
    results: dict[int, list] = {k: [] for k in k_list}
    for _, g in pred_df.groupby("date"):
        rel = (max_pos - g["finish"].values + 1).astype(float)
        scores = g["pred_score"].values
        for k in k_list:
            val = ndcg_score(rel.reshape(1, -1), scores.reshape(1, -1), k=min(k, len(g)))
            results[k].append(val)
    return {k: float(np.mean(v)) for k, v in results.items()}


def summarize(pred_df: pd.DataFrame) -> dict:
    spearman = spearman_per_race(pred_df)
    ndcg = ndcg_per_race(pred_df)
    return {
        "mean_spearman": float(np.nanmean(spearman)),
        "median_spearman": float(np.nanmedian(spearman)),
        "min_spearman": float(np.nanmin(spearman)),
        "max_spearman": float(np.nanmax(spearman)),
        "ndcg_at_3": ndcg[3],
        "ndcg_at_10": ndcg[10],
        "ndcg_at_20": ndcg[20],
    }
