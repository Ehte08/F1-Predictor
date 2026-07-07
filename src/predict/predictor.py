import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODEL_PATH
from src.features.engineer import FINISH_DERIVED_FEATURES, set_categoricals
from src.models.simulate import simulate_race


class F1Predictor:
    """
    Loads a trained bundle {model (ranker), dnf_model, feature_names, tau, ...} and
    exposes predict() (ranker order) plus predict_with_sim() which layers the DNF
    model and Plackett-Luce simulation to return win/podium/points/DNF probabilities.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.feature_names: list[str] = bundle["feature_names"]
        self.dnf_model = bundle.get("dnf_model")
        self.dnf_feature_names: list[str] = bundle.get(
            "dnf_feature_names",
            [f for f in self.feature_names if f not in FINISH_DERIVED_FEATURES],
        )
        self.tau: float = float(bundle.get("tau", 1.0))
        self.training_cutoff: str = bundle.get("training_cutoff", "unknown")

    def predict(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """Score all drivers; returns a copy sorted by pred_finish (ranker order)."""
        df = set_categoricals(race_df.copy())
        df["pred_score"] = self.model.predict(df[self.feature_names])
        df["pred_finish"] = (
            df["pred_score"].rank(ascending=False, method="first").astype(int)
        )
        return df.sort_values("pred_finish").reset_index(drop=True)

    def predict_with_sim(self, race_df: pd.DataFrame, n_sims: int = 10_000) -> pd.DataFrame:
        """
        Full prediction: ranker score + DNF probability + Plackett-Luce simulation.
        Adds p_win, p_podium, p_points, p_dnf, position_probs and a simulation-aware
        pred_finish (ranked by expected finishing position). Sorted by pred_finish.
        """
        df = set_categoricals(race_df.copy())
        scores = self.model.predict(df[self.feature_names])
        if self.dnf_model is not None:
            p_dnf = self.dnf_model.predict_proba(df[self.dnf_feature_names])[:, 1]
        else:
            p_dnf = np.zeros(len(df))

        sim = simulate_race(scores, p_dnf, tau=self.tau, n_sims=n_sims)
        n = len(scores)
        positions = np.arange(1, n + 1)
        expected_pos = (sim["position_probs"] * positions[None, :]).sum(axis=1)

        df["pred_score"] = scores
        df["p_win"] = sim["p_win"]
        df["p_podium"] = sim["p_podium"]
        df["p_points"] = sim["p_points"]
        df["p_dnf"] = sim["p_dnf"]
        df["position_probs"] = list(sim["position_probs"])
        df["expected_pos"] = expected_pos
        df["pred_finish"] = df["expected_pos"].rank(ascending=True, method="first").astype(int)
        return df.sort_values("pred_finish").reset_index(drop=True)
