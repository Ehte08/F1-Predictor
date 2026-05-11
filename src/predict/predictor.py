import pickle
from pathlib import Path

import pandas as pd

from src.config import MODEL_PATH
from src.features.engineer import set_categoricals


class F1Predictor:
    """Loads a trained LGBMRanker and exposes a single predict() method."""

    def __init__(self, model_path: Path = MODEL_PATH):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.feature_names: list[str] = bundle["feature_names"]
        self.training_cutoff: str = bundle.get("training_cutoff", "unknown")

    def predict(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score all drivers in a race DataFrame.

        race_df must contain the feature columns the model was trained on.
        Returns a copy sorted by pred_finish with 'pred_score' and 'pred_finish' added.
        """
        df = set_categoricals(race_df.copy())
        df["pred_score"] = self.model.predict(df[self.feature_names])
        df["pred_finish"] = (
            df["pred_score"].rank(ascending=False, method="first").astype(int)
        )
        return df.sort_values("pred_finish").reset_index(drop=True)
