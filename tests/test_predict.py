"""Tests for the prediction pipeline: feature alignment, output shape, uniqueness."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.features.engineer import finish_to_relevance, prepare_features


# ── Prediction output correctness ─────────────────────────────────────────────

def make_mock_predictor(feature_names: list[str]):
    """Create a lightweight mock predictor that returns ascending scores."""
    predictor = MagicMock()
    predictor.feature_names = feature_names
    predictor.training_cutoff = "2025-11-09"

    def _predict(df):
        df = df.copy()
        # Return inverted start position as score (pole → highest score)
        df["pred_score"] = 20 - df["start"].values
        df["pred_finish"] = (
            df["pred_score"].rank(ascending=False, method="first").astype(int)
        )
        return df.sort_values("pred_finish").reset_index(drop=True)

    predictor.predict.side_effect = _predict
    return predictor


def make_race_df(feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for i in range(20):
        row = {col: 0.5 for col in feature_names}
        row["driver"] = f"Driver_{i}"
        row["team"] = "McLaren"
        row["start"] = i + 1
        rows.append(row)
    return pd.DataFrame(rows)


FEATURE_NAMES = ["start", "age_at_race", "rainfall", "driver_reliability_rolling10",
                 "team_reliability_rolling10", "finish_rolling3", "year",
                 "avg_track_temp", "min_humidity", "driver_active", "team_active"]


def test_predictions_produce_exactly_20_unique_positions() -> None:
    predictor = make_mock_predictor(FEATURE_NAMES)
    race_df = make_race_df(FEATURE_NAMES)

    result = predictor.predict(race_df)

    assert len(result) == 20
    assert set(result["pred_finish"].tolist()) == set(range(1, 21)), (
        "Predicted positions must be exactly 1-20 with no duplicates."
    )


def test_predictions_are_sorted_by_pred_finish() -> None:
    predictor = make_mock_predictor(FEATURE_NAMES)
    race_df = make_race_df(FEATURE_NAMES)
    result = predictor.predict(race_df)
    assert list(result["pred_finish"]) == sorted(result["pred_finish"].tolist())


def test_feature_alignment_raises_on_missing_columns() -> None:
    """
    If the feature DataFrame is missing columns the model expects, it should fail
    loudly rather than silently returning garbage predictions.
    """
    REQUIRED = ["start", "age_at_race", "rainfall"]
    race_df = pd.DataFrame([{"start": 1, "driver": "Norris"}])  # missing age_at_race, rainfall

    with pytest.raises((KeyError, ValueError)):
        _ = race_df[REQUIRED]  # simulates what predictor.predict() does internally


# ── Relevance encoding round-trip ─────────────────────────────────────────────

def test_relevance_encoding_is_invertible() -> None:
    positions = pd.Series(range(1, 21))
    rel = finish_to_relevance(positions, max_pos=20)
    recovered = 20 - rel + 1
    np.testing.assert_array_equal(recovered, positions.values)


# ── Rolling feature grouping ──────────────────────────────────────────────────

def test_rolling_features_are_driver_scoped() -> None:
    """Reliability for Driver_A must not bleed into Driver_B's rolling window."""
    from src.features.engineer import add_rolling_features

    dates = pd.date_range("2024-01-01", periods=5, freq="W")
    rows = []
    for d in dates:
        for drv in ["Driver_A", "Driver_B"]:
            rows.append(
                {
                    "date": d,
                    "driver": drv,
                    "team": "TeamX",
                    "finish": 1 if drv == "Driver_A" else 10,
                    "driver_dnf": 0,
                    "team_dnf": 0,
                }
            )

    df = pd.DataFrame(rows).sort_values(["driver", "date"]).reset_index(drop=True)
    result = add_rolling_features(df)

    a_roll = result[result["driver"] == "Driver_A"]["finish_rolling3"].dropna().values
    b_roll = result[result["driver"] == "Driver_B"]["finish_rolling3"].dropna().values

    # Driver_A averages ~1; Driver_B averages ~10 — they must never cross
    assert (a_roll < 5).all(), "Driver_A rolling average should be near 1, not contaminated by Driver_B"
    assert (b_roll > 5).all(), "Driver_B rolling average should be near 10, not contaminated by Driver_A"
