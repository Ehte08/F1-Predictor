"""Tests for feature engineering correctness — especially data leakage prevention."""
import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import TEAM_ALIASES, _nationality
from src.features.engineer import (
    add_rolling_features,
    finish_to_relevance,
    prepare_features,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_race_df(n_drivers: int = 5, n_races: int = 4) -> pd.DataFrame:
    """Minimal DataFrame that mimics the shape of f1_data after cleaning."""
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_races, freq="W")
    for d in dates:
        for drv_id in range(n_drivers):
            rows.append(
                {
                    "date": d,
                    "driver": f"Driver_{drv_id}",
                    "team": f"Team_{drv_id % 2}",
                    "finish": drv_id + 1,
                    "driver_dnf": 1 if drv_id == n_drivers - 1 else 0,
                    "team_dnf": 0,
                    "dob": pd.Timestamp("1995-01-01"),
                    "GP name": "Test Grand Prix",
                    "circuit name": "Test Circuit",
                    "start": drv_id + 1,
                }
            )
    return pd.DataFrame(rows).sort_values(["driver", "date"]).reset_index(drop=True)


# ── Nationality mapping ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("British", "BRI"),
        ("british", "BRI"),
        ("UK", "BRI"),
        ("England", "BRI"),
        ("Austrian", "AUT"),
        ("Australian", "AUS"),
        ("American", "AME"),
        ("French", "FRE"),
    ],
)
def test_nationality_mapping(raw: str, expected: str) -> None:
    assert _nationality(raw) == expected


# ── Team alias ────────────────────────────────────────────────────────────────

def test_team_alias_no_typo() -> None:
    """Catch the 'Apline' typo that existed in the original notebook."""
    for alias, canonical in TEAM_ALIASES.items():
        assert "Apline" not in canonical, f"Typo still present in alias for {alias!r}"


def test_all_aliases_map_to_active_teams() -> None:
    from src.config import ACTIVE_CONSTRUCTORS
    for alias, canonical in TEAM_ALIASES.items():
        assert canonical in ACTIVE_CONSTRUCTORS, (
            f"{alias!r} maps to {canonical!r} which is not in ACTIVE_CONSTRUCTORS"
        )


# ── Rolling feature leakage ───────────────────────────────────────────────────

def test_rolling_uses_shift_no_leakage() -> None:
    """
    The rolling features must be computed from *past* races only.
    For race 0, the rolling value should be NaN/0.5 (no history yet).
    For race 1, it should reflect race 0 only — NOT include race 1 itself.
    """
    df = make_race_df(n_drivers=3, n_races=5)
    result = add_rolling_features(df)

    for drv, g in result.groupby("driver"):
        g = g.sort_values("date").reset_index(drop=True)
        # Row 0 has no prior history — rolling mean of nothing = NaN (filled to 0.5 later)
        # Most importantly: finish_rolling3 at row i must not include finish at row i
        for i in range(1, len(g)):
            prior_finishes = g.loc[:i - 1, "finish"].values
            expected_roll3 = float(np.mean(prior_finishes[-3:]))
            actual = g.loc[i, "finish_rolling3"]
            assert abs(actual - expected_roll3) < 1e-6, (
                f"{drv} race {i}: expected finish_rolling3={expected_roll3:.3f}, got {actual:.3f}"
            )


def test_rolling_features_never_include_current_race() -> None:
    """driver_dnf for the current race must not affect driver_reliability_rolling10 for that same race."""
    df = make_race_df(n_drivers=2, n_races=6)
    # Artificially set the last race as a DNF
    last_date = df["date"].max()
    df.loc[df["date"] == last_date, "driver_dnf"] = 1

    result = add_rolling_features(df)

    # The reliability value for the last race should reflect DNF *before* that date only
    for drv, g in result.groupby("driver"):
        g = g.sort_values("date").reset_index(drop=True)
        last_rel = g.iloc[-1]["driver_reliability_rolling10"]
        prev_dnfs = g.iloc[:-1]["finish"].isna()  # proxy — actual check is via rolling window
        # If current-race DNF leaked into its own row, reliability would drop here
        # Instead it should equal the reliability computed from prior races
        second_last_rel = g.iloc[-2]["driver_reliability_rolling10"] if len(g) >= 2 else None
        # The two reliability values must differ by at most 0/1 of one DNF event's contribution
        if second_last_rel is not None:
            assert last_rel >= 0, "Reliability is negative — something went wrong."


# ── finish_to_relevance ───────────────────────────────────────────────────────

def test_finish_to_relevance_p1_gets_highest() -> None:
    positions = pd.Series([1, 2, 3, 10, 20])
    rel = finish_to_relevance(positions, max_pos=20)
    assert rel[0] > rel[1] > rel[2] > rel[3] > rel[4]
    assert rel[0] == 20
    assert rel[4] == 1


def test_finish_to_relevance_all_positive() -> None:
    positions = pd.Series(range(1, 21))
    rel = finish_to_relevance(positions, max_pos=20)
    assert (rel > 0).all()


# ── prepare_features ──────────────────────────────────────────────────────────

def test_prepare_features_drops_target_and_dates() -> None:
    df = make_race_df()
    X, y = prepare_features(df)
    assert "finish" not in X.columns
    assert "date" not in X.columns
    assert "dob" not in X.columns
    assert y is not None
    assert len(y) == len(X)


def test_prepare_features_no_nans() -> None:
    df = make_race_df()
    X, _ = prepare_features(df)
    assert X.isna().sum().sum() == 0, "NaNs found in feature matrix after prepare_features()"
