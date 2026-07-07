"""Tests for feature engineering correctness — especially data leakage prevention."""
import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import TEAM_ALIASES, _nationality
from src.features.engineer import (
    ELO_INIT,
    add_elo_ratings,
    add_rolling_features,
    add_team_pace,
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

def test_ewm_uses_shift_no_leakage() -> None:
    """
    finish_ewm3 must be computed from *past* races only (shift(1), span=3 EWM).
    For race i the value must equal the EWM over finishes strictly before i.
    """
    df = make_race_df(n_drivers=3, n_races=6)
    result = add_rolling_features(df)

    for drv, g in result.groupby("driver"):
        g = g.sort_values("date").reset_index(drop=True)
        for i in range(1, len(g)):
            prior = g.loc[: i - 1, "finish"]
            expected = float(prior.ewm(span=3, min_periods=1).mean().iloc[-1])
            actual = g.loc[i, "finish_ewm3"]
            assert abs(actual - expected) < 1e-6, (
                f"{drv} race {i}: expected finish_ewm3={expected:.3f}, got {actual:.3f}"
            )


def test_reliability_never_includes_current_race() -> None:
    """driver_dnf for the current race must not affect driver_reliability_ewm10 that same race."""
    df = make_race_df(n_drivers=2, n_races=6)
    last_date = df["date"].max()
    df.loc[df["date"] == last_date, "driver_dnf"] = 1

    result = add_rolling_features(df)
    for drv, g in result.groupby("driver"):
        g = g.sort_values("date").reset_index(drop=True)
        # Reliability at the last race must equal 1 - EWM(shift(1) DNF) over prior races.
        prior_dnf = g.loc[: len(g) - 2, "driver_dnf"].astype(int)
        expected = 1 - float(prior_dnf.ewm(span=10, min_periods=1).mean().iloc[-1])
        actual = g.iloc[-1]["driver_reliability_ewm10"]
        assert abs(actual - expected) < 1e-6, (
            f"{drv}: reliability leaked current DNF (got {actual:.3f}, want {expected:.3f})"
        )


def test_elo_no_leakage() -> None:
    """
    driver_elo for race N uses the rating BEFORE race N.
    First race must be the init rating; a perpetual winner's rating must only rise
    on races *after* the wins (i.e. race 0 is untouched by its own result).
    """
    df = make_race_df(n_drivers=4, n_races=6)
    result = add_elo_ratings(df)

    for _, g in result.groupby("driver"):
        g = g.sort_values("date").reset_index(drop=True)
        assert abs(g.loc[0, "driver_elo"] - ELO_INIT) < 1e-9, "First race elo must equal init"

    # Consistent winner (Driver_0, finish=1) should have non-decreasing pre-race elo
    winner = result[result["driver"] == "Driver_0"].sort_values("date")
    elos = winner["driver_elo"].to_numpy()
    assert np.all(np.diff(elos) >= -1e-9), "Winner's pre-race elo should never fall"
    assert elos[-1] > elos[0], "Consistent winner should gain elo over time"


def test_team_pace_no_leakage() -> None:
    """team_pace_ewm5 at race i must equal the EWM of prior team-race mean finishes (shift(1))."""
    df = make_race_df(n_drivers=4, n_races=6)
    result = add_team_pace(df)

    # Rebuild expectation: per-team per-race mean finish, EWM span=5, shift(1)
    race_team = df.groupby(["date", "team"])["finish"].mean().reset_index(name="tf")
    race_team = race_team.sort_values("date")
    race_team["exp"] = race_team.groupby("team")["tf"].transform(
        lambda s: s.shift(1).ewm(span=5, min_periods=1).mean()
    )
    merged = result.merge(race_team[["date", "team", "exp"]], on=["date", "team"])
    both = merged.dropna(subset=["team_pace_ewm5", "exp"])
    assert np.allclose(both["team_pace_ewm5"], both["exp"]), "team_pace_ewm5 leaked current race"

    # First race for each team must be NaN (no prior history)
    first = result.sort_values("date").drop_duplicates("team", keep="first")
    assert first["team_pace_ewm5"].isna().all(), "First team race should have no pace history"


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
