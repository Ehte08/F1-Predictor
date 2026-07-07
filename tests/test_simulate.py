"""Tests for the Plackett-Luce Monte-Carlo simulation layer."""
import numpy as np

from src.models.simulate import calibrate_tau, simulate_race


def _base_scores(n: int = 10) -> np.ndarray:
    # Strictly descending scores: driver 0 strongest.
    return np.linspace(3.0, -3.0, n)


def test_probabilities_in_unit_interval() -> None:
    n = 10
    sim = simulate_race(_base_scores(n), np.full(n, 0.1), tau=0.6, n_sims=5000)
    for key in ("p_win", "p_podium", "p_points", "p_dnf"):
        v = sim[key]
        assert np.all(v >= 0.0) and np.all(v <= 1.0), f"{key} out of [0,1]"
    assert np.all(sim["position_probs"] >= 0.0)
    assert np.all(sim["position_probs"] <= 1.0)


def test_position_probs_row_sums_to_one() -> None:
    n = 12
    sim = simulate_race(_base_scores(n), np.full(n, 0.15), tau=0.8, n_sims=6000)
    row_sums = sim["position_probs"].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9), "Each driver's position_probs must sum to 1"


def test_position_probs_col_sums_to_one() -> None:
    n = 12
    sim = simulate_race(_base_scores(n), np.full(n, 0.15), tau=0.8, n_sims=6000)
    col_sums = sim["position_probs"].sum(axis=0)
    assert np.allclose(col_sums, 1.0, atol=1e-9), "Each position column must sum to 1"


def test_p_win_sums_to_one() -> None:
    n = 10
    sim = simulate_race(_base_scores(n), np.zeros(n), tau=0.7, n_sims=8000)
    assert abs(sim["p_win"].sum() - 1.0) < 1e-9


def test_higher_score_means_higher_pwin() -> None:
    """With equal DNF risk, p_win must be (weakly) monotonically decreasing in rank."""
    n = 10
    sim = simulate_race(_base_scores(n), np.full(n, 0.05), tau=0.5, n_sims=20000)
    pw = sim["p_win"]
    # Allow tiny Monte-Carlo noise
    assert np.all(np.diff(pw) <= 0.01), f"p_win not monotone in score: {pw}"
    assert pw[0] == pw.max(), "Top-scored driver must have the highest p_win"


def test_dnf_reduces_win_probability() -> None:
    n = 8
    scores = _base_scores(n)
    p_low = simulate_race(scores, np.zeros(n), tau=0.6, n_sims=8000)["p_win"][0]
    p_dnf = np.zeros(n)
    p_dnf[0] = 0.9  # cripple the favourite
    p_high = simulate_race(scores, p_dnf, tau=0.6, n_sims=8000)["p_win"][0]
    assert p_high < p_low, "A high DNF probability must lower a driver's win probability"


def test_calibrate_tau_returns_positive() -> None:
    races = [
        {"scores": _base_scores(10), "p_dnf": np.full(10, 0.1), "winner_idx": 0},
        {"scores": _base_scores(10), "p_dnf": np.full(10, 0.1), "winner_idx": 1},
    ]
    tau = calibrate_tau(races, n_sims=800)
    assert tau > 0.0
