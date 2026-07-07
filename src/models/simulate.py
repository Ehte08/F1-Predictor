"""
Plackett-Luce Monte-Carlo race simulation.

Given per-driver ranker scores and P(DNF), simulate N races. Each sim:
  1. sample DNFs via Bernoulli(p_dnf);
  2. rank surviving drivers by Plackett-Luce sampling — the Gumbel-max trick:
     rank by score/tau + Gumbel(0,1) noise (equivalent to sampling without
     replacement proportional to exp(score/tau));
  3. DNF drivers fill the back positions (ordered worst score last).

tau controls how deterministic the order is (small tau -> chalky, large tau ->
noisy). Calibrate it on validation races so predicted P(win) is roughly calibrated.
"""
from __future__ import annotations

import numpy as np

DEFAULT_TAU = 1.0
N_SIMS = 10_000


def _standardize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    sd = scores.std()
    if sd < 1e-9:
        return scores - scores.mean()
    return (scores - scores.mean()) / sd


def simulate_race(
    scores,
    p_dnf,
    tau: float = DEFAULT_TAU,
    n_sims: int = N_SIMS,
    seed: int | None = 0,
) -> dict:
    """
    Monte-Carlo a single race.

    Returns a dict with arrays aligned to the input order:
      p_win, p_podium, p_points (top-10), p_dnf, and position_probs
      (shape [n_drivers, n_drivers]; row d, col p = P(driver d finishes position p+1)).
    """
    rng = np.random.default_rng(seed)
    z = _standardize(scores)
    p_dnf = np.clip(np.asarray(p_dnf, dtype=float), 0.0, 1.0)
    n = len(z)
    tau = max(float(tau), 1e-6)

    utilities = z / tau  # higher => more likely to finish ahead

    position_counts = np.zeros((n, n), dtype=float)
    dnf_counts = np.zeros(n, dtype=float)

    # Vectorized over sims: draw DNF mask and Gumbel noise for all sims at once.
    dnf_draw = rng.random((n_sims, n)) < p_dnf[None, :]          # [S, n]
    gumbel = rng.gumbel(size=(n_sims, n))                         # [S, n]
    noisy = utilities[None, :] + gumbel                          # [S, n]

    # Survivors ranked by noisy utility (desc); DNFs pushed to the back but still
    # ordered among themselves by noisy utility so "least bad" DNF is ahead.
    order_key = np.where(dnf_draw, noisy - 1e6, noisy)           # big penalty for DNF
    # argsort descending: position 0 = best
    finishing_order = np.argsort(-order_key, axis=1)             # [S, n] driver idx per position

    # position_of[s, driver] = finishing position (0-based)
    position_of = np.empty((n_sims, n), dtype=np.int64)
    rows = np.arange(n_sims)[:, None]
    position_of[rows, finishing_order] = np.arange(n)[None, :]

    for d in range(n):
        pos = position_of[:, d]
        np.add.at(position_counts[d], pos, 1.0)
    dnf_counts = dnf_draw.sum(axis=0).astype(float)

    position_probs = position_counts / n_sims
    # Aggregates are exact in count arithmetic; clip guards float-summation overshoot.
    p_win = np.clip(position_probs[:, 0], 0.0, 1.0)
    p_podium = np.clip(position_probs[:, : min(3, n)].sum(axis=1), 0.0, 1.0)
    p_points = np.clip(position_probs[:, : min(10, n)].sum(axis=1), 0.0, 1.0)
    p_dnf_emp = np.clip(dnf_counts / n_sims, 0.0, 1.0)

    return {
        "p_win": p_win,
        "p_podium": p_podium,
        "p_points": p_points,
        "p_dnf": p_dnf_emp,
        "position_probs": position_probs,
    }


def _winner_loglik(races: list[dict], tau: float, n_sims: int, seed: int) -> float:
    """Sum of log P(actual winner) across validation races for a given tau."""
    total = 0.0
    eps = 1.0 / (n_sims * 10.0)
    for i, r in enumerate(races):
        sim = simulate_race(r["scores"], r["p_dnf"], tau=tau, n_sims=n_sims, seed=seed + i)
        pw = np.clip(sim["p_win"], eps, 1.0)
        total += float(np.log(pw[r["winner_idx"]]))
    return total


def calibrate_tau(
    races: list[dict],
    grid: np.ndarray | None = None,
    n_sims: int = 2000,
    seed: int = 0,
) -> float:
    """
    Grid-search tau maximizing winner log-likelihood on held-out races.

    Each race dict needs: scores (array), p_dnf (array), winner_idx (int, position of
    the actual winner within the score array). Returns the best tau; defaults to 1.0
    when no valid races are supplied.
    """
    races = [r for r in races if r.get("winner_idx") is not None and len(r["scores"]) > 1]
    if not races:
        return DEFAULT_TAU
    if grid is None:
        grid = np.concatenate([np.linspace(0.1, 1.0, 10), np.linspace(1.2, 4.0, 8)])

    best_tau, best_ll = DEFAULT_TAU, -np.inf
    for tau in grid:
        ll = _winner_loglik(races, float(tau), n_sims=n_sims, seed=seed)
        if ll > best_ll:
            best_ll, best_tau = ll, float(tau)
    return best_tau
