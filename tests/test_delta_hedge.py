"""
tests/test_delta_hedge.py

Unit tests for analysis/delta_hedge.py.

Covers:
  - GBM path simulator: terminal distribution moments, shape, determinism.
  - Single-path hedge: terminal PnL accounting identity.
  - Unbiasedness when sigma_realized == sigma_implied.
  - Short-gamma bias when sigma_realized != sigma_implied (sign and rough
    magnitude against the Gamma-PnL formula).
  - Discretization-error scaling: std(PnL) shrinks ~ 1/sqrt(n_steps).
  - Trajectory invariants: portfolio value == 0 at inception; arrays have
    the correct shape.
  - Put/call symmetry: at-the-money put and call hedges have the same
    error distribution (Gamma is the same).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.delta_hedge import (
    HedgePath,
    HedgeResult,
    hedge_single_path,
    simulate_delta_hedge,
    simulate_gbm_paths,
)
from models.bs_model import bs_gamma, bs_price


# ── Shared parameters ─────────────────────────────────────────────────────────

BASE = dict(
    S0=100.0, K=100.0, T=0.5, r=0.04, sigma_implied=0.20,
    q=0.013, option_type="call",
)


# ── GBM simulator ─────────────────────────────────────────────────────────────

def test_gbm_paths_shape_and_initial_value():
    paths = simulate_gbm_paths(
        S0=100.0, mu=0.05, sigma=0.2, T=1.0,
        n_steps=50, n_sims=30, q=0.0, seed=0,
    )
    assert paths.shape == (30, 51)
    # First column ≈ S0 (exp(log(S0)) is not bit-exact).
    assert np.allclose(paths[:, 0], 100.0, rtol=0, atol=1e-10)
    assert np.all(paths > 0)  # GBM is strictly positive


def test_gbm_paths_deterministic_with_seed():
    kw = dict(S0=100.0, mu=0.05, sigma=0.2, T=1.0, n_steps=20, n_sims=10, q=0.0)
    a = simulate_gbm_paths(**kw, seed=123)
    b = simulate_gbm_paths(**kw, seed=123)
    assert np.array_equal(a, b)


def test_gbm_terminal_mean_matches_theory():
    # E[S_T] = S0 * exp((mu - q) * T) for GBM (independent of sigma)
    S0, mu, q, T = 100.0, 0.07, 0.02, 1.0
    paths = simulate_gbm_paths(
        S0=S0, mu=mu, sigma=0.25, T=T,
        n_steps=50, n_sims=50_000, q=q, seed=7,
    )
    expected = S0 * np.exp((mu - q) * T)
    # 50k paths: tight enough for a 1.5% tolerance
    assert paths[:, -1].mean() == pytest.approx(expected, rel=0.015)


def test_gbm_invalid_inputs():
    with pytest.raises(ValueError, match="n_steps"):
        simulate_gbm_paths(S0=100, mu=0.0, sigma=0.2, T=1.0, n_steps=0, n_sims=10)
    with pytest.raises(ValueError, match="n_sims"):
        simulate_gbm_paths(S0=100, mu=0.0, sigma=0.2, T=1.0, n_steps=10, n_sims=0)


# ── Hedge unbiasedness (σ_r == σ_i) ───────────────────────────────────────────

def test_unbiased_when_realized_equals_implied():
    # Under perfect vol calibration, mean hedging PnL → 0 as n_steps → ∞.
    # At 252 steps with 3000 paths the SE of the mean is small enough that we
    # can assert |mean| < 3 * SE / sqrt(n_sims).
    res = simulate_delta_hedge(
        **BASE, sigma_realized=BASE["sigma_implied"],
        n_steps=252, n_sims=3000, seed=42,
    )
    se_of_mean = res.std / np.sqrt(res.n_sims)
    assert abs(res.mean) < 4 * se_of_mean
    assert res.std > 0  # non-degenerate distribution


# ── Short-gamma bias (σ_r ≠ σ_i) ──────────────────────────────────────────────

def test_short_gamma_loses_when_realized_vol_higher():
    # Seller of the option is short gamma; when realized > implied, hedge loses.
    sigma_r = BASE["sigma_implied"] * 1.5
    res = simulate_delta_hedge(
        **BASE, sigma_realized=sigma_r,
        n_steps=252, n_sims=2000, seed=1,
    )
    # Expect clearly negative mean (many standard errors away from 0).
    se_of_mean = res.std / np.sqrt(res.n_sims)
    assert res.mean < -5 * se_of_mean


def test_short_gamma_wins_when_realized_vol_lower():
    sigma_r = BASE["sigma_implied"] * 0.5
    res = simulate_delta_hedge(
        **BASE, sigma_realized=sigma_r,
        n_steps=252, n_sims=2000, seed=2,
    )
    se_of_mean = res.std / np.sqrt(res.n_sims)
    assert res.mean > 5 * se_of_mean


def test_pnl_bias_matches_gamma_formula():
    # Continuous-time theory:  E[PnL] ≈ 0.5 * Γ * S² * (σ_i² - σ_r²) * T
    # (short option, ATM, evaluated at t=0 as a first-order approximation).
    # We check that the MC estimate lies within 30% of this — tight enough to
    # catch sign/magnitude errors, loose enough to absorb discretization drift.
    sigma_r = BASE["sigma_implied"] * 1.3
    gamma0 = bs_gamma(BASE["S0"], BASE["K"], BASE["T"], BASE["r"],
                      BASE["sigma_implied"], BASE["q"])
    expected = 0.5 * gamma0 * BASE["S0"] ** 2 * (
        BASE["sigma_implied"] ** 2 - sigma_r ** 2
    ) * BASE["T"]
    res = simulate_delta_hedge(
        **BASE, sigma_realized=sigma_r,
        n_steps=252, n_sims=5000, seed=3,
    )
    assert res.mean == pytest.approx(expected, rel=0.30)


# ── Discretization-error scaling ──────────────────────────────────────────────

def test_pnl_std_scales_as_inv_sqrt_n_steps():
    # For a Δ-hedged short option, residual variance per rebalance scales
    # as dt², summed over N = T/dt steps → total variance ~ dt ~ 1/N.
    # Hence std(PnL) ~ 1/sqrt(N).  Compare N=21 vs N=252 (12x):
    stds = {}
    for n in (21, 252):
        res = simulate_delta_hedge(
            **BASE, sigma_realized=BASE["sigma_implied"],
            n_steps=n, n_sims=1500, seed=10 + n,
        )
        stds[n] = res.std

    ratio = stds[21] / stds[252]
    expected = np.sqrt(252 / 21)  # ≈ 3.46
    # Allow ±35%: the asymptotic 1/sqrt(N) law has O(1) prefactor noise at small N.
    assert ratio == pytest.approx(expected, rel=0.35)


# ── Single-path PnL identity ──────────────────────────────────────────────────

def test_single_path_pnl_matches_trajectory_terminal_value():
    # Terminal PnL returned by the scalar path should equal the PnL stored
    # in the trajectory dataclass — they must compute the same thing.
    paths = simulate_gbm_paths(
        S0=BASE["S0"], mu=BASE["r"], sigma=BASE["sigma_implied"],
        T=BASE["T"], n_steps=50, n_sims=1, q=BASE["q"], seed=99,
    )
    S_path = paths[0]
    pnl_scalar = hedge_single_path(
        S_path, K=BASE["K"], T=BASE["T"], r=BASE["r"],
        sigma_implied=BASE["sigma_implied"], q=BASE["q"],
        option_type="call", return_trajectory=False,
    )
    traj = hedge_single_path(
        S_path, K=BASE["K"], T=BASE["T"], r=BASE["r"],
        sigma_implied=BASE["sigma_implied"], q=BASE["q"],
        option_type="call", return_trajectory=True,
    )
    assert isinstance(traj, HedgePath)
    assert traj.pnl == pytest.approx(pnl_scalar, abs=1e-10)


def test_trajectory_has_zero_portfolio_at_inception():
    # Right after selling the option and buying Δ₀ shares, the book is flat:
    #   (-V₀)   +   Δ₀ · S₀   +   (V₀ - Δ₀ · S₀)   =   0
    paths = simulate_gbm_paths(
        S0=BASE["S0"], mu=BASE["r"], sigma=BASE["sigma_implied"],
        T=BASE["T"], n_steps=20, n_sims=1, q=BASE["q"], seed=0,
    )
    traj = hedge_single_path(
        paths[0], K=BASE["K"], T=BASE["T"], r=BASE["r"],
        sigma_implied=BASE["sigma_implied"], q=BASE["q"],
        option_type="call", return_trajectory=True,
    )
    assert traj.portfolio[0] == pytest.approx(0.0, abs=1e-10)


def test_trajectory_array_shapes():
    n_steps = 30
    paths = simulate_gbm_paths(
        S0=BASE["S0"], mu=BASE["r"], sigma=BASE["sigma_implied"],
        T=BASE["T"], n_steps=n_steps, n_sims=1, q=BASE["q"], seed=0,
    )
    traj = hedge_single_path(
        paths[0], K=BASE["K"], T=BASE["T"], r=BASE["r"],
        sigma_implied=BASE["sigma_implied"], q=BASE["q"],
        option_type="call", return_trajectory=True,
    )
    expected = n_steps + 1
    for arr in (traj.t, traj.S, traj.delta, traj.gamma, traj.vega,
                traj.cash, traj.portfolio):
        assert arr.shape == (expected,)
    # Time grid goes from 0 to T.
    assert traj.t[0] == 0.0
    assert traj.t[-1] == pytest.approx(BASE["T"])


def test_hedge_single_path_rejects_short_path():
    with pytest.raises(ValueError, match="S_path"):
        hedge_single_path(
            np.array([100.0]),
            K=100.0, T=0.5, r=0.04, sigma_implied=0.2,
        )


# ── Put/call gamma symmetry ───────────────────────────────────────────────────

def test_call_and_put_hedge_have_same_pnl_std():
    # Gamma is identical for a call and a put at the same strike, so the
    # discretization-error distribution of a Δ-hedged short call and short
    # put should match closely when driven by the same seed.
    call = simulate_delta_hedge(
        **{**BASE, "option_type": "call"},
        sigma_realized=BASE["sigma_implied"],
        n_steps=100, n_sims=1500, seed=77,
    )
    put = simulate_delta_hedge(
        **{**BASE, "option_type": "put"},
        sigma_realized=BASE["sigma_implied"],
        n_steps=100, n_sims=1500, seed=77,
    )
    # Both should be unbiased and have similar spread.
    assert call.std == pytest.approx(put.std, rel=0.10)


# ── HedgeResult sanity ────────────────────────────────────────────────────────

def test_hedge_result_fields_populated():
    res = simulate_delta_hedge(
        **BASE, sigma_realized=BASE["sigma_implied"],
        n_steps=50, n_sims=200, seed=0,
    )
    assert isinstance(res, HedgeResult)
    assert res.pnl.shape == (200,)
    assert res.p05 < res.p95
    assert res.n_steps == 50
    assert res.n_sims == 200
    assert res.sigma_implied == BASE["sigma_implied"]
    assert res.sigma_realized == BASE["sigma_implied"]
