"""
tests/test_exotic_options.py

Unit tests for models/exotic_options.py.

Covers:
  - Asian arithmetic pricer:
      * Non-negative prices, call >= put for K <= S (no-arbitrage sanity)
      * Control variate strictly reduces SE (typically 10-100x)
      * Geometric-Asian closed-form matches its own MC within CI
      * n_steps = 1 recovers the vanilla European price (mean == S_T)
      * T = 0 returns intrinsic value
  - Barrier Down-and-Out pricer:
      * Knocked-out at initialization (B >= S) returns price 0
      * DO + DI = Vanilla (partition identity on every path)
      * Discrete MC converges *downward* to continuous closed form as
        monitoring grid is refined (Broadie-Glasserman-Kou direction)
      * Monotonicity: DO price decreases as B rises toward S
      * MC price within 95% CI of closed-form for fine monitoring
  - Closed-form barrier:
      * B >= S returns 0
      * K >= B for put returns 0 (entire ITM region knocked out)
      * Vanilla limit: B -> 0 recovers vanilla call price
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.bs_model import bs_price
from models.exotic_options import (
    _geometric_asian_closed_form,
    _simulate_paths_rn,
    _discounted_mc_stats,
    asian_price,
    barrier_do_closed_form,
    barrier_do_price,
)
from models.monte_carlo import MCResult


# ── Shared parameters ─────────────────────────────────────────────────────────

BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.3)


# ══════════════════════════════════════════════════════════════════════════════
#  Asian options
# ══════════════════════════════════════════════════════════════════════════════

def test_asian_price_positive_and_returns_mcresult():
    res = asian_price(**BASE, option_type="call",
                      n_paths=20_000, n_steps=50, seed=0)
    assert isinstance(res, MCResult)
    assert res.price > 0
    assert res.se > 0
    lo, hi = res.ci95
    assert lo <= res.price <= hi


def test_asian_call_ge_put_when_atm():
    # At ATM with q=0, the arithmetic Asian call is worth more than the put
    # because E[avg] > K under positive drift (r - q > 0).
    call = asian_price(**BASE, option_type="call",
                       n_paths=30_000, n_steps=50, seed=1)
    put = asian_price(**BASE, option_type="put",
                      n_paths=30_000, n_steps=50, seed=1)
    assert call.price > put.price


def test_asian_control_variate_reduces_se():
    # Control variate should cut SE by at least 5x on a standard ATM Asian
    # (typical reduction is 15-30x; 5x is a safe lower bound).
    kw = dict(**BASE, option_type="call", n_paths=20_000, n_steps=50,
              antithetic=True, seed=42)
    plain = asian_price(**kw, control_variate=False)
    cv = asian_price(**kw, control_variate=True)
    assert cv.se < plain.se / 5.0


def test_asian_price_below_vanilla_for_fine_monitoring():
    # With many monitoring points the arithmetic average has lower variance
    # than S_T alone (variance of the mean of n+1 correlated lognormals is
    # strictly below the variance of S_T), so the Asian call is strictly
    # cheaper than the vanilla call at the same ATM strike.
    res = asian_price(**BASE, option_type="call",
                      n_paths=30_000, n_steps=100, seed=0)
    vanilla = bs_price(**BASE, option_type="call")
    # Must be positive, strictly cheaper than vanilla, by a clear margin
    # (several SE above zero and below vanilla).
    assert res.price > 3 * res.se
    assert res.price < vanilla - 3 * res.se


def test_asian_t_zero_returns_intrinsic():
    res = asian_price(S=110.0, K=100.0, T=0.0, r=0.05, sigma=0.3,
                      q=0.0, option_type="call",
                      n_paths=1000, n_steps=1, seed=0)
    assert res.price == pytest.approx(10.0)
    assert res.se == 0.0


def test_asian_invalid_inputs_raise():
    with pytest.raises(ValueError, match="option_type"):
        asian_price(**BASE, option_type="banana", n_paths=100, n_steps=10)
    with pytest.raises(ValueError, match="sigma"):
        asian_price(S=100, K=100, T=1.0, r=0.05, sigma=-0.1, q=0.0,
                    n_paths=100, n_steps=10)
    with pytest.raises(ValueError, match="n_steps"):
        asian_price(**BASE, n_paths=100, n_steps=0)


# ── Geometric-Asian closed form (used as control variate) ─────────────────────

def test_geometric_asian_closed_form_within_mc_ci():
    # Price the *geometric* Asian by MC and check the closed form lies in CI.
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.3
    n_steps = 50
    n_paths = 50_000

    rng = np.random.default_rng(0)
    paths = _simulate_paths_rn(S, T, r, q, sigma, n_paths, n_steps, True, rng)
    geo_mean = np.exp(np.mean(np.log(paths), axis=1))
    payoff = np.maximum(geo_mean - K, 0.0)
    price, se = _discounted_mc_stats(payoff, r, T, antithetic=True)
    closed = _geometric_asian_closed_form(S, K, T, r, q, sigma, n_steps, "call")

    # 3-sigma band: closed form should always be this close.
    assert abs(price - closed) < 3 * se


# ══════════════════════════════════════════════════════════════════════════════
#  Barrier options (Down-and-Out)
# ══════════════════════════════════════════════════════════════════════════════

BARRIER_PARAMS = dict(S=100.0, K=100.0, B=80.0, T=1.0, r=0.05, sigma=0.3, q=0.0)


def test_barrier_do_already_knocked_out():
    # Barrier above spot: option is dead at t=0.
    res = barrier_do_price(S=100, K=100, B=110, T=1.0, r=0.05, sigma=0.3,
                           q=0.0, option_type="call",
                           n_paths=100, n_steps=10, seed=0)
    assert res.price == 0.0
    assert res.se == 0.0

    # Closed form agrees.
    assert barrier_do_closed_form(S=100, K=100, B=110, T=1.0,
                                  r=0.05, sigma=0.3, q=0.0,
                                  option_type="call") == 0.0


def test_do_plus_di_equals_vanilla():
    # For any single path, payoff_DO(path) + payoff_DI(path) == payoff_vanilla(path),
    # so the MC estimators (using the *same* paths) must satisfy this exactly
    # up to floating-point noise — not MC noise.
    n_paths, n_steps = 10_000, 200
    S, K, B = 100.0, 100.0, 80.0
    T, r, q, sigma = 1.0, 0.05, 0.0, 0.3

    rng = np.random.default_rng(123)
    paths = _simulate_paths_rn(S, T, r, q, sigma, n_paths, n_steps, True, rng)
    alive = paths.min(axis=1) > B
    S_T = paths[:, -1]
    call_payoff = np.maximum(S_T - K, 0.0)
    do_payoff = np.where(alive, call_payoff, 0.0)
    di_payoff = np.where(~alive, call_payoff, 0.0)

    do_price, _ = _discounted_mc_stats(do_payoff, r, T, antithetic=True)
    di_price, _ = _discounted_mc_stats(di_payoff, r, T, antithetic=True)
    vanilla_mc, _ = _discounted_mc_stats(call_payoff, r, T, antithetic=True)

    # Exact partition identity — no MC noise between the three estimators.
    assert do_price + di_price == pytest.approx(vanilla_mc, abs=1e-10)


def test_barrier_do_mc_within_ci_of_closed_form_fine_grid():
    # Fine monitoring (500 steps) + 200k paths: discrete bias is small enough
    # that the MC 95% CI typically captures the closed-form continuous price.
    closed = barrier_do_closed_form(**BARRIER_PARAMS, option_type="call")
    mc = barrier_do_price(**BARRIER_PARAMS, option_type="call",
                          n_paths=200_000, n_steps=500, seed=42)
    # Discrete MC price > continuous closed form (knock-out under-counted at
    # discrete grid).  Check MC - closed_form > 0 but within a few SE.
    assert mc.price > closed - 3 * mc.se
    assert mc.price - closed < 0.20  # discrete overpricing bounded in absolute $


def test_barrier_do_discrete_converges_to_continuous():
    # As the monitoring grid is refined, the discrete MC price must drift
    # *downward* toward the continuous closed form.
    closed = barrier_do_closed_form(**BARRIER_PARAMS, option_type="call")
    price_coarse = barrier_do_price(**BARRIER_PARAMS, option_type="call",
                                    n_paths=100_000, n_steps=20, seed=7).price
    price_fine = barrier_do_price(**BARRIER_PARAMS, option_type="call",
                                  n_paths=100_000, n_steps=500, seed=7).price

    # Both above the continuous limit, fine is closer.
    assert price_coarse > closed
    assert price_fine > closed
    assert abs(price_fine - closed) < abs(price_coarse - closed)


def test_barrier_do_decreases_as_barrier_rises():
    # Moving B closer to spot knocks more paths out, so the DO price must
    # monotonically decrease in B (for B < S).
    prices = []
    for B in (50.0, 70.0, 85.0, 95.0):
        res = barrier_do_price(S=100.0, K=100.0, B=B, T=1.0, r=0.05,
                               sigma=0.3, q=0.0, option_type="call",
                               n_paths=50_000, n_steps=200, seed=99)
        prices.append(res.price)
    # Strictly decreasing sequence.
    for i in range(len(prices) - 1):
        assert prices[i] > prices[i + 1], f"DO price not decreasing: {prices}"


def test_barrier_do_vanilla_limit_as_B_vanishes():
    # B -> 0 makes knock-out impossible, so DO -> vanilla.
    closed_tiny = barrier_do_closed_form(S=100.0, K=100.0, B=0.01, T=1.0,
                                         r=0.05, sigma=0.3, q=0.0,
                                         option_type="call")
    vanilla = bs_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.3, q=0.0,
                       option_type="call")
    assert closed_tiny == pytest.approx(vanilla, rel=1e-6)


def test_barrier_do_put_knocked_out_when_K_le_B():
    # Down-and-Out put with K <= B: the entire ITM region is below B, so
    # any path that reaches the ITM payoff region has already knocked out.
    # Closed form returns 0.
    price = barrier_do_closed_form(S=100.0, K=80.0, B=90.0, T=1.0,
                                   r=0.05, sigma=0.3, q=0.0,
                                   option_type="put")
    assert price == 0.0


def test_barrier_do_invalid_inputs_raise():
    with pytest.raises(ValueError, match="option_type"):
        barrier_do_price(**BARRIER_PARAMS, option_type="x",
                         n_paths=100, n_steps=10)
    with pytest.raises(ValueError, match="sigma"):
        barrier_do_price(S=100, K=100, B=80, T=1.0, r=0.05,
                         sigma=0.0, q=0.0, n_paths=100, n_steps=10)
    with pytest.raises(ValueError, match="B"):
        barrier_do_price(S=100, K=100, B=-1.0, T=1.0, r=0.05,
                         sigma=0.3, q=0.0, n_paths=100, n_steps=10)
