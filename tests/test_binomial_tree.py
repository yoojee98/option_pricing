"""
tests/test_binomial_tree.py

Unit tests for models/binomial_tree.py.

Covers:
  - Convergence to Black-Scholes as N → ∞ (European)
  - Put-Call Parity on the tree
  - American vs European: early-exercise premium sign
  - Tree Greeks vs BS closed-form Greeks
  - Edge cases: T=0 intrinsic, invalid inputs
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.binomial_tree import (
    crr_greeks,
    crr_price,
    early_exercise_premium,
)
from models.bs_model import bs_delta, bs_gamma, bs_price, bs_rho, bs_vega


# ── Convergence to Black-Scholes ──────────────────────────────────────────────

CONV_PARAMS = dict(S=100.0, K=100.0, T=0.5, r=0.05, q=0.013, sigma=0.22)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_converges_to_bs(option_type):
    p = CONV_PARAMS
    bs = bs_price(**p, option_type=option_type)
    crr = crr_price(**p, N=2000, option_type=option_type, american=False)
    assert crr == pytest.approx(bs, abs=2e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_convergence_rate(option_type):
    # Error should shrink roughly linearly in 1/N: err(2N) < err(N)
    p = CONV_PARAMS
    bs = bs_price(**p, option_type=option_type)
    err_200  = abs(crr_price(**p, N=200,  option_type=option_type) - bs)
    err_1000 = abs(crr_price(**p, N=1000, option_type=option_type) - bs)
    assert err_1000 < err_200


# ── Put-Call Parity on the tree ───────────────────────────────────────────────

@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("T", [0.1, 0.5, 1.5])
def test_european_parity_on_tree(K, T):
    S, r, q, sigma, N = 100.0, 0.05, 0.013, 0.22, 1000
    call = crr_price(S, K, T, r, sigma, q, N, "call", american=False)
    put  = crr_price(S, K, T, r, sigma, q, N, "put",  american=False)
    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    # Parity holds exactly in the continuous limit; at finite N there is a
    # small discretization residual, same order as the BS→CRR error.
    assert lhs == pytest.approx(rhs, abs=5e-3)


# ── American vs European ──────────────────────────────────────────────────────

def test_american_call_no_dividend_equals_european():
    # Merton: non-dividend American call == European call (never optimal to exercise early)
    S, K, T, r, q, sigma, N = 100.0, 100.0, 0.5, 0.05, 0.0, 0.22, 500
    eu = crr_price(S, K, T, r, sigma, q, N, "call", american=False)
    am = crr_price(S, K, T, r, sigma, q, N, "call", american=True)
    assert am == pytest.approx(eu, abs=1e-10)


def test_american_put_premium_positive():
    # American put on a non-dividend stock: early exercise can be optimal
    eep = early_exercise_premium(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.22,
                                 q=0.0, N=500, option_type="put")
    assert eep > 0.01


def test_american_put_deep_itm_exceeds_intrinsic():
    # Deep ITM American put must be >= intrinsic K - S
    S, K = 70.0, 100.0
    am = crr_price(S, K, T=1.0, r=0.05, sigma=0.22, q=0.013, N=500,
                   option_type="put", american=True)
    assert am >= (K - S) - 1e-10


# ── Tree Greeks vs BS closed-form ─────────────────────────────────────────────

GREEK_PARAMS = dict(S=100.0, K=100.0, T=0.5, r=0.05, q=0.013, sigma=0.22)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_tree_delta_matches_bs(option_type):
    p = GREEK_PARAMS
    g = crr_greeks(**p, N=2000, option_type=option_type, american=False)
    bs = bs_delta(**p, option_type=option_type)
    assert g["delta"] == pytest.approx(bs, abs=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_tree_gamma_matches_bs(option_type):
    p = GREEK_PARAMS
    g = crr_greeks(**p, N=2000, option_type=option_type, american=False)
    bs = bs_gamma(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    assert g["gamma"] == pytest.approx(bs, abs=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_tree_vega_matches_bs(option_type):
    p = GREEK_PARAMS
    g = crr_greeks(**p, N=1000, option_type=option_type, american=False)
    bs = bs_vega(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    assert g["vega"] == pytest.approx(bs, rel=1e-2)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_tree_rho_matches_bs(option_type):
    p = GREEK_PARAMS
    g = crr_greeks(**p, N=1000, option_type=option_type, american=False)
    bs = bs_rho(**p, option_type=option_type)
    assert g["rho"] == pytest.approx(bs, rel=1e-2)


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_intrinsic_at_expiry():
    assert crr_price(110, 100, 0, 0.05, 0.2, 0.0, N=100, option_type="call") == pytest.approx(10.0)
    assert crr_price(90,  100, 0, 0.05, 0.2, 0.0, N=100, option_type="call") == pytest.approx(0.0)
    assert crr_price(90,  100, 0, 0.05, 0.2, 0.0, N=100, option_type="put")  == pytest.approx(10.0)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError, match="option_type"):
        crr_price(100, 100, 0.5, 0.05, 0.2, 0.0, N=50, option_type="banana")


def test_invalid_sigma_raises():
    with pytest.raises(ValueError, match="sigma"):
        crr_price(100, 100, 0.5, 0.05, 0.0, 0.0, N=50, option_type="call")


def test_invalid_N_raises():
    with pytest.raises(ValueError, match="N"):
        crr_price(100, 100, 0.5, 0.05, 0.2, 0.0, N=0, option_type="call")
