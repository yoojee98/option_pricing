"""
tests/test_monte_carlo.py

Unit tests for models/monte_carlo.py.

Covers:
  - MC price within 95% CI of the BS closed-form price
  - Antithetic variance reduction (SE strictly smaller at identical N)
  - Put-Call Parity using a single shared Z draw (should be near-exact)
  - CRN bump-and-reprice Greeks vs BS closed-form
  - Convergence: SE shrinks with more paths (approximately 1/sqrt(N))
  - Edge cases: T=0 intrinsic, invalid inputs
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.bs_model import bs_delta, bs_gamma, bs_price, bs_rho, bs_vega
from models.monte_carlo import (
    compare_antithetic,
    mc_greeks,
    mc_price,
    _price_with_Z,
)


# ── Accuracy vs Black-Scholes ─────────────────────────────────────────────────

BASE_PARAMS = dict(S=100.0, K=100.0, T=0.5, r=0.05, q=0.013, sigma=0.22)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mc_within_ci_of_bs(option_type):
    # With 200k paths the 95% CI half-width is ~0.03-0.04 for ATM — BS should
    # sit inside that interval almost always (by construction, for a fixed
    # seed, if the estimator is correctly specified).
    res = mc_price(**BASE_PARAMS, option_type=option_type,
                   n_paths=200_000, antithetic=True, seed=42)
    bs = bs_price(**BASE_PARAMS, option_type=option_type)
    lo, hi = res.ci95
    assert lo <= bs <= hi, f"BS {bs:.4f} outside CI [{lo:.4f}, {hi:.4f}]"


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mc_price_close_to_bs_high_N(option_type):
    # At 1M paths the error should be ~0.01 for ATM
    res = mc_price(**BASE_PARAMS, option_type=option_type,
                   n_paths=1_000_000, antithetic=True, seed=42)
    bs = bs_price(**BASE_PARAMS, option_type=option_type)
    assert res.price == pytest.approx(bs, abs=0.03)


# ── Antithetic variance reduction ─────────────────────────────────────────────

def test_antithetic_reduces_se():
    cmp = compare_antithetic(**BASE_PARAMS, option_type="call",
                             n_paths=100_000, seed=42)
    assert cmp["anti"].se < cmp["plain"].se
    assert cmp["se_ratio"] < 1.0


# ── Put-Call Parity under common random numbers ───────────────────────────────

def test_parity_with_common_Z():
    # Use the same Z draw for call and put — parity then holds to machine precision
    # (modulo the identical disc factor). This is a stronger test than the CI-based
    # one: it checks the estimator's *algebraic* consistency, not its statistical one.
    p = BASE_PARAMS
    rng = np.random.default_rng(123)
    Z = rng.standard_normal(50_000)

    call = _price_with_Z(p["S"], p["K"], p["T"], p["r"], p["q"], p["sigma"], "call", Z)
    put  = _price_with_Z(p["S"], p["K"], p["T"], p["r"], p["q"], p["sigma"], "put",  Z)

    lhs = call - put
    rhs = p["S"] * np.exp(-p["q"] * p["T"]) - p["K"] * np.exp(-p["r"] * p["T"])
    # The identity S_T * e^-rT averaged = S * e^-qT holds only in expectation,
    # so there is residual MC noise here. At 50k paths it should be ~0.01.
    assert lhs == pytest.approx(rhs, abs=0.05)


# ── CRN Greeks vs BS ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_crn_delta_matches_bs(option_type):
    g = mc_greeks(**BASE_PARAMS, option_type=option_type,
                  n_paths=200_000, seed=42)
    bs = bs_delta(**BASE_PARAMS, option_type=option_type)
    assert g["delta"] == pytest.approx(bs, abs=3e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_crn_vega_matches_bs(option_type):
    g = mc_greeks(**BASE_PARAMS, option_type=option_type,
                  n_paths=200_000, seed=42)
    bs = bs_vega(BASE_PARAMS["S"], BASE_PARAMS["K"], BASE_PARAMS["T"],
                 BASE_PARAMS["r"], BASE_PARAMS["sigma"], BASE_PARAMS["q"])
    assert g["vega"] == pytest.approx(bs, rel=2e-2)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_crn_rho_matches_bs(option_type):
    g = mc_greeks(**BASE_PARAMS, option_type=option_type,
                  n_paths=200_000, seed=42)
    bs = bs_rho(**BASE_PARAMS, option_type=option_type)
    assert g["rho"] == pytest.approx(bs, rel=2e-2)


def test_crn_gamma_positive():
    # Gamma for a vanilla ATM option is small but positive. At finite N the
    # second-difference is noisy, so just sign-check + order of magnitude.
    g = mc_greeks(**BASE_PARAMS, option_type="call", n_paths=500_000, seed=42)
    bs = bs_gamma(BASE_PARAMS["S"], BASE_PARAMS["K"], BASE_PARAMS["T"],
                  BASE_PARAMS["r"], BASE_PARAMS["sigma"], BASE_PARAMS["q"])
    assert g["gamma"] > 0
    assert g["gamma"] == pytest.approx(bs, abs=3e-3)


# ── Convergence rate ──────────────────────────────────────────────────────────

def test_se_shrinks_with_n():
    # SE scales as 1/sqrt(N): doubling N should cut SE by ~sqrt(2) ≈ 0.71
    r1 = mc_price(**BASE_PARAMS, option_type="call", n_paths=50_000,
                  antithetic=False, seed=42)
    r2 = mc_price(**BASE_PARAMS, option_type="call", n_paths=200_000,
                  antithetic=False, seed=42)
    # 4x paths → SE should drop by ~2x
    assert r2.se < r1.se * 0.6


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_intrinsic_at_expiry():
    r = mc_price(110.0, 100.0, 0.0, 0.05, 0.2, 0.0, "call", n_paths=1000, seed=0)
    assert r.price == pytest.approx(10.0)
    assert r.se == 0.0


def test_invalid_option_type_raises():
    with pytest.raises(ValueError, match="option_type"):
        mc_price(100, 100, 0.5, 0.05, 0.2, 0.0, "banana", n_paths=1000)


def test_invalid_sigma_raises():
    with pytest.raises(ValueError, match="sigma"):
        mc_price(100, 100, 0.5, 0.05, 0.0, 0.0, "call", n_paths=1000)


def test_antithetic_requires_even_n():
    with pytest.raises(ValueError, match="even"):
        mc_price(100, 100, 0.5, 0.05, 0.2, 0.0, "call", n_paths=999,
                 antithetic=True)


def test_reproducible_with_seed():
    # Same seed → identical results, different seed → different results
    r1 = mc_price(**BASE_PARAMS, option_type="call", n_paths=10_000, seed=7)
    r2 = mc_price(**BASE_PARAMS, option_type="call", n_paths=10_000, seed=7)
    r3 = mc_price(**BASE_PARAMS, option_type="call", n_paths=10_000, seed=8)
    assert r1.price == r2.price
    assert r1.price != r3.price
