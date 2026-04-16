"""
tests/test_bs_model.py

Unit tests for models/bs_model.py.

Covers:
  - Hull textbook reference values (price + delta)
  - Put-Call Parity identity across a grid of strikes and maturities
  - Greeks validation against central finite differences
  - Edge cases: T=0 intrinsic value, invalid inputs
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.bs_model import (
    bs_delta,
    bs_gamma,
    bs_greeks,
    bs_price,
    bs_rho,
    bs_theta,
    bs_vega,
)


# ── Hull reference values ─────────────────────────────────────────────────────
# Hull, "Options, Futures, and Other Derivatives" 10e, Ch. 15 example:
# S=49, K=50, T=0.3846 (20 weeks), r=0.05, q=0, sigma=0.20
# Expected: Call ≈ 2.40, Delta ≈ 0.522

HULL_PARAMS = dict(S=49.0, K=50.0, T=0.3846, r=0.05, q=0.0, sigma=0.20)


def test_hull_call_price():
    price = bs_price(**HULL_PARAMS, option_type="call")
    assert price == pytest.approx(2.40, abs=0.01)


def test_hull_call_delta():
    delta = bs_delta(**HULL_PARAMS, option_type="call")
    assert delta == pytest.approx(0.522, abs=0.005)


# ── Put-Call Parity identity ──────────────────────────────────────────────────

@pytest.mark.parametrize("S", [50.0, 100.0, 500.0])
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("T", [0.05, 0.5, 2.0])
@pytest.mark.parametrize("sigma", [0.10, 0.25, 0.60])
def test_put_call_parity(S, K, T, sigma):
    r, q = 0.05, 0.013
    call = bs_price(S, K, T, r, sigma, q, "call")
    put = bs_price(S, K, T, r, sigma, q, "put")
    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-10)


# ── Greeks vs finite differences ──────────────────────────────────────────────

FD_PARAMS = dict(S=100.0, K=100.0, T=0.5, r=0.04, q=0.013, sigma=0.22)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_finite_difference(option_type):
    p = FD_PARAMS
    h = 1e-4 * p["S"]
    up = bs_price(p["S"] + h, p["K"], p["T"], p["r"], p["sigma"], p["q"], option_type)
    dn = bs_price(p["S"] - h, p["K"], p["T"], p["r"], p["sigma"], p["q"], option_type)
    fd = (up - dn) / (2 * h)
    analytic = bs_delta(**p, option_type=option_type)
    assert analytic == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_gamma_finite_difference(option_type):
    p = FD_PARAMS
    h = 1e-3 * p["S"]
    up = bs_price(p["S"] + h, p["K"], p["T"], p["r"], p["sigma"], p["q"], option_type)
    mid = bs_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"], option_type)
    dn = bs_price(p["S"] - h, p["K"], p["T"], p["r"], p["sigma"], p["q"], option_type)
    fd = (up - 2 * mid + dn) / (h ** 2)
    analytic = bs_gamma(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    assert analytic == pytest.approx(fd, abs=1e-4)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vega_finite_difference(option_type):
    p = FD_PARAMS
    h = 1e-5
    up = bs_price(p["S"], p["K"], p["T"], p["r"], p["sigma"] + h, p["q"], option_type)
    dn = bs_price(p["S"], p["K"], p["T"], p["r"], p["sigma"] - h, p["q"], option_type)
    fd = (up - dn) / (2 * h)
    analytic = bs_vega(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    assert analytic == pytest.approx(fd, rel=1e-4)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_finite_difference(option_type):
    # bs_theta returns ∂V/∂t = -∂V/∂T (market convention, negative for long options)
    p = FD_PARAMS
    h = 1e-5
    up = bs_price(p["S"], p["K"], p["T"] + h, p["r"], p["sigma"], p["q"], option_type)
    dn = bs_price(p["S"], p["K"], p["T"] - h, p["r"], p["sigma"], p["q"], option_type)
    dV_dT = (up - dn) / (2 * h)
    analytic = bs_theta(**p, option_type=option_type)
    assert analytic == pytest.approx(-dV_dT, rel=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_rho_finite_difference(option_type):
    p = FD_PARAMS
    h = 1e-6
    up = bs_price(p["S"], p["K"], p["T"], p["r"] + h, p["sigma"], p["q"], option_type)
    dn = bs_price(p["S"], p["K"], p["T"], p["r"] - h, p["sigma"], p["q"], option_type)
    fd = (up - dn) / (2 * h)
    analytic = bs_rho(**p, option_type=option_type)
    assert analytic == pytest.approx(fd, rel=1e-4)


# ── Gamma / Vega symmetry (same for calls and puts) ───────────────────────────

def test_gamma_identical_call_put():
    p = FD_PARAMS
    call_up = bs_price(p["S"] + 1, p["K"], p["T"], p["r"], p["sigma"], p["q"], "call")
    call_md = bs_price(p["S"],     p["K"], p["T"], p["r"], p["sigma"], p["q"], "call")
    call_dn = bs_price(p["S"] - 1, p["K"], p["T"], p["r"], p["sigma"], p["q"], "call")
    put_up  = bs_price(p["S"] + 1, p["K"], p["T"], p["r"], p["sigma"], p["q"], "put")
    put_md  = bs_price(p["S"],     p["K"], p["T"], p["r"], p["sigma"], p["q"], "put")
    put_dn  = bs_price(p["S"] - 1, p["K"], p["T"], p["r"], p["sigma"], p["q"], "put")
    call_gamma = call_up - 2 * call_md + call_dn
    put_gamma  = put_up  - 2 * put_md  + put_dn
    assert call_gamma == pytest.approx(put_gamma, abs=1e-8)


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_intrinsic_at_expiry():
    # T=0 should return intrinsic value
    assert bs_price(110, 100, 0, 0.05, 0.2, 0.0, "call") == pytest.approx(10.0)
    assert bs_price(90, 100, 0, 0.05, 0.2, 0.0, "call") == pytest.approx(0.0)
    assert bs_price(90, 100, 0, 0.05, 0.2, 0.0, "put") == pytest.approx(10.0)
    assert bs_price(110, 100, 0, 0.05, 0.2, 0.0, "put") == pytest.approx(0.0)


def test_greeks_at_expiry_zero_except_delta():
    # Gamma, Vega, Theta, Rho all degenerate to 0 at T=0
    p = dict(S=100, K=100, T=0, r=0.05, sigma=0.2, q=0.0)
    assert bs_gamma(**p) == 0.0
    assert bs_vega(**p) == 0.0
    assert bs_theta(**p, option_type="call") == 0.0
    assert bs_rho(**p, option_type="call") == 0.0


def test_invalid_option_type_raises():
    with pytest.raises(ValueError, match="option_type"):
        bs_price(100, 100, 0.5, 0.05, 0.2, 0.0, "banana")


def test_invalid_sigma_raises():
    with pytest.raises(ValueError, match="sigma"):
        bs_price(100, 100, 0.5, 0.05, 0.0, 0.0, "call")


def test_bs_greeks_keys():
    g = bs_greeks(100, 100, 0.5, 0.05, 0.2, 0.0, "call")
    assert set(g.keys()) == {"price", "delta", "gamma", "vega", "theta", "rho"}
