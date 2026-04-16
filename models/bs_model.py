"""
models/bs_model.py

Generalized Black-Scholes pricing for European options with continuous dividend yield.
Supports calls and puts; computes full first-order Greeks (Delta, Gamma, Vega, Theta, Rho).

Model:
    dS = (r - q) S dt + σ S dW

Closed-form price:
    Call: S * e^{-qT} * N(d1) - K * e^{-rT} * N(d2)
    Put:  K * e^{-rT} * N(-d2) - S * e^{-qT} * N(-d1)

where:
    d1 = [log(S/K) + (r - q + σ²/2) T] / (σ √T)
    d2 = d1 - σ √T
"""

import numpy as np
from scipy.stats import norm


# ── Internal helpers ──────────────────────────────────────────────────────────

def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    """Return (d1, d2) for the GBS formula."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


# ── Pricing ───────────────────────────────────────────────────────────────────

def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> float:
    """
    Generalized Black-Scholes price for a European option.

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualized volatility.
    q : float
        Continuously compounded dividend yield (default 0).
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        Option price.
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return float(intrinsic)

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# ── Greeks ────────────────────────────────────────────────────────────────────

def bs_delta(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call",
) -> float:
    """
    Delta: ∂V/∂S

    Call: e^{-qT} N(d1)
    Put:  e^{-qT} [N(d1) - 1]
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


def bs_gamma(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0,
) -> float:
    """
    Gamma: ∂²V/∂S²  (identical for calls and puts)

    e^{-qT} * n(d1) / (S σ √T)
    """
    if T <= 0:
        return 0.0

    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0,
) -> float:
    """
    Vega: ∂V/∂σ  (identical for calls and puts)

    S * e^{-qT} * n(d1) * √T

    Returned per 1.0 move in sigma (not per 1% move).
    """
    if T <= 0:
        return 0.0

    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_theta(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call",
) -> float:
    """
    Theta: ∂V/∂t  (per calendar year, market convention)

    Call: -[S e^{-qT} n(d1) σ / (2√T)] + q S e^{-qT} N(d1)  - r K e^{-rT} N(d2)
    Put:  -[S e^{-qT} n(d1) σ / (2√T)] - q S e^{-qT} N(-d1) + r K e^{-rT} N(-d2)

    Convention: returns ∂V/∂t (negative for long options — price decays as
    calendar time advances).  Relationship to ∂V/∂T: bs_theta = -∂V/∂T.
    To get per-day decay: bs_theta(...) / 365.
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    if T <= 0:
        return 0.0

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    sqrt_T = np.sqrt(T)

    common = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * sqrt_T)

    if option_type == "call":
        return common + q * S * np.exp(-q * T) * norm.cdf(d1) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return common - q * S * np.exp(-q * T) * norm.cdf(-d1) + r * K * np.exp(-r * T) * norm.cdf(-d2)


def bs_rho(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call",
) -> float:
    """
    Rho: ∂V/∂r

    Call:  K T e^{-rT} N(d2)
    Put:  -K T e^{-rT} N(-d2)

    Returned per 1.0 move in r (not per 1% move).
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    if T <= 0:
        return 0.0

    _, d2 = _d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call",
) -> dict:
    """
    Return all Greeks in a single dict.

    Keys: price, delta, gamma, vega, theta, rho
    """
    return {
        "price": bs_price(S, K, T, r, sigma, q, option_type),
        "delta": bs_delta(S, K, T, r, sigma, q, option_type),
        "gamma": bs_gamma(S, K, T, r, sigma, q),
        "vega":  bs_vega(S, K, T, r, sigma, q),
        "theta": bs_theta(S, K, T, r, sigma, q, option_type),
        "rho":   bs_rho(S, K, T, r, sigma, q, option_type),
    }


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Reference values from Hull (Options, Futures, and Other Derivatives, 10e)
    # S=49, K=50, T=0.3846, r=0.05, q=0, sigma=0.2
    # Call: ~2.40,  Delta: ~0.522
    S, K, T, r, q, sigma = 49.0, 50.0, 0.3846, 0.05, 0.0, 0.2

    print("=== Hull reference (q=0) ===")
    g = bs_greeks(S, K, T, r, sigma, q, "call")
    for name, val in g.items():
        print(f"  {name:6s}: {val:.4f}")

    print()
    print("=== SPY-like ATM call (q=1.3%) ===")
    # Use typical SPY values from data_loader
    S2, K2, T2, r2, q2, sigma2 = 700.0, 700.0, 0.5, 0.05, 0.013, 0.18
    g2 = bs_greeks(S2, K2, T2, r2, sigma2, q2, "call")
    for name, val in g2.items():
        print(f"  {name:6s}: {val:.4f}")

    print()
    print("=== Put-Call Parity check (should be ~0) ===")
    call_p = bs_price(S2, K2, T2, r2, sigma2, q2, "call")
    put_p  = bs_price(S2, K2, T2, r2, sigma2, q2, "put")
    parity_lhs = call_p - put_p
    parity_rhs = S2 * np.exp(-q2 * T2) - K2 * np.exp(-r2 * T2)
    print(f"  C - P          = {parity_lhs:.6f}")
    print(f"  S*e^-qT - K*e^-rT = {parity_rhs:.6f}")
    print(f"  Difference     = {abs(parity_lhs - parity_rhs):.2e}")
