"""
models/monte_carlo.py

Monte Carlo pricer for European options under Geometric Brownian Motion with
continuous dividend yield, with optional antithetic-variates variance reduction.

Model
-----
Risk-neutral GBM (dividend-adjusted):

    dS = (r - q) S dt + sigma S dW

Closed-form terminal distribution:

    S_T = S_0 * exp( (r - q - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z ),   Z ~ N(0, 1)

For vanilla European payoffs we don't need to simulate the full path — one
draw of Z per path is sufficient.  This keeps the estimator unbiased and
makes the 1/sqrt(N) convergence easy to reason about.

Antithetic variates
-------------------
For every Z drawn, also use -Z.  This exploits the symmetry of the standard
normal: paired payoffs f(Z) and f(-Z) are negatively correlated when the
payoff is monotonic in Z (true for calls and puts), so

    Var( 0.5 * (f(Z) + f(-Z)) )  <  0.5 * Var( f(Z) )

The variance reduction is usually ~40-70% for ATM/ITM vanilla options; it
can be smaller for deep OTM where most paired draws contribute zero.

We report two estimates:
  - price:  discounted mean payoff
  - se:     standard error = std_of_sample_mean * exp(-rT)
  - ci95:   95% confidence interval (price ± 1.96 * se)

Greeks via pathwise / bump-and-reprice
--------------------------------------
- Delta, Gamma: bump-and-reprice with common random numbers (CRN).  CRN means
  we reuse the *same* Z array across the three price evaluations, which
  eliminates the Monte Carlo noise between them — without CRN, FD Greeks
  on MC would be unusable at any realistic N.
- Vega, Rho: same idea, bump sigma or r, reuse Z.
- Theta: bump T (forward difference — theta is ∂V/∂t = -∂V/∂T).

All bump sizes are chosen to sit well above the sqrt-N MC noise floor yet
small enough that higher-order bias is negligible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class MCResult:
    """Monte Carlo price estimate with uncertainty."""
    price: float
    se: float                  # standard error of the mean (discounted)
    ci95: tuple[float, float]  # 95% confidence interval
    n_paths: int               # effective number of samples (counts antithetic pairs as 2)
    antithetic: bool


# ── Core simulator ────────────────────────────────────────────────────────────

def _terminal_prices(
    S: float, T: float, r: float, q: float, sigma: float,
    n_paths: int, antithetic: bool, rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw terminal asset prices S_T under risk-neutral GBM.

    If antithetic=True, draws n_paths/2 independent Z values and returns
    their mirrored pair, giving exactly n_paths samples.  n_paths must be
    even in that case.
    """
    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError(f"n_paths must be even when antithetic=True, got {n_paths}")
        half = n_paths // 2
        Z_half = rng.standard_normal(half)
        Z = np.concatenate([Z_half, -Z_half])
    else:
        Z = rng.standard_normal(n_paths)

    drift = (r - q - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * Z
    return S * np.exp(drift + diffusion)


def _payoff(S_T: np.ndarray, K: float, option_type: str) -> np.ndarray:
    """Terminal payoff for a European call or put."""
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    else:
        return np.maximum(K - S_T, 0.0)


# ── Pricing ───────────────────────────────────────────────────────────────────

def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    n_paths: int = 100_000,
    antithetic: bool = True,
    seed: int | None = None,
) -> MCResult:
    """
    Price a European option by Monte Carlo.

    Parameters
    ----------
    S, K, T, r, sigma, q : float
        Standard BS inputs.
    option_type : str
        'call' or 'put'.
    n_paths : int
        Number of simulated paths.  Error ~ O(1/sqrt(n_paths)); 100k gives
        ~0.01 relative error for ATM vanillas.
    antithetic : bool
        Enable antithetic variates (requires n_paths even).
    seed : int or None
        Seed for numpy's PCG64 RNG.  Pass an int for reproducibility.

    Returns
    -------
    MCResult
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if n_paths < 2:
        raise ValueError(f"n_paths must be >= 2, got {n_paths}")

    # T == 0: intrinsic, matching bs_price convention
    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return MCResult(price=float(intrinsic), se=0.0,
                        ci95=(float(intrinsic), float(intrinsic)),
                        n_paths=n_paths, antithetic=antithetic)

    rng = np.random.default_rng(seed)
    S_T = _terminal_prices(S, T, r, q, sigma, n_paths, antithetic, rng)
    payoff = _payoff(S_T, K, option_type)

    disc = np.exp(-r * T)

    if antithetic:
        # Average each antithetic pair first, then take the mean/std across pairs.
        # This is the correct variance estimator — the n_paths samples are *not*
        # i.i.d., the pair-averages are.
        half = n_paths // 2
        pair_means = 0.5 * (payoff[:half] + payoff[half:])
        mean_payoff = pair_means.mean()
        se_payoff = pair_means.std(ddof=1) / np.sqrt(half)
    else:
        mean_payoff = payoff.mean()
        se_payoff = payoff.std(ddof=1) / np.sqrt(n_paths)

    price = disc * mean_payoff
    se = disc * se_payoff
    ci95 = (price - 1.96 * se, price + 1.96 * se)

    return MCResult(price=float(price), se=float(se), ci95=ci95,
                    n_paths=n_paths, antithetic=antithetic)


# ── Greeks via CRN bump-and-reprice ───────────────────────────────────────────

def _price_with_Z(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    option_type: str, Z: np.ndarray,
) -> float:
    """Price using a pre-drawn Z array (common random numbers)."""
    drift = (r - q - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * Z
    S_T = S * np.exp(drift + diffusion)
    payoff = _payoff(S_T, K, option_type)
    return float(np.exp(-r * T) * payoff.mean())


def mc_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    n_paths: int = 200_000,
    antithetic: bool = True,
    seed: int | None = 0,
) -> dict:
    """
    Monte Carlo Greeks via bump-and-reprice with common random numbers (CRN).

    CRN is essential here: reusing the same Z array across the bumped prices
    cancels the O(1/sqrt(N)) noise and recovers O(h) finite-difference error.
    Without CRN, FD Greeks on MC would require ~10^8 paths to be usable.

    Returns dict with: price, delta, gamma, vega, theta, rho, se.
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    rng = np.random.default_rng(seed)
    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError(f"n_paths must be even when antithetic=True, got {n_paths}")
        half = n_paths // 2
        Z_half = rng.standard_normal(half)
        Z = np.concatenate([Z_half, -Z_half])
    else:
        Z = rng.standard_normal(n_paths)

    base = _price_with_Z(S, K, T, r, q, sigma, option_type, Z)

    # Delta, Gamma: bump spot, reuse Z
    h_S = 1e-2 * S
    up_S = _price_with_Z(S + h_S, K, T, r, q, sigma, option_type, Z)
    dn_S = _price_with_Z(S - h_S, K, T, r, q, sigma, option_type, Z)
    delta = (up_S - dn_S) / (2 * h_S)
    gamma = (up_S - 2 * base + dn_S) / (h_S ** 2)

    # Vega: bump sigma
    h_sig = 1e-3
    up_sig = _price_with_Z(S, K, T, r, q, sigma + h_sig, option_type, Z)
    dn_sig = _price_with_Z(S, K, T, r, q, sigma - h_sig, option_type, Z)
    vega = (up_sig - dn_sig) / (2 * h_sig)

    # Rho: bump r (note: r enters both drift and discount)
    h_r = 1e-4
    up_r = _price_with_Z(S, K, T, r + h_r, q, sigma, option_type, Z)
    dn_r = _price_with_Z(S, K, T, r - h_r, q, sigma, option_type, Z)
    rho = (up_r - dn_r) / (2 * h_r)

    # Theta: ∂V/∂t = -∂V/∂T
    h_T = 1e-4
    up_T = _price_with_Z(S, K, T + h_T, r, q, sigma, option_type, Z)
    dn_T = _price_with_Z(S, K, T - h_T, r, q, sigma, option_type, Z)
    theta = -(up_T - dn_T) / (2 * h_T)

    # SE of the base price
    res = mc_price(S, K, T, r, sigma, q, option_type, n_paths, antithetic, seed=seed)

    return {
        "price": base,
        "delta": delta,
        "gamma": gamma,
        "vega":  vega,
        "theta": theta,
        "rho":   rho,
        "se":    res.se,
    }


# ── Variance-reduction diagnostic ─────────────────────────────────────────────

def compare_antithetic(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call",
    n_paths: int = 100_000, seed: int = 0,
) -> dict:
    """
    Price the same option with and without antithetic variates at identical
    n_paths.  Returns a dict with both estimates and the SE reduction ratio.
    Useful for benchmarking the variance-reduction gain.
    """
    plain = mc_price(S, K, T, r, sigma, q, option_type, n_paths,
                     antithetic=False, seed=seed)
    anti  = mc_price(S, K, T, r, sigma, q, option_type, n_paths,
                     antithetic=True, seed=seed)
    return {
        "plain":  plain,
        "anti":   anti,
        "se_ratio": anti.se / plain.se if plain.se > 0 else float("nan"),
    }


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    import time
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models.bs_model import bs_price, bs_greeks

    print("=== MC vs Black-Scholes (European call) ===")
    S, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.05, 0.013, 0.22
    bs = bs_price(S, K, T, r, sigma, q, "call")
    for n in (10_000, 100_000, 1_000_000):
        res = mc_price(S, K, T, r, sigma, q, "call", n_paths=n, seed=42)
        err = res.price - bs
        in_ci = res.ci95[0] <= bs <= res.ci95[1]
        print(f"  N={n:>8}  MC={res.price:.4f} ± {1.96*res.se:.4f}  "
              f"BS={bs:.4f}  err={err:+.4f}  BS∈CI95? {in_ci}")

    print()
    print("=== Antithetic variance reduction ===")
    cmp = compare_antithetic(S, K, T, r, sigma, q, "call", n_paths=100_000, seed=42)
    print(f"  Plain       : {cmp['plain'].price:.4f}  SE={cmp['plain'].se:.5f}")
    print(f"  Antithetic  : {cmp['anti'].price:.4f}  SE={cmp['anti'].se:.5f}")
    print(f"  SE ratio    : {cmp['se_ratio']:.3f}  (lower = better)")

    print()
    print("=== MC Greeks vs BS Greeks (CRN bump-and-reprice) ===")
    t0 = time.perf_counter()
    g_mc = mc_greeks(S, K, T, r, sigma, q, "call", n_paths=200_000, seed=42)
    elapsed = time.perf_counter() - t0
    g_bs = bs_greeks(S, K, T, r, sigma, q, "call")
    print(f"  (200k paths, 5-bump CRN, {elapsed*1000:.0f} ms)")
    for key in ("price", "delta", "gamma", "vega", "theta", "rho"):
        print(f"  {key:6s}: MC={g_mc[key]:+.6f}  BS={g_bs[key]:+.6f}  "
              f"diff={g_mc[key] - g_bs[key]:+.2e}")
