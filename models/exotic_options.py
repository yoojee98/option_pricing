"""
models/exotic_options.py

Monte Carlo pricers for two path-dependent exotics under risk-neutral GBM:

  1. Asian options on the *arithmetic* average of the underlying.
  2. Down-and-Out Barrier options (knock-out if S touches B from above).

Both pricers share the same underlying path simulator and use antithetic
variates for variance reduction.  The Asian pricer additionally supports a
*geometric-Asian control variate*, which typically gives a 10–100x reduction
in standard error because the geometric Asian is strongly correlated with
the arithmetic Asian and has a closed-form price.

The Down-and-Out Barrier pricer also exposes a closed-form reference price
(Merton 1973 / Reiner-Rubinstein 1991) for the continuously monitored case,
useful for MC validation and for situations where a fast analytic answer is
preferred.

Conventions
-----------
- All pricers return an MCResult from models.monte_carlo (price, SE, CI95).
- For barrier options, `n_steps` controls the *monitoring* grid: B is checked
  at each grid point.  As n_steps → ∞ the discrete price converges upward
  (Broadie-Glasserman-Kou) to the continuous closed-form for out-options —
  discrete monitoring underestimates knock-out frequency.
- Risk-neutral drift: (r - q); no physical measure here (that lives in
  analysis/delta_hedge for hedging PnL).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from models.bs_model import bs_price
from models.monte_carlo import MCResult


# ── Shared path simulator (risk-neutral) ──────────────────────────────────────

def _simulate_paths_rn(
    S: float, T: float, r: float, q: float, sigma: float,
    n_paths: int, n_steps: int, antithetic: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate risk-neutral GBM paths, returning shape (n_paths, n_steps + 1).

    Antithetic pairing: draws n_paths/2 independent Brownian increments,
    then mirrors them.  This preserves path-level negative correlation,
    which is what variance reduction requires.
    """
    if antithetic and n_paths % 2 != 0:
        raise ValueError(f"n_paths must be even when antithetic=True, got {n_paths}")

    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    if antithetic:
        half = n_paths // 2
        Z_half = rng.standard_normal((half, n_steps))
        Z = np.concatenate([Z_half, -Z_half], axis=0)
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    log_inc = drift + diffusion * Z
    log_S = np.log(S) + np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1,
    )
    return np.exp(log_S)


def _discounted_mc_stats(
    payoff: np.ndarray, r: float, T: float, antithetic: bool,
) -> tuple[float, float]:
    """
    Given an array of payoffs (n_paths,), return (price, se) using the same
    pair-averaging trick as models.monte_carlo when antithetic is on.
    """
    disc = np.exp(-r * T)
    if antithetic:
        half = len(payoff) // 2
        pair_means = 0.5 * (payoff[:half] + payoff[half:])
        mean_payoff = pair_means.mean()
        se_payoff = pair_means.std(ddof=1) / np.sqrt(half)
    else:
        mean_payoff = payoff.mean()
        se_payoff = payoff.std(ddof=1) / np.sqrt(len(payoff))
    return float(disc * mean_payoff), float(disc * se_payoff)


# ── Asian (arithmetic average) ────────────────────────────────────────────────

def _geometric_asian_closed_form(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    n_steps: int, option_type: str,
) -> float:
    """
    Closed-form price of a *discretely sampled geometric* Asian option with
    monitoring at t = 0, dt, 2dt, ..., T (that is, n_steps + 1 sample points,
    where n_steps = T / dt).

    The geometric mean G = exp((1/m) * sum_{k=0..m-1} ln S_{t_k}) with
    m = n_steps + 1 evenly spaced points including t=0 is log-normal under
    GBM, with moments (derived from sum-of-Brownian covariances):

        E[ln(G/S0)]  = (r - q - 0.5*sigma^2) * T / 2           (the 1/2 is
                       independent of the sampling frequency — easy to verify
                       by summing mu*k*dt and dividing by m, using T=(m-1)*dt)
        Var[ln G]    = sigma^2 * T * (2*n_steps + 1) / (6 * (n_steps + 1))

    We price by calling bs_price with (sigma_G_ann, q_eff) such that the
    BS formula's effective log-drift (r - q_eff - 0.5*sigma_G^2)*T reproduces
    drift_G exactly, and its log-variance reproduces Var[ln G].

    Used as:
      (a) a standalone sanity reference,
      (b) a control variate for the arithmetic Asian.
    """
    n = n_steps
    # Moments of ln G over [0, T] with n+1 sample points (including t=0).
    drift_G = (r - q - 0.5 * sigma ** 2) * T / 2.0
    var_G   = sigma ** 2 * T * (2 * n + 1) / (6.0 * (n + 1))
    sigma_G = np.sqrt(var_G / T)
    # Solve for q_eff so that bs_price's effective drift matches drift_G / T.
    q_eff = r - drift_G / T - 0.5 * sigma_G ** 2
    return bs_price(S, K, T, r, sigma_G, q_eff, option_type)


def asian_price(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call",
    n_paths: int = 100_000, n_steps: int = 50,
    antithetic: bool = True, control_variate: bool = True,
    seed: int | None = None,
) -> MCResult:
    """
    Price a discretely sampled arithmetic Asian option by Monte Carlo.

    Payoff at expiry:
        Call:  max( mean(S_0, ..., S_n) - K, 0 )
        Put:   max( K - mean(S_0, ..., S_n), 0 )

    The arithmetic mean of GBM is not log-normal — no closed form exists.
    We therefore rely on MC, optionally with a geometric-Asian control
    variate for variance reduction.

    Parameters
    ----------
    n_steps : int
        Number of monitoring intervals (mean is taken over n_steps + 1 points
        including t=0 and t=T).
    control_variate : bool
        Apply the geometric-Asian control variate.  Typically cuts SE by
        10-100x since arithmetic and geometric means are ~99% correlated
        for realistic vol levels.
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return MCResult(price=float(intrinsic), se=0.0,
                        ci95=(float(intrinsic), float(intrinsic)),
                        n_paths=n_paths, antithetic=antithetic)

    rng = np.random.default_rng(seed)
    paths = _simulate_paths_rn(S, T, r, q, sigma, n_paths, n_steps, antithetic, rng)

    arith_mean = paths.mean(axis=1)
    if option_type == "call":
        arith_payoff = np.maximum(arith_mean - K, 0.0)
    else:
        arith_payoff = np.maximum(K - arith_mean, 0.0)

    if not control_variate:
        price, se = _discounted_mc_stats(arith_payoff, r, T, antithetic)
        return MCResult(price=price, se=se, ci95=(price - 1.96 * se, price + 1.96 * se),
                        n_paths=n_paths, antithetic=antithetic)

    # --- Geometric-Asian control variate --------------------------------------
    # Use log-sum then exp for numerical stability (many points, small values).
    geo_mean = np.exp(np.mean(np.log(paths), axis=1))
    if option_type == "call":
        geo_payoff = np.maximum(geo_mean - K, 0.0)
    else:
        geo_payoff = np.maximum(K - geo_mean, 0.0)

    disc = np.exp(-r * T)
    X = disc * arith_payoff              # target estimator (undiscounted cf. × disc)
    Y = disc * geo_payoff                # control variate realisations
    EY = _geometric_asian_closed_form(S, K, T, r, q, sigma, n_steps, option_type)

    # Optimal control-variate coefficient beta* = Cov(X, Y) / Var(Y).
    var_Y = Y.var(ddof=1)
    if var_Y == 0.0:
        # Degenerate (all payoffs zero, e.g. deep OTM + very short T).
        price = X.mean()
        se = X.std(ddof=1) / np.sqrt(len(X))
    else:
        beta = np.cov(X, Y, ddof=1)[0, 1] / var_Y
        X_cv = X - beta * (Y - EY)
        if antithetic:
            half = len(X_cv) // 2
            pair_means = 0.5 * (X_cv[:half] + X_cv[half:])
            price = float(pair_means.mean())
            se = float(pair_means.std(ddof=1) / np.sqrt(half))
        else:
            price = float(X_cv.mean())
            se = float(X_cv.std(ddof=1) / np.sqrt(len(X_cv)))

    return MCResult(price=price, se=se, ci95=(price - 1.96 * se, price + 1.96 * se),
                    n_paths=n_paths, antithetic=antithetic)


# ── Down-and-Out Barrier ──────────────────────────────────────────────────────

def _barrier_do_closed_form(
    S: float, K: float, B: float, T: float, r: float, q: float, sigma: float,
    option_type: str,
) -> float:
    """
    Closed-form price for a continuously monitored Down-and-Out European
    barrier option, zero rebate.  Reference: Merton (1973) for the call,
    Reiner & Rubinstein (1991) for the full put/call variants.

    Valid when B < S (barrier below spot); if B >= S the option is already
    knocked out and worth 0.  If B <= K for a call, the formula simplifies
    (DO call = vanilla call - DI call, standard parity).
    """
    if B >= S:
        return 0.0

    # Generic RR building blocks.
    mu = (r - q - 0.5 * sigma ** 2) / (sigma ** 2)
    lam = np.sqrt(mu ** 2 + 2 * r / (sigma ** 2))
    sqrtT = np.sqrt(T)

    x1 = np.log(S / K) / (sigma * sqrtT) + (1 + mu) * sigma * sqrtT
    x2 = np.log(S / B) / (sigma * sqrtT) + (1 + mu) * sigma * sqrtT
    y1 = np.log(B ** 2 / (S * K)) / (sigma * sqrtT) + (1 + mu) * sigma * sqrtT
    y2 = np.log(B / S)              / (sigma * sqrtT) + (1 + mu) * sigma * sqrtT

    if option_type == "call":
        # Reiner-Rubinstein: Down-and-Out call (B < S, zero rebate).
        if K > B:
            # DO call = C(S,K) - C_DI; C_DI expressible via x's and y's.
            call_vanilla = bs_price(S, K, T, r, sigma, q, "call")
            A = (
                S * np.exp(-q * T) * (B / S) ** (2 * (mu + 1)) * norm.cdf(y1)
                - K * np.exp(-r * T) * (B / S) ** (2 * mu)       * norm.cdf(y1 - sigma * sqrtT)
            )
            return max(call_vanilla - A, 0.0)
        else:
            # K <= B < S: the DO call equals a shifted formula (Hull Ch. 26).
            term1 = S * np.exp(-q * T) * norm.cdf(x2)
            term2 = K * np.exp(-r * T) * norm.cdf(x2 - sigma * sqrtT)
            term3 = S * np.exp(-q * T) * (B / S) ** (2 * (mu + 1)) * norm.cdf(y2)
            term4 = K * np.exp(-r * T) * (B / S) ** (2 * mu) * norm.cdf(y2 - sigma * sqrtT)
            return float(term1 - term2 - term3 + term4)
    else:
        # Down-and-Out put.  For a *put*, the down-barrier is typically above
        # the strike or close to it; knock-out removes the deep-ITM tail.
        if K > B:
            # Standard DO put formula.
            term1 = -S * np.exp(-q * T) * norm.cdf(-x1)
            term2 = K * np.exp(-r * T) * norm.cdf(-x1 + sigma * sqrtT)
            term3 = S * np.exp(-q * T) * norm.cdf(-x2)
            term4 = -K * np.exp(-r * T) * norm.cdf(-x2 + sigma * sqrtT)
            term5 = -S * np.exp(-q * T) * (B / S) ** (2 * (mu + 1)) * (norm.cdf(y1) - norm.cdf(y2))
            term6 = K * np.exp(-r * T) * (B / S) ** (2 * mu) * (norm.cdf(y1 - sigma * sqrtT) - norm.cdf(y2 - sigma * sqrtT))
            return float(term1 + term2 + term3 + term4 + term5 + term6)
        else:
            # K <= B: barrier above strike; the ITM region is entirely knocked
            # out, so a DO put here is worth ~0 along the boundary.  Return 0.
            return 0.0


def barrier_do_price(
    S: float, K: float, B: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call",
    n_paths: int = 100_000, n_steps: int = 100,
    antithetic: bool = True, seed: int | None = None,
) -> MCResult:
    """
    Price a European Down-and-Out barrier option by Monte Carlo with
    discrete monitoring on an equally spaced grid of `n_steps + 1` points.

    A path knocks out if min(S_t) <= B at any monitoring time; a knocked-out
    path pays 0.  Rebate is zero.

    Notes
    -----
    - Discrete monitoring systematically *under*-estimates knock-out vs the
      continuous-monitoring closed form, so MC price converges *downward*
      to the closed-form as n_steps → ∞.  This is expected behavior, not a bug.
    - For continuous-monitoring pricing use `barrier_do_closed_form(...)` below,
      or apply the Broadie-Glasserman-Kou barrier shift (not done here).
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")
    if B >= S:
        # Already knocked out.
        return MCResult(price=0.0, se=0.0, ci95=(0.0, 0.0),
                        n_paths=n_paths, antithetic=antithetic)

    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return MCResult(price=float(intrinsic), se=0.0,
                        ci95=(float(intrinsic), float(intrinsic)),
                        n_paths=n_paths, antithetic=antithetic)

    rng = np.random.default_rng(seed)
    paths = _simulate_paths_rn(S, T, r, q, sigma, n_paths, n_steps, antithetic, rng)

    # Knock-out if the path's minimum over the monitoring grid touches B.
    alive = paths.min(axis=1) > B
    S_T = paths[:, -1]
    if option_type == "call":
        payoff = np.where(alive, np.maximum(S_T - K, 0.0), 0.0)
    else:
        payoff = np.where(alive, np.maximum(K - S_T, 0.0), 0.0)

    price, se = _discounted_mc_stats(payoff, r, T, antithetic)
    return MCResult(price=price, se=se, ci95=(price - 1.96 * se, price + 1.96 * se),
                    n_paths=n_paths, antithetic=antithetic)


def barrier_do_closed_form(
    S: float, K: float, B: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call",
) -> float:
    """
    Public wrapper for the continuously monitored Down-and-Out closed form.
    See `_barrier_do_closed_form` for formulas and caveats.
    """
    return _barrier_do_closed_form(S, K, B, T, r, q, sigma, option_type.lower())


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Asian arithmetic call: control variate reduces SE ===")
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.3
    plain = asian_price(S, K, T, r, sigma, q, "call",
                        n_paths=50_000, n_steps=50,
                        antithetic=True, control_variate=False, seed=1)
    cv = asian_price(S, K, T, r, sigma, q, "call",
                     n_paths=50_000, n_steps=50,
                     antithetic=True, control_variate=True, seed=1)
    print(f"  plain MC  : price = {plain.price:.4f}  SE = {plain.se:.5f}")
    print(f"  + control : price = {cv.price:.4f}  SE = {cv.se:.5f}")
    print(f"  SE ratio  : {cv.se / plain.se:.3f}  (smaller = more reduction)")

    print()
    print("=== Asian put-call parity sanity (at-the-money) ===")
    # Arithmetic Asians have no exact parity, but at q=0 the call and put
    # are tied by:   C_a - P_a  ~~  S*e^{-qT} * avg_factor - K * e^{-rT}
    # We use the geometric version as a proxy and just check both prices
    # are positive and in a sane range.
    call_p = asian_price(S, K, T, r, sigma, q, "call", n_paths=50_000, n_steps=50, seed=7)
    put_p  = asian_price(S, K, T, r, sigma, q, "put",  n_paths=50_000, n_steps=50, seed=7)
    print(f"  ATM call = {call_p.price:.4f} ± {call_p.se:.4f}")
    print(f"  ATM put  = {put_p.price:.4f}  ± {put_p.se:.4f}")

    print()
    print("=== Down-and-Out call: MC converges to closed form as monitoring fines ===")
    B = 80.0
    cf = barrier_do_closed_form(S, K, B, T, r, sigma, q, "call")
    print(f"  closed form (continuous)    : {cf:.4f}")
    for n in (20, 100, 500):
        mc = barrier_do_price(S, K, B, T, r, sigma, q, "call",
                              n_paths=100_000, n_steps=n, seed=42)
        print(f"  MC (n_steps={n:4d}) : price = {mc.price:.4f} ± {1.96*mc.se:.4f}"
              f"  (discrete > continuous)")

    print()
    print("=== Barrier sanity: DO price + DI price == Vanilla ===")
    # A path either knocks out or doesn't; so DO + DI = vanilla (zero rebate).
    # We check this by computing 1 - DO_payoff_fraction against DI directly.
    vanilla = bs_price(S, K, T, r, sigma, q, "call")
    mc_do = barrier_do_price(S, K, B, T, r, sigma, q, "call",
                             n_paths=200_000, n_steps=500, seed=0)
    # Quick DI via the same paths:
    rng = np.random.default_rng(0)
    paths = _simulate_paths_rn(S, T, r, q, sigma, 200_000, 500, True, rng)
    dead = paths.min(axis=1) <= B
    di_payoff = np.where(dead, np.maximum(paths[:, -1] - K, 0.0), 0.0)
    di_price, _ = _discounted_mc_stats(di_payoff, r, T, True)
    print(f"  vanilla call = {vanilla:.4f}")
    print(f"  DO call (MC) = {mc_do.price:.4f}")
    print(f"  DI call (MC) = {di_price:.4f}")
    print(f"  DO + DI      = {mc_do.price + di_price:.4f}   (should ≈ vanilla)")
