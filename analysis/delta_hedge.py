"""
analysis/delta_hedge.py

Dynamic Delta-hedging simulation for a short European option under GBM.

Setup
-----
A trader sells one European option at its Black-Scholes price at t=0, then
rebalances a stock + cash portfolio on a discrete grid of rehedge times so
that the portfolio Delta matches the option Delta at each step.  Between
rebalances the portfolio is *not* Delta-neutral — this mismatch is what
generates hedging PnL.

PnL decomposition (continuous-time, Ito):

    dPi = - Theta_opt dt - 0.5 * Gamma_opt * (dS)^2 + O(dt^{3/2})

Under the physical measure dS = mu*S*dt + sigma_r * S * dW, where sigma_r is
the *realized* vol.  Taking expectations and using E[(dS)^2] = sigma_r^2*S^2*dt:

    E[dPi] ~ 0.5 * Gamma_opt * S^2 * (sigma_i^2 - sigma_r^2) * dt

where sigma_i is the implied vol used to price and hedge.  So:
  - If realized == implied: expected PnL is zero; residual is pure discretization
    error (variance ~ 1/N_steps for a Delta-hedged short gamma book).
  - If realized > implied: hedger loses on average (selling cheap gamma).
  - If realized < implied: hedger wins on average.

This module exposes that behavior so the user can stress the book and see the
PnL distribution shift as sigma_r diverges from sigma_i.

Conventions
-----------
- Short one option (trader is the seller).  Terminal cashflow at expiry is
  `-payoff(S_T)`.
- Hedge is rebalanced at `n_steps + 1` grid points including t=0.  Cash earns
  the risk-free rate r continuously; stock pays continuous dividend q which
  the hedger receives on their long stock position.
- All Greeks used in the hedge (Delta for rebalancing, Gamma/Vega for exposure
  tracking) are computed with sigma_implied — that is what a real trader
  uses at the hedge desk, even if realized vol differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.bs_model import bs_price, bs_delta, bs_gamma, bs_vega


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class HedgePath:
    """Time series for a single hedging trajectory.  Lengths = n_steps + 1."""
    t: np.ndarray           # grid times in years, shape (n_steps+1,)
    S: np.ndarray           # spot path
    delta: np.ndarray       # option Delta at each grid point
    gamma: np.ndarray       # option Gamma at each grid point
    vega: np.ndarray        # option Vega at each grid point
    cash: np.ndarray        # cash account balance after rebalance at each grid point
    portfolio: np.ndarray   # total portfolio value (short option + hedge)
    pnl: float              # terminal PnL of the hedged book


@dataclass
class HedgeResult:
    """Aggregate statistics across many hedging trajectories."""
    pnl: np.ndarray         # shape (n_sims,), terminal PnL per trajectory
    mean: float
    std: float
    p05: float
    p95: float
    sharpe: float           # mean / std (NaN if std == 0)
    n_sims: int
    n_steps: int
    sigma_implied: float
    sigma_realized: float


# ── GBM path simulation ───────────────────────────────────────────────────────

def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_sims: int,
    q: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate GBM paths under the *physical* measure.

        dS = (mu - q) S dt + sigma S dW

    Parameters
    ----------
    S0 : initial spot
    mu : physical drift (not r — this is the realized-world drift).
    sigma : realized (physical) volatility used to generate paths.
    T : horizon in years
    n_steps : number of sub-intervals (grid has n_steps+1 points including t=0)
    n_sims : number of independent paths
    q : continuous dividend yield (stock pays out; total return drift is mu)
    seed : RNG seed

    Returns
    -------
    S : ndarray of shape (n_sims, n_steps + 1)
        S[:, 0] == S0.  Exact log-Euler (no time-discretization bias for GBM).
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if n_sims < 1:
        raise ValueError(f"n_sims must be >= 1, got {n_sims}")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (mu - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    Z = rng.standard_normal((n_sims, n_steps))
    log_increments = drift + diffusion * Z
    log_S = np.log(S0) + np.concatenate(
        [np.zeros((n_sims, 1)), np.cumsum(log_increments, axis=1)],
        axis=1,
    )
    return np.exp(log_S)


# ── Single-path hedge ─────────────────────────────────────────────────────────

def hedge_single_path(
    S_path: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma_implied: float,
    q: float = 0.0,
    option_type: str = "call",
    return_trajectory: bool = False,
) -> HedgePath | float:
    """
    Run the Delta hedge on a single price path.

    Trader is short one option.  At each grid point t_i:
      1. Compute option Delta with sigma_implied and current (S_i, T - t_i).
      2. Rebalance the stock position so the hedge matches: hold +Delta_i shares.
      3. Adjust cash by -(Delta_i - Delta_{i-1}) * S_i (funding / proceeds).
      4. Between steps, cash grows at r and stock pays dividend q on long shares.

    At t=0 the trader receives the option premium V_0 and uses it as initial cash
    (before funding the first Delta_0 share purchase).

    At T, the option is settled: trader pays `payoff(S_T)`.  Terminal hedged PnL is:

        cash_T  +  Delta_last * S_T  -  payoff(S_T)

    which should be close to 0 when sigma_realized == sigma_implied and the
    rebalance grid is fine.

    Parameters
    ----------
    S_path : ndarray, shape (n_steps+1,)
        Spot prices on the rebalance grid, S_path[0] = S0, S_path[-1] = S_T.
    K, T, r, sigma_implied, q, option_type : option parameters for pricing/Greeks.
    return_trajectory : if True, return a HedgePath; otherwise just the terminal PnL.

    Returns
    -------
    HedgePath or float
    """
    n_steps = len(S_path) - 1
    if n_steps < 1:
        raise ValueError(f"S_path must have length >= 2, got {len(S_path)}")
    dt = T / n_steps
    t_grid = np.linspace(0.0, T, n_steps + 1)

    # Initial option premium received (trader is short — this is a cash inflow).
    V0 = bs_price(S_path[0], K, T, r, sigma_implied, q, option_type)

    # Pre-allocate trajectory arrays only if requested, to keep MC runs cheap.
    if return_trajectory:
        delta_arr = np.empty(n_steps + 1)
        gamma_arr = np.empty(n_steps + 1)
        vega_arr = np.empty(n_steps + 1)
        cash_arr = np.empty(n_steps + 1)
        port_arr = np.empty(n_steps + 1)

    # Initial hedge: buy Delta_0 shares, funded from premium + borrowing.
    tau = T  # time-to-expiry at t=0
    delta_prev = bs_delta(S_path[0], K, tau, r, sigma_implied, q, option_type)
    cash = V0 - delta_prev * S_path[0]
    shares = delta_prev

    if return_trajectory:
        delta_arr[0] = delta_prev
        gamma_arr[0] = bs_gamma(S_path[0], K, tau, r, sigma_implied, q)
        vega_arr[0]  = bs_vega (S_path[0], K, tau, r, sigma_implied, q)
        cash_arr[0]  = cash
        # Portfolio = short option + long shares + cash
        port_arr[0]  = -V0 + shares * S_path[0] + cash  # == 0 at inception

    # Step through the grid.
    for i in range(1, n_steps + 1):
        S_i = S_path[i]
        tau = T - t_grid[i]

        # Accrue cash at r; dividends on the long stock position accrue to cash.
        cash = cash * np.exp(r * dt) + shares * S_path[i - 1] * (np.exp(q * dt) - 1.0)

        if tau > 0:
            delta_i = bs_delta(S_i, K, tau, r, sigma_implied, q, option_type)
        else:
            # At expiry Delta is degenerate; don't bother rebalancing — the option
            # settles at payoff(S_T) regardless of our last hedge.
            delta_i = delta_prev

        # Rebalance: buy/sell (delta_i - delta_prev) shares at S_i.
        cash -= (delta_i - delta_prev) * S_i
        shares = delta_i
        delta_prev = delta_i

        if return_trajectory:
            delta_arr[i] = delta_i
            if tau > 0:
                gamma_arr[i] = bs_gamma(S_i, K, tau, r, sigma_implied, q)
                vega_arr[i]  = bs_vega (S_i, K, tau, r, sigma_implied, q)
            else:
                gamma_arr[i] = 0.0
                vega_arr[i]  = 0.0
            # Option mark-to-market value (0 at expiry, else BS).
            V_i = (
                bs_price(S_i, K, tau, r, sigma_implied, q, option_type) if tau > 0
                else max(S_i - K, 0.0) if option_type == "call" else max(K - S_i, 0.0)
            )
            cash_arr[i]  = cash
            port_arr[i]  = -V_i + shares * S_i + cash

    # Terminal settlement of the short option.
    S_T = S_path[-1]
    payoff = max(S_T - K, 0.0) if option_type == "call" else max(K - S_T, 0.0)
    pnl = cash + shares * S_T - payoff

    if return_trajectory:
        return HedgePath(
            t=t_grid,
            S=np.asarray(S_path, dtype=float),
            delta=delta_arr,
            gamma=gamma_arr,
            vega=vega_arr,
            cash=cash_arr,
            portfolio=port_arr,
            pnl=float(pnl),
        )
    return float(pnl)


# ── Monte Carlo hedge ─────────────────────────────────────────────────────────

def simulate_delta_hedge(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma_implied: float,
    sigma_realized: float | None = None,
    mu: float | None = None,
    q: float = 0.0,
    option_type: str = "call",
    n_steps: int = 252,
    n_sims: int = 1000,
    seed: int | None = None,
) -> HedgeResult:
    """
    Monte Carlo simulation of a short-option Delta-hedging program.

    The option is priced and hedged with sigma_implied; paths are generated
    under the physical measure with sigma_realized (defaults to sigma_implied,
    i.e. the "ideal" case where hedging PnL has zero mean and residual variance
    comes only from rebalance discretization).

    Parameters
    ----------
    S0, K, T, r, q, option_type : standard option parameters.
    sigma_implied : vol used for pricing and for computing the hedge ratio.
    sigma_realized : physical vol driving the path simulation. Default = sigma_implied.
    mu : physical drift (defaults to r — risk-neutral). Irrelevant for mean
         hedging PnL under continuous rebalancing but affects discrete error;
         use r unless explicitly stress-testing.
    n_steps : number of rebalances per path.  252 = daily for a 1-year option.
    n_sims : number of Monte Carlo trajectories.
    seed : RNG seed.

    Returns
    -------
    HedgeResult with PnL array and summary stats.
    """
    if sigma_realized is None:
        sigma_realized = sigma_implied
    if mu is None:
        mu = r

    paths = simulate_gbm_paths(
        S0=S0, mu=mu, sigma=sigma_realized, T=T,
        n_steps=n_steps, n_sims=n_sims, q=q, seed=seed,
    )

    pnl = np.empty(n_sims)
    for j in range(n_sims):
        pnl[j] = hedge_single_path(
            S_path=paths[j],
            K=K, T=T, r=r, sigma_implied=sigma_implied,
            q=q, option_type=option_type,
            return_trajectory=False,
        )

    std = float(pnl.std(ddof=1)) if n_sims > 1 else 0.0
    mean = float(pnl.mean())
    sharpe = mean / std if std > 0 else float("nan")

    return HedgeResult(
        pnl=pnl,
        mean=mean,
        std=std,
        p05=float(np.percentile(pnl, 5)),
        p95=float(np.percentile(pnl, 95)),
        sharpe=sharpe,
        n_sims=n_sims,
        n_steps=n_steps,
        sigma_implied=sigma_implied,
        sigma_realized=sigma_realized,
    )


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    # SPY-like ATM 6-month call.
    S0, K, T = 700.0, 700.0, 0.5
    r, q = 0.05, 0.013
    sigma = 0.18

    print("=== Sanity: sigma_realized == sigma_implied, many rebalances ===")
    res = simulate_delta_hedge(
        S0=S0, K=K, T=T, r=r, sigma_implied=sigma, sigma_realized=sigma,
        q=q, option_type="call",
        n_steps=252, n_sims=2000, seed=42,
    )
    print(f"  n_steps={res.n_steps}, n_sims={res.n_sims}")
    print(f"  PnL mean   = {res.mean:+.4f}")
    print(f"  PnL std    = {res.std:.4f}")
    print(f"  PnL [p05, p95] = [{res.p05:+.3f}, {res.p95:+.3f}]")
    print("  (mean should be ~0; residual std is discretization error)")

    print()
    print("=== Stress: realized vol > implied vol (short gamma loses) ===")
    res_hi = simulate_delta_hedge(
        S0=S0, K=K, T=T, r=r,
        sigma_implied=sigma, sigma_realized=sigma * 1.4,
        q=q, option_type="call",
        n_steps=252, n_sims=2000, seed=42,
    )
    print(f"  sigma_imp={sigma:.2f}, sigma_real={sigma*1.4:.2f}")
    print(f"  PnL mean   = {res_hi.mean:+.4f}  (expect negative)")
    print(f"  PnL std    = {res_hi.std:.4f}")

    print()
    print("=== Rebalance-frequency scaling (sigma_real == sigma_imp) ===")
    for n in [5, 21, 63, 252]:
        r_n = simulate_delta_hedge(
            S0=S0, K=K, T=T, r=r, sigma_implied=sigma, sigma_realized=sigma,
            q=q, option_type="call",
            n_steps=n, n_sims=1000, seed=1,
        )
        print(f"  n_steps={n:4d}:  std = {r_n.std:.4f}   (should shrink ~1/sqrt(n))")

    # --- Plot: PnL histograms, matched vs stressed -----------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "delta_hedge_pnl.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(res.pnl,    bins=50, alpha=0.6, label=f"σ_r = σ_i = {sigma:.2f}")
    ax.hist(res_hi.pnl, bins=50, alpha=0.6, label=f"σ_r = {sigma*1.4:.2f}  >  σ_i = {sigma:.2f}")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Hedged P&L at expiry ($)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Short-call Δ-hedge P&L  (K={K}, T={T}y, {res.n_steps} rebalances, {res.n_sims} paths)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved: {out_path}")
