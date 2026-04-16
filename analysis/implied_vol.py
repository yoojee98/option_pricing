"""
analysis/implied_vol.py

Implied volatility extraction from market option prices using Brent's method.

Given an observed market price, we invert the Black-Scholes formula numerically
to find the sigma that makes BS price == market price.

Key design choices:
- Uses scipy.optimize.brentq: guaranteed convergence on a bracketed interval,
  no initial guess needed, more robust than Newton-Raphson near flat vega regions.
- Vectorized entry point (compute_iv_surface) operates row-wise on a DataFrame
  and stores results back into an 'iv' column, suitable for vol surface fitting.
- Intrinsic value check: if market_price < intrinsic, IV is undefined (arbitrage);
  we return NaN rather than raising.
- Bounds: sigma in [1e-4, 5.0] covers any realistic equity option scenario.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from tqdm import tqdm

from models.bs_model import bs_price


# ── Constants ─────────────────────────────────────────────────────────────────

SIGMA_LOW  = 1e-4   # lower bound for Brent search
SIGMA_HIGH = 5.0    # upper bound  (500% vol)
BRENT_TOL  = 1e-8   # price tolerance for convergence
MAX_ITER   = 200


# ── Core solver ───────────────────────────────────────────────────────────────

def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    sigma_low: float = SIGMA_LOW,
    sigma_high: float = SIGMA_HIGH,
    tol: float = BRENT_TOL,
) -> float:
    """
    Compute implied volatility for a single option via Brent's method.

    Parameters
    ----------
    market_price : float
        Observed mid price of the option.
    S, K, T, r, q : float
        Standard BS inputs (spot, strike, TTM in years, risk-free rate,
        dividend yield).
    option_type : str
        'call' or 'put'.
    sigma_low, sigma_high : float
        Search bracket for volatility.
    tol : float
        Convergence tolerance on the price residual.

    Returns
    -------
    float
        Implied volatility, or np.nan if:
        - market_price is below intrinsic value (no-arb violation)
        - Brent's method fails to bracket a root (market price outside
          the BS price range achievable within [sigma_low, sigma_high])
    """
    option_type = option_type.lower()

    # Intrinsic value check — market price below intrinsic is arbitrage
    if option_type == "call":
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    if market_price < intrinsic - tol:
        return np.nan

    # Objective: BS price minus market price
    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, q, option_type) - market_price

    # Check bracket: f(sigma_low) and f(sigma_high) must have opposite signs
    try:
        f_low  = objective(sigma_low)
        f_high = objective(sigma_high)
    except Exception:
        return np.nan

    if f_low * f_high > 0:
        # Market price is outside the achievable BS range — return NaN
        return np.nan

    try:
        iv = brentq(objective, sigma_low, sigma_high, xtol=tol, maxiter=MAX_ITER)
        return float(iv)
    except ValueError:
        return np.nan


# ── Vectorized surface computation ────────────────────────────────────────────

def compute_iv_surface(
    df: pd.DataFrame,
    price_col: str = "market_price",
    iv_col: str = "iv",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Compute implied volatility for every row in a filtered option DataFrame
    (as returned by data_loader.load_option_data).

    Expected columns: market_price (or price_col), spot, strike, ttm, r, q,
                      option_type.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered option chain.
    price_col : str
        Column name for the market price to invert.
    iv_col : str
        Output column name for the computed IV.
    show_progress : bool
        Display a tqdm progress bar (useful for chains with thousands of rows).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional `iv_col` column.
        Rows where IV cannot be computed receive NaN.
    """
    df = df.copy()

    ivs = np.empty(len(df), dtype=float)
    ivs[:] = np.nan

    iterator = df.itertuples(index=False)
    if show_progress:
        iterator = tqdm(iterator, total=len(df), desc="Computing IV", unit="opt")

    for i, row in enumerate(iterator):
        ivs[i] = implied_vol(
            market_price=getattr(row, price_col),
            S=row.spot,
            K=row.strike,
            T=row.ttm,
            r=row.r,
            q=row.q,
            option_type=row.option_type,
        )

    df[iv_col] = ivs
    return df


def iv_summary(df: pd.DataFrame, iv_col: str = "iv") -> None:
    """Print a quick diagnostic on IV computation results."""
    total   = len(df)
    valid   = df[iv_col].notna().sum()
    missing = total - valid
    print(f"IV computed: {valid}/{total}  ({missing} NaN)")
    if valid > 0:
        desc = df[iv_col].dropna().describe()
        print(f"  min={desc['min']:.4f}  median={desc['50%']:.4f}  "
              f"max={desc['max']:.4f}  std={desc['std']:.4f}")


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    print("=== Single-contract IV tests ===")

    # Round-trip test: price a BS call, then recover the sigma
    S, K, T, r, q, sigma_true = 100.0, 100.0, 0.5, 0.05, 0.013, 0.20
    price = bs_price(S, K, T, r, sigma_true, q, "call")
    iv_recovered = implied_vol(price, S, K, T, r, q, "call")
    print(f"  True sigma:      {sigma_true:.6f}")
    print(f"  BS price:        {price:.6f}")
    print(f"  Recovered IV:    {iv_recovered:.6f}")
    print(f"  Round-trip error:{abs(iv_recovered - sigma_true):.2e}")

    print()

    # Put round-trip
    price_put = bs_price(S, K, T, r, sigma_true, q, "put")
    iv_put = implied_vol(price_put, S, K, T, r, q, "put")
    print(f"  Put IV recovered: {iv_put:.6f}  (error: {abs(iv_put - sigma_true):.2e})")

    print()

    # Edge cases
    print("=== Edge cases ===")
    print(f"  Below intrinsic (call): {implied_vol(0.001, 100, 110, 0.5, 0.05):.4f}")  # deep OTM, price too low
    print(f"  T→0 deep ITM:           {implied_vol(10.0, 110, 100, 1e-6, 0.05):.4f}")

    print()

    # DataFrame round-trip using data_loader
    try:
        from data_loader import load_option_data
        print("=== Live data IV surface ===")
        df = load_option_data(use_cache=True)
        df_iv = compute_iv_surface(df)
        iv_summary(df_iv)

        # Compare against yfinance IV where available
        both = df_iv.dropna(subset=["iv", "iv_yf"])
        if len(both) > 0:
            corr = both["iv"].corr(both["iv_yf"])
            mae  = (both["iv"] - both["iv_yf"]).abs().mean()
            print(f"\n  vs yfinance IV  —  corr={corr:.4f}  MAE={mae:.4f}")
    except Exception as e:
        print(f"  (Skipping live data test: {e})")
