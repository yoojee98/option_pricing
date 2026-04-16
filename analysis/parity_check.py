"""
analysis/parity_check.py

Put-Call Parity arbitrage detection for European options.

For European options with continuous dividend yield, put-call parity states:

    C - P = S * e^{-qT} - K * e^{-rT}

Any significant deviation from this identity signals either:
  1. A data quality issue (stale quotes, wide spreads, mismatched strikes/expiries)
  2. A genuine arbitrage opportunity (rare in liquid markets like SPY)

This module:
  - Pairs calls and puts sharing the same (expiry, strike)
  - Computes the parity residual and a tolerance-adjusted flag
  - Summarizes violations and exports a clean DataFrame for further analysis
"""

import numpy as np
import pandas as pd


# ── Tolerance ─────────────────────────────────────────────────────────────────

# Flag a pair as a parity violation if |residual| exceeds this fraction of S.
# 0.5% of spot is a reasonable threshold for liquid ETF options after bid-ask
# spread costs; tighten to 0.2% for research-grade filtering.
DEFAULT_TOL_PCT = 0.005   # 0.5% of spot


# ── Core parity logic ─────────────────────────────────────────────────────────

def compute_parity(
    df: pd.DataFrame,
    price_col: str = "market_price",
    tol_pct: float = DEFAULT_TOL_PCT,
) -> pd.DataFrame:
    """
    Pair calls and puts on the same (expiry, strike) and compute parity residuals.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered option chain from data_loader (must contain option_type, expiry,
        strike, ttm, spot, r, q, and price_col columns).
    price_col : str
        Column to use as the observed option price.
    tol_pct : float
        Violation threshold as a fraction of spot price.

    Returns
    -------
    pd.DataFrame with columns:
        expiry, strike, ttm, spot, r, q,
        call_price, put_price,
        parity_lhs   — C - P  (observed)
        parity_rhs   — S*e^{-qT} - K*e^{-rT}  (theoretical)
        residual     — lhs - rhs
        residual_pct — residual / spot
        violation    — bool, |residual_pct| > tol_pct
    """
    calls = (
        df[df["option_type"] == "call"]
        [["expiry", "strike", "ttm", "spot", "r", "q", price_col]]
        .rename(columns={price_col: "call_price"})
    )
    puts = (
        df[df["option_type"] == "put"]
        [["expiry", "strike", price_col]]
        .rename(columns={price_col: "put_price"})
    )

    paired = calls.merge(puts, on=["expiry", "strike"], how="inner")

    if paired.empty:
        return paired

    paired["parity_lhs"] = paired["call_price"] - paired["put_price"]
    paired["parity_rhs"] = (
        paired["spot"] * np.exp(-paired["q"] * paired["ttm"])
        - paired["strike"] * np.exp(-paired["r"] * paired["ttm"])
    )
    paired["residual"]     = paired["parity_lhs"] - paired["parity_rhs"]
    paired["residual_pct"] = paired["residual"] / paired["spot"]
    paired["violation"]    = paired["residual_pct"].abs() > tol_pct

    return paired.reset_index(drop=True)


# ── Summary helpers ───────────────────────────────────────────────────────────

def parity_summary(pairs: pd.DataFrame) -> None:
    """Print a concise summary of parity check results."""
    total    = len(pairs)
    if total == 0 or "violation" not in pairs.columns:
        print("Put-Call Parity Check\n  No matched pairs found.")
        return
    n_viol    = pairs["violation"].sum()
    pct_viol  = 100 * n_viol / total if total > 0 else 0.0

    print(f"Put-Call Parity Check")
    print(f"  Pairs checked : {total}")
    print(f"  Violations    : {n_viol}  ({pct_viol:.1f}%)")
    if total > 0:
        res = pairs["residual_pct"]
        print(f"  Residual (% of spot):")
        print(f"    mean = {res.mean():.4%}")
        print(f"    std  = {res.std():.4%}")
        print(f"    max  = {res.max():.4%}  (most positive)")
        print(f"    min  = {res.min():.4%}  (most negative)")


def get_violations(pairs: pd.DataFrame) -> pd.DataFrame:
    """Return only the pairs that violate parity, sorted by |residual|."""
    if pairs.empty or "violation" not in pairs.columns:
        return pairs
    viols = pairs[pairs["violation"]].copy()
    viols["abs_residual_pct"] = viols["residual_pct"].abs()
    return viols.sort_values("abs_residual_pct", ascending=False).drop(
        columns="abs_residual_pct"
    )


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    print("=== Synthetic parity test (should show 0 violations) ===")
    from models.bs_model import bs_price

    S, r, q, T = 100.0, 0.05, 0.013, 0.5
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    sigma = 0.20

    rows = []
    for K in strikes:
        for opt in ("call", "put"):
            rows.append({
                "expiry": "2026-10-16",
                "strike": K,
                "ttm": T,
                "spot": S,
                "r": r,
                "q": q,
                "option_type": opt,
                "market_price": bs_price(S, K, T, r, sigma, q, opt),
            })
    synthetic = pd.DataFrame(rows)
    pairs = compute_parity(synthetic)
    parity_summary(pairs)

    print()
    print("=== Synthetic test with injected violation ===")
    synthetic.loc[
        (synthetic["option_type"] == "call") & (synthetic["strike"] == 100.0),
        "market_price"
    ] += 2.0   # artificially inflate one call price
    pairs_dirty = compute_parity(synthetic)
    parity_summary(pairs_dirty)
    viols = get_violations(pairs_dirty)
    if not viols.empty:
        print("\n  Violations:")
        print(viols[["expiry", "strike", "call_price", "put_price",
                      "residual", "residual_pct"]].to_string(index=False))

    print()

    # Live data test
    try:
        from data_loader import load_option_data
        print("=== Live SPY data parity check ===")
        df = load_option_data(use_cache=True)
        pairs_live = compute_parity(df)
        parity_summary(pairs_live)
        viols_live = get_violations(pairs_live)
        if not viols_live.empty:
            print("\n  Top violations:")
            print(viols_live[["expiry", "strike", "call_price", "put_price",
                               "residual_pct"]].head(5).to_string(index=False))
    except Exception as e:
        print(f"  (Skipping live data test: {e})")
