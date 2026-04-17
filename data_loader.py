"""
data_loader.py

Fetches, filters, and caches SPY option chain data from yfinance.
Applies standard data quality filters and dividend adjustment.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf

# ── Cache configuration ───────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_FILE = os.path.join(CACHE_DIR, "spy_options.parquet")

# ── Market constants ──────────────────────────────────────────────────────────
TICKER = "SPY"
RISK_FREE_RATE_FALLBACK = 0.04  # Used only if ^IRX fetch fails
DIVIDEND_YIELD = 0.013          # SPY annualized dividend yield (~1.3%)

# ── Option type constants ─────────────────────────────────────────────────────
CALL = "call"
PUT = "put"

# ── Filter thresholds ─────────────────────────────────────────────────────────
MAX_LOG_MONEYNESS = 0.1823  # |log(K/S)| upper bound ≈ log(1.20), symmetric for calls & puts
MAX_SPREAD_RATIO = 0.15    # (Ask - Bid) / Ask  (ask as denominator handles bid=0 quotes)
MIN_LIQUIDITY = 1           # volume >= 1  OR  open_interest >= 1
MIN_TTM_DAYS = 7           # Minimum days to expiration


def _fetch_risk_free_rate() -> float:
    """
    Return the current annualized risk-free rate from the 3-month T-bill yield (^IRX).
    ^IRX is quoted as a percentage (e.g. 3.61), so divide by 100.
    Falls back to RISK_FREE_RATE_FALLBACK if the fetch fails.
    """
    try:
        hist = yf.Ticker("^IRX").history(period="5d")
        if hist.empty:
            raise ValueError("Empty ^IRX history")
        return float(hist["Close"].iloc[-1]) / 100.0
    except Exception:
        print(f"[data_loader] ^IRX fetch failed, using fallback r={RISK_FREE_RATE_FALLBACK:.2%}")
        return RISK_FREE_RATE_FALLBACK


def _fetch_spot_price(ticker: str = TICKER) -> float:
    """Return the current spot price for the given ticker."""
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1d")
    if hist.empty:
        raise ValueError(f"Could not fetch spot price for {ticker}")
    return float(hist["Close"].iloc[-1])


def _fetch_raw_chain(ticker: str = TICKER) -> pd.DataFrame:
    """
    Download the full option chain from yfinance and return a combined
    calls + puts DataFrame with expiry and option_type columns added.
    """
    tk = yf.Ticker(ticker)
    expirations = tk.options

    frames = []
    for exp in expirations:
        try:
            chain = tk.option_chain(exp)
        except Exception:
            continue

        for side, opt_type in [(chain.calls, CALL), (chain.puts, PUT)]:
            tagged = side.assign(expiry=exp, option_type=opt_type)
            frames.append(tagged)

    if not frames:
        raise RuntimeError(f"No option chain data returned for {ticker}")

    return pd.concat(frames, ignore_index=True)


def _apply_filters(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """
    Apply data quality filters. Returns a cleaned DataFrame.
    Filters:
        - Moneyness: MIN_MONEYNESS <= K/S <= MAX_MONEYNESS
        - Bid-Ask spread ratio: (Ask - Bid) / Mid <= MAX_SPREAD_RATIO
        - Volume >= MIN_VOLUME (falls back to open_interest when volume absent)
        - TTM >= MIN_TTM_DAYS (in calendar days)
        - Drop rows with non-positive bid/ask or missing lastPrice
    """
    # Only rename columns that actually differ from the target name
    df = df.rename(columns={
        "lastPrice": "last_price",
        "openInterest": "open_interest",
        "impliedVolatility": "iv_yf",
    })

    required = ["strike", "ask", "last_price"]
    df = df.dropna(subset=required)
    df = df[(df["ask"] > 0) & (df["last_price"] > 0)]
    df = df[df["bid"].fillna(0) <= df["ask"]]  # drop inverted quotes

    df["moneyness"] = df["strike"] / spot
    df["log_moneyness"] = np.log(df["moneyness"])
    df = df[df["log_moneyness"].abs() <= MAX_LOG_MONEYNESS]

    # Use ask as denominator so bid=0 quotes are handled correctly
    df["mid"] = (df["bid"].fillna(0) + df["ask"]) / 2.0
    df["spread_ratio"] = (df["ask"] - df["bid"].fillna(0)) / df["ask"]
    df = df[df["spread_ratio"] <= MAX_SPREAD_RATIO]

    # Accept if volume >= 1 OR open_interest >= 1
    vol = df["volume"].fillna(0)
    oi = df["open_interest"].fillna(0)
    df = df[(vol >= MIN_LIQUIDITY) | (oi >= MIN_LIQUIDITY)]

    # Vectorized TTM: avoid per-row Python calls
    today = pd.Timestamp("today").normalize()
    df["ttm"] = (pd.to_datetime(df["expiry"]) - today).dt.days / 365.0
    df = df[df["ttm"] >= MIN_TTM_DAYS / 365.0]

    return df.reset_index(drop=True)


def _add_derived_fields(df: pd.DataFrame, spot: float, r: float) -> pd.DataFrame:
    """Add model-ready columns: spot, r, q, log_moneyness, market_price."""
    df = df.copy()
    df["spot"] = spot
    df["r"] = r
    df["q"] = DIVIDEND_YIELD
    df["log_moneyness"] = np.log(df["moneyness"])
    df["market_price"] = df["mid"]
    # yfinance fills iv_yf with 0.00001 when it cannot converge; treat as missing
    df.loc[df["iv_yf"] < 0.001, "iv_yf"] = np.nan
    return df


def load_option_data(
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Main entry point. Returns a filtered, enriched SPY option chain DataFrame.

    Parameters
    ----------
    use_cache : bool
        If True and a cache file exists, load from disk instead of fetching.
    force_refresh : bool
        If True, ignore the cache and always fetch fresh data.

    Returns
    -------
    pd.DataFrame with columns:
        strike, bid, ask, last_price, mid, market_price,
        expiry, option_type, ttm, moneyness, log_moneyness,
        spread_ratio, volume (or open_interest),
        spot, r, q, iv_yf
    """
    serve_from_cache = use_cache and not force_refresh and os.path.exists(CACHE_FILE)

    if serve_from_cache:
        print(f"[data_loader] Loading cached data from {CACHE_FILE}")
        return pd.read_parquet(CACHE_FILE)

    print(f"[data_loader] Fetching live option chain for {TICKER}...")
    spot = _fetch_spot_price()
    r = _fetch_risk_free_rate()
    print(f"[data_loader] Spot price: {spot:.2f}  |  Risk-free rate (^IRX): {r:.2%}")

    raw = _fetch_raw_chain()
    print(f"[data_loader] Raw rows fetched: {len(raw)}")

    filtered = _apply_filters(raw, spot)
    print(f"[data_loader] Rows after filtering: {len(filtered)}")

    enriched = _add_derived_fields(filtered, spot, r)

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        enriched.to_parquet(CACHE_FILE, index=False)
        print(f"[data_loader] Saved to cache: {CACHE_FILE}")

    return enriched


# ── Convenience helpers ───────────────────────────────────────────────────────

def _filter_col(df: pd.DataFrame, col: str, val) -> pd.DataFrame:
    return df[df[col] == val].reset_index(drop=True)


def get_calls(df: pd.DataFrame) -> pd.DataFrame:
    return _filter_col(df, "option_type", CALL)


def get_puts(df: pd.DataFrame) -> pd.DataFrame:
    return _filter_col(df, "option_type", PUT)


def get_by_expiry(df: pd.DataFrame, expiry: str) -> pd.DataFrame:
    return _filter_col(df, "expiry", expiry)


def get_expiry_list(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique expiry dates in the dataset."""
    return sorted(df["expiry"].unique())


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # data = load_option_data(use_cache=False)
    data = load_option_data(use_cache=False,force_refresh=True)
    print("\nSample rows:")
    print(data[["option_type", "expiry", "strike", "mid", "ttm", "moneyness", "iv_yf"]].head(10).to_string(index=False))
    print(f"\nTotal: {len(data)} contracts | Expiries: {len(get_expiry_list(data))}")
    print(f"Calls: {len(get_calls(data))} | Puts: {len(get_puts(data))}")
