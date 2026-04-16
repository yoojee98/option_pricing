"""
analysis/vol_surface.py

SVI (Stochastic Volatility Inspired) parametric fitting of the implied
volatility surface, slice-by-slice over expiries.

Raw SVI parameterization (Gatheral 2004):

    w(k) = a + b * { rho * (k - m) + sqrt((k - m)^2 + sigma^2) }

where:
    k = log(K / F)              (log-moneyness, using forward F = S e^{(r-q)T})
    w = sigma_imp^2 * T         (total implied variance)
    a       ∈ R                 vertical level
    b       ≥ 0                 ATM slope magnitude
    rho     ∈ (-1, 1)           skew asymmetry
    m       ∈ R                 horizontal shift
    sigma   > 0                 smoothness at the minimum (ATM curvature)

Necessary no-butterfly-arbitrage condition on each slice:
    a + b * sigma * sqrt(1 - rho^2) >= 0        (ensures w(k) >= 0 everywhere)

Calibration — Zeliade "quasi-explicit" two-step approach
--------------------------------------------------------
For fixed (m, sigma), SVI is LINEAR in (a, d = b*rho, c = b):

    w(k) = a + d * (k - m) + c * sqrt((k - m)^2 + sigma^2)

This inner problem is a constrained linear least-squares with:
    c >= 0,   |d| <= c,   a + c * sqrt(1 - (d/c)^2) * sigma >= 0

We solve the inner problem via scipy.optimize.minimize with SLSQP and
a closed-form-style warm start, then optimize (m, sigma) in the outer
loop via Nelder-Mead. This is empirically much more robust than a
direct 5-dimensional non-linear fit, which is notoriously prone to
local minima and parameter drift.

Entry points
------------
    fit_svi_slice(k, w)          → single-expiry fit, returns SVIParams
    fit_svi_surface(df_iv)       → per-expiry fit, returns {expiry: SVIParams}
    build_surface_grid(...)      → dense (k, T, iv) grid for plotting
    plot_vol_surface(...)        → plotly 3D surface
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ── Parameters container ──────────────────────────────────────────────────────

@dataclass
class SVIParams:
    """Calibrated SVI parameters for a single expiry slice."""
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    T: float           # time to maturity in years (stored for convenience)
    rmse: float        # in-sample RMSE on total variance w
    n_points: int      # number of observations used in the fit

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        """Evaluate w(k) for a vector of log-moneyness values."""
        return svi_total_variance(k, self.a, self.b, self.rho, self.m, self.sigma)

    def implied_vol(self, k: np.ndarray) -> np.ndarray:
        """Convert total variance back to annualized implied vol."""
        w = self.total_variance(k)
        return np.sqrt(np.maximum(w, 0.0) / self.T)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Core SVI function ─────────────────────────────────────────────────────────

def svi_total_variance(
    k: np.ndarray,
    a: float, b: float, rho: float, m: float, sigma: float,
) -> np.ndarray:
    """Raw SVI total variance w(k)."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


# ── Inner problem: fit (a, b, rho) for fixed (m, sigma) ───────────────────────

def _fit_adc_given_m_sigma(
    k: np.ndarray,
    w: np.ndarray,
    m: float,
    sigma: float,
) -> tuple[float, float, float, float]:
    """
    For fixed (m, sigma), fit (a, d = b*rho, c = b) by constrained least
    squares. Returns (a, b, rho, rmse_w).

    Constraints:
        c >= 0             (i.e. b >= 0)
        |d| <= c           (i.e. |rho| <= 1)
        a + c * sqrt(1 - (d/c)^2) * sigma >= 0     (w >= 0 everywhere)
    """
    y = k - m
    z = np.sqrt(y ** 2 + sigma ** 2)

    # Design matrix: w ≈ a * 1 + d * y + c * z
    X = np.column_stack([np.ones_like(y), y, z])

    # Unconstrained LS warm start
    adc0, *_ = np.linalg.lstsq(X, w, rcond=None)
    a0, d0, c0 = adc0
    # Project warm start into feasible region (c >= 0, |d| <= c)
    c0 = max(c0, 1e-6)
    d0 = np.clip(d0, -c0 + 1e-6, c0 - 1e-6)
    a0 = max(a0, -c0 * sigma * np.sqrt(max(1 - (d0 / c0) ** 2, 0.0)))

    def obj(adc):
        a, d, c = adc
        pred = a + d * y + c * z
        return float(np.mean((pred - w) ** 2))

    constraints = [
        {"type": "ineq", "fun": lambda x: x[2]},                    # c >= 0
        {"type": "ineq", "fun": lambda x: x[2] - abs(x[1])},        # |d| <= c
        {"type": "ineq",
         "fun": lambda x: x[0] + x[2] * sigma *
                          np.sqrt(max(1.0 - (x[1] / max(x[2], 1e-12)) ** 2, 0.0))},
    ]

    res = minimize(
        obj,
        x0=np.array([a0, d0, c0]),
        method="SLSQP",
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 200},
    )

    a, d, c = res.x
    b = max(c, 0.0)
    rho = 0.0 if b < 1e-12 else np.clip(d / b, -0.999, 0.999)
    rmse = float(np.sqrt(obj(res.x)))
    return a, b, rho, rmse


# ── Outer problem: search over (m, sigma) ─────────────────────────────────────

def fit_svi_slice(
    k: np.ndarray,
    w: np.ndarray,
    T: Optional[float] = None,
) -> SVIParams:
    """
    Fit SVI parameters to a single expiry slice.

    Parameters
    ----------
    k : np.ndarray
        Log-moneyness values log(K / F).
    w : np.ndarray
        Total implied variance (= iv^2 * T) at each k.
    T : float, optional
        Time to maturity. Stored in the returned params; not used in the fit
        itself (since w already embeds T).

    Returns
    -------
    SVIParams
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)

    mask = np.isfinite(k) & np.isfinite(w) & (w > 0)
    k, w = k[mask], w[mask]

    if len(k) < 5:
        raise ValueError(
            f"Not enough valid points to fit SVI slice (got {len(k)}, need >= 5)"
        )

    # Outer bounds for (m, sigma).  sigma_min is the key regularizer:
    # as sigma → 0 the SVI curvature term vanishes and the smile degenerates
    # to piecewise-linear in k, which lets (b, ρ) drift to extreme values
    # (e.g. b in the thousands, |ρ| → 1) that "fit" noisy quotes but give
    # an economically meaningless surface. An absolute floor of 0.01 is
    # well below any realistic smile curvature scale yet large enough to
    # keep the Jacobian well-conditioned.
    k_span = max(float(k.max() - k.min()), 0.05)
    sigma_min = 0.01
    sigma_max = 5.0 * k_span
    m_bound = 2.0 * k_span
    bounds = [(-m_bound, m_bound), (sigma_min, sigma_max)]

    # Penalty on the inner solution to discourage extreme b.  For liquid ETF
    # options b rarely exceeds ~2 in total-variance units; values >> that
    # indicate the degenerate regime even if sigma is above the floor.
    B_SOFT_CAP = 10.0
    LAMBDA_B = 1e-4

    def outer_obj(params):
        m, sigma = params
        _, b, _, rmse = _fit_adc_given_m_sigma(k, w, m, sigma)
        penalty = LAMBDA_B * max(b - B_SOFT_CAP, 0.0) ** 2
        return rmse ** 2 + penalty

    # Multi-start L-BFGS-B — robust to warm-start choice and respects bounds.
    starts = [
        (float(np.mean(k)),   0.5 * k_span),
        (0.0,                 0.2 * k_span),
        (float(np.mean(k)),   0.1 * k_span),
        (float(np.median(k)), 1.0 * k_span),
    ]

    best = None
    for m0, sigma0 in starts:
        x0 = np.array([
            np.clip(m0, -m_bound, m_bound),
            np.clip(sigma0, sigma_min, sigma_max),
        ])
        res = minimize(
            outer_obj,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 500},
        )
        if best is None or res.fun < best.fun:
            best = res

    m_opt, sigma_opt = best.x
    a, b, rho, rmse = _fit_adc_given_m_sigma(k, w, m_opt, sigma_opt)

    return SVIParams(
        a=a, b=b, rho=rho, m=m_opt, sigma=sigma_opt,
        T=T if T is not None else np.nan,
        rmse=rmse,
        n_points=len(k),
    )


# ── Surface fitting across expiries ───────────────────────────────────────────

def fit_svi_surface(
    df: pd.DataFrame,
    iv_col: str = "iv",
    min_points_per_slice: int = 5,
    verbose: bool = True,
) -> Dict[str, SVIParams]:
    """
    Fit SVI independently for each expiry in `df`.

    Expected columns: expiry, ttm, spot, strike, r, q, and iv_col.

    Uses forward-based log-moneyness k = log(K / F), F = S * exp((r - q) * T),
    which is the standard SVI input — this centers the smile at k = 0 ATM-forward.

    Returns
    -------
    dict mapping expiry string → SVIParams
    """
    required = {"expiry", "ttm", "spot", "strike", "r", "q", iv_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fit_svi_surface: missing required columns {missing}")

    df = df.dropna(subset=[iv_col]).copy()
    # Collapse call/put duplicates on same (expiry, strike) by averaging their IVs.
    # Under put-call parity, call IV == put IV in theory; averaging dampens quote noise.
    if "option_type" in df.columns:
        df = (
            df.groupby(["expiry", "strike"], as_index=False)
              .agg({"ttm": "first", "spot": "first", "r": "first", "q": "first",
                    iv_col: "mean"})
        )

    df["forward"] = df["spot"] * np.exp((df["r"] - df["q"]) * df["ttm"])
    df["k"] = np.log(df["strike"] / df["forward"])
    df["w"] = df[iv_col] ** 2 * df["ttm"]

    results: Dict[str, SVIParams] = {}
    for expiry, grp in df.groupby("expiry"):
        if len(grp) < min_points_per_slice:
            if verbose:
                print(f"  [skip] {expiry}: only {len(grp)} points")
            continue
        try:
            params = fit_svi_slice(grp["k"].values, grp["w"].values,
                                   T=float(grp["ttm"].iloc[0]))
            results[str(expiry)] = params
            if verbose:
                print(f"  [ok]   {expiry}: n={params.n_points:3d}  "
                      f"RMSE(w)={params.rmse:.2e}  "
                      f"a={params.a:+.4f} b={params.b:.4f} "
                      f"rho={params.rho:+.3f} m={params.m:+.4f} "
                      f"sigma={params.sigma:.4f}")
        except Exception as e:
            if verbose:
                print(f"  [fail] {expiry}: {e}")

    return results


# ── Dense grid for plotting ───────────────────────────────────────────────────

def build_surface_grid(
    svi_by_expiry: Dict[str, SVIParams],
    k_range: tuple[float, float] = (-0.25, 0.25),
    n_k: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample each fitted SVI slice on a common log-moneyness grid.

    Returns
    -------
    K : (n_T, n_k) log-moneyness grid (same across rows)
    T : (n_T, n_k) time-to-maturity grid (same within rows)
    IV: (n_T, n_k) implied volatility surface

    Slices are sorted by T ascending.
    """
    slices = sorted(svi_by_expiry.values(), key=lambda p: p.T)
    if not slices:
        raise ValueError("build_surface_grid: no fitted slices provided")

    k_grid = np.linspace(k_range[0], k_range[1], n_k)
    T_vals = np.array([s.T for s in slices])

    iv_matrix = np.vstack([s.implied_vol(k_grid) for s in slices])
    K, T = np.meshgrid(k_grid, T_vals)
    return K, T, iv_matrix


# ── Plotly visualization ──────────────────────────────────────────────────────

def plot_vol_surface(
    svi_by_expiry: Dict[str, SVIParams],
    df_iv: Optional[pd.DataFrame] = None,
    iv_col: str = "iv",
    k_range: tuple[float, float] = (-0.25, 0.25),
    n_k: int = 60,
    title: str = "SPY Implied Volatility Surface (SVI-fitted)",
):
    """
    Build an interactive 3D plotly figure of the fitted SVI surface.
    If df_iv is provided, overlay raw market IV points as a scatter.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    K, T, IV = build_surface_grid(svi_by_expiry, k_range=k_range, n_k=n_k)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=K, y=T, z=IV,
        colorscale="Viridis",
        name="SVI fit",
        showscale=True,
        opacity=0.9,
        colorbar=dict(title="IV"),
    ))

    if df_iv is not None and iv_col in df_iv.columns:
        pts = df_iv.dropna(subset=[iv_col]).copy()
        pts["forward"] = pts["spot"] * np.exp((pts["r"] - pts["q"]) * pts["ttm"])
        pts["k"] = np.log(pts["strike"] / pts["forward"])
        fig.add_trace(go.Scatter3d(
            x=pts["k"], y=pts["ttm"], z=pts[iv_col],
            mode="markers",
            marker=dict(size=2, color="red", opacity=0.6),
            name="Market IV",
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Log-moneyness  k = log(K/F)",
            yaxis_title="Time to maturity  T (yrs)",
            zaxis_title="Implied volatility",
        ),
        width=950, height=700,
    )
    return fig


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    print("=== Synthetic round-trip: recover SVI params from generated w(k) ===")
    true_params = dict(a=0.04, b=0.12, rho=-0.4, m=-0.02, sigma=0.10)
    k_true = np.linspace(-0.2, 0.2, 40)
    w_true = svi_total_variance(k_true, **true_params)
    fit = fit_svi_slice(k_true, w_true, T=0.5)
    print(f"  True:       {true_params}")
    print(f"  Recovered:  a={fit.a:+.4f} b={fit.b:.4f} rho={fit.rho:+.3f} "
          f"m={fit.m:+.4f} sigma={fit.sigma:.4f}")
    print(f"  RMSE(w):    {fit.rmse:.2e}  (should be ~0)")

    print()
    print("=== Synthetic with noise ===")
    rng = np.random.default_rng(0)
    w_noisy = w_true + rng.normal(0, 5e-4, size=w_true.shape)
    fit_n = fit_svi_slice(k_true, w_noisy, T=0.5)
    print(f"  Recovered:  a={fit_n.a:+.4f} b={fit_n.b:.4f} rho={fit_n.rho:+.3f} "
          f"m={fit_n.m:+.4f} sigma={fit_n.sigma:.4f}")
    print(f"  RMSE(w):    {fit_n.rmse:.2e}")

    print()

    # Live data surface fit
    try:
        from data_loader import load_option_data
        from analysis.implied_vol import compute_iv_surface

        print("=== Live SPY surface fit ===")
        df = load_option_data(use_cache=True)
        df_iv = compute_iv_surface(df, show_progress=False)
        df_iv = df_iv.dropna(subset=["iv"])
        print(f"  Usable rows after IV computation: {len(df_iv)}")
        print()
        surface = fit_svi_surface(df_iv)
        print(f"\n  Fitted {len(surface)} expiry slices")
    except Exception as e:
        print(f"  (Skipping live data test: {e})")
