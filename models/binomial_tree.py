"""
models/binomial_tree.py

Cox-Ross-Rubinstein (CRR) binomial tree for European and American options
with continuous dividend yield.

Lattice construction
--------------------
Over N steps of size dt = T / N:

    u = exp( sigma * sqrt(dt) )              up factor
    d = 1 / u                                 down factor
    p = ( exp((r - q) * dt) - d ) / (u - d)   risk-neutral up probability

The p formula uses the (r - q) drift so the tree is arbitrage-free under the
dividend-adjusted measure — this is what lets us price the same contract that
bs_model.py prices, and is what makes N→∞ convergence to the BS price hold.

A no-arbitrage sanity check requires d < exp((r - q) * dt) < u.  For small
enough dt this is automatic given sigma > 0; we assert it at build time to
catch pathological inputs (e.g. very large r - q relative to sigma).

Backward induction
------------------
Terminal payoff at step N:
    V_N(j) = payoff( S * u^j * d^(N-j) )        for j = 0..N

Rollback:
    continuation = exp(-r * dt) * ( p * V_{n+1}(j+1) + (1 - p) * V_{n+1}(j) )
    V_n(j) = continuation                      (European)
    V_n(j) = max(continuation, payoff(S_n(j))) (American — early exercise)

We walk the tree with a single length-(N+1) array, overwriting in place from
step N back to step 0.  This is O(N) memory and O(N^2) time.

Greeks
------
Delta and Gamma are read off the step-2 nodes directly — no re-pricing needed.
This is the standard "tree Greeks" trick and is essentially free once the
tree has been built.  Theta uses the step-2 middle node (same S, time +2*dt)
minus the root price, which is the usual lattice theta approximation.
"""

from __future__ import annotations

import numpy as np


# ── Pricing ───────────────────────────────────────────────────────────────────

def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    N: int = 200,
    option_type: str = "call",
    american: bool = False,
) -> float:
    """
    Price a European or American option on a CRR binomial tree.

    Parameters
    ----------
    S, K, T, r, sigma, q : float
        Standard BS inputs.
    N : int
        Number of time steps.  Accuracy ~ O(1/N); 200 is a good default for
        equity options, push to 1000+ if you need tight agreement with BS.
    option_type : str
        'call' or 'put'.
    american : bool
        If True, allow early exercise at every node.

    Returns
    -------
    float
        Option price at t = 0.
    """
    price, _tree_data = _crr_backward(S, K, T, r, sigma, q, N, option_type, american,
                                      return_tree_data=False)
    return price


# ── Greeks from the tree ──────────────────────────────────────────────────────

def crr_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    N: int = 200,
    option_type: str = "call",
    american: bool = False,
) -> dict:
    """
    Return price + (delta, gamma, theta) extracted from the tree's first two
    time steps.  Vega and rho are computed by bump-and-reprice (no closed-form
    lattice expression).

    Keys: price, delta, gamma, theta, vega, rho
    """
    # Need N >= 2 to read delta/gamma off the step-2 nodes
    N = max(N, 2)

    price, td = _crr_backward(S, K, T, r, sigma, q, N, option_type, american,
                              return_tree_data=True)

    u, d = td["u"], td["d"]
    V2 = td["V_step2"]            # values at time 2*dt, indexed j=0..2
    dt = T / N

    S_uu = S * u * u
    S_ud = S                       # u * d == 1
    S_dd = S * d * d

    # Central difference in S, using the two outer step-2 nodes
    delta = (V2[2] - V2[0]) / (S_uu - S_dd)

    # Second-order difference using all three step-2 nodes, then divided by
    # spacings.  This is the standard lattice gamma estimator.
    gamma_num = ((V2[2] - V2[1]) / (S_uu - S_ud)) - ((V2[1] - V2[0]) / (S_ud - S_dd))
    gamma = gamma_num / (0.5 * (S_uu - S_dd))

    # Theta: V at (S, t=2*dt) minus V at (S, t=0), divided by 2*dt.  Returns
    # dV/dt (negative for long options, matching bs_theta convention).
    theta = (V2[1] - price) / (2.0 * dt)

    # Vega and rho: bump-and-reprice.  Bump sizes chosen to keep the forward
    # difference numerically stable at N=200 (tree discretization noise is ~1e-3).
    h_sigma = 1e-3
    vega = (
        crr_price(S, K, T, r, sigma + h_sigma, q, N, option_type, american)
        - crr_price(S, K, T, r, sigma - h_sigma, q, N, option_type, american)
    ) / (2 * h_sigma)

    h_r = 1e-4
    rho = (
        crr_price(S, K, T, r + h_r, sigma, q, N, option_type, american)
        - crr_price(S, K, T, r - h_r, sigma, q, N, option_type, american)
    ) / (2 * h_r)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega":  vega,
        "rho":   rho,
    }


# ── Early-exercise premium ────────────────────────────────────────────────────

def early_exercise_premium(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, N: int = 200, option_type: str = "put",
) -> float:
    """
    American price minus European price on the same tree.  For non-dividend
    calls this is ~0 (never optimal to exercise early); for puts and for
    dividend-paying calls it can be meaningfully positive.
    """
    am = crr_price(S, K, T, r, sigma, q, N, option_type, american=True)
    eu = crr_price(S, K, T, r, sigma, q, N, option_type, american=False)
    return am - eu


# ── Core backward induction ───────────────────────────────────────────────────

def _crr_backward(
    S: float, K: float, T: float, r: float, sigma: float, q: float,
    N: int, option_type: str, american: bool,
    return_tree_data: bool,
):
    """
    Run the CRR backward induction.  Returns (price, tree_data) where
    tree_data carries u, d, and the value array at step 2 when
    return_tree_data=True (needed for lattice Greeks).
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")

    # T == 0: return intrinsic, matching bs_price convention
    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return float(intrinsic), {"u": 1.0, "d": 1.0, "V_step2": None}

    dt = T / N
    u = float(np.exp(sigma * np.sqrt(dt)))
    d = 1.0 / u
    disc = float(np.exp(-r * dt))
    p = (np.exp((r - q) * dt) - d) / (u - d)

    if not (0.0 < p < 1.0):
        raise ValueError(
            f"No-arbitrage violation: p = {p:.6f} outside (0, 1). "
            "Increase N or check inputs (r, q, sigma)."
        )

    # Terminal asset prices at step N.  S_N(j) = S * u^j * d^(N-j) for j=0..N.
    j = np.arange(N + 1)
    S_N = S * (u ** j) * (d ** (N - j))

    # Terminal payoff
    if option_type == "call":
        V = np.maximum(S_N - K, 0.0)
    else:
        V = np.maximum(K - S_N, 0.0)

    V_step2 = None  # will hold values at step 2 for lattice Greeks

    # Roll back from step N-1 down to step 0
    for n in range(N - 1, -1, -1):
        # Continuation value at each of the n+1 nodes at step n
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])

        if american:
            j_n = np.arange(n + 1)
            S_n = S * (u ** j_n) * (d ** (n - j_n))
            if option_type == "call":
                intrinsic = np.maximum(S_n - K, 0.0)
            else:
                intrinsic = np.maximum(K - S_n, 0.0)
            V = np.maximum(V, intrinsic)

        if return_tree_data and n == 2:
            V_step2 = V.copy()

    price = float(V[0])
    tree_data = {"u": u, "d": d, "V_step2": V_step2} if return_tree_data else None
    return price, tree_data


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models.bs_model import bs_price, bs_greeks

    print("=== CRR vs Black-Scholes (European call, should converge) ===")
    S, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.05, 0.013, 0.22
    bs = bs_price(S, K, T, r, sigma, q, "call")
    for N in (50, 200, 1000, 5000):
        crr = crr_price(S, K, T, r, sigma, q, N, "call", american=False)
        print(f"  N={N:5d}  CRR={crr:.6f}  BS={bs:.6f}  diff={crr - bs:+.2e}")

    print()
    print("=== European put: CRR vs BS ===")
    bs_p = bs_price(S, K, T, r, sigma, q, "put")
    crr_p = crr_price(S, K, T, r, sigma, q, 1000, "put", american=False)
    print(f"  CRR={crr_p:.6f}  BS={bs_p:.6f}  diff={crr_p - bs_p:+.2e}")

    print()
    print("=== American put early-exercise premium (should be > 0) ===")
    eep = early_exercise_premium(S, K, T, r, sigma, q, N=500, option_type="put")
    print(f"  Premium: {eep:.6f}  (American > European for puts)")

    print()
    print("=== American call on non-dividend stock (premium ~ 0) ===")
    eep_call_noq = early_exercise_premium(S, K, T, r, sigma, q=0.0, N=500,
                                          option_type="call")
    print(f"  Premium (q=0): {eep_call_noq:.2e}  (should be ~0)")

    print()
    print("=== Tree Greeks vs BS Greeks ===")
    g_crr = crr_greeks(S, K, T, r, sigma, q, N=1000, option_type="call",
                       american=False)
    g_bs  = bs_greeks(S, K, T, r, sigma, q, "call")
    for key in ("price", "delta", "gamma", "theta", "vega", "rho"):
        print(f"  {key:6s}: CRR={g_crr[key]:+.6f}  BS={g_bs[key]:+.6f}  "
              f"diff={g_crr[key] - g_bs[key]:+.2e}")
