# Option Pricing & Volatility Surface Construction

A comprehensive option pricing and risk analysis platform built in Python, using real SPY market data. This project implements classical pricing models, constructs an implied volatility surface via SVI fitting, simulates dynamic Delta hedging, and benchmarks a neural network-based pricing model against classical methods.

---

## Project Structure

```
option_pricing/
├── data/                   # Cached market data (excluded from Git)
├── models/
│   ├── bs_model.py         # Black-Scholes pricing + full Greeks
│   ├── binomial_tree.py    # CRR Binomial Tree (European & American)
│   ├── monte_carlo.py      # Monte Carlo simulation + Antithetic Variates
│   └── exotic_options.py   # Asian & Barrier option pricing
├── analysis/
│   ├── implied_vol.py      # Implied volatility extraction (brentq)
│   ├── vol_surface.py      # SVI parametric fitting + 3D surface
│   ├── delta_hedge.py      # Dynamic Delta hedging simulation
│   └── parity_check.py     # Put-Call Parity arbitrage detection
├── ml/
│   ├── model.py            # MLP model definition (PyTorch)
│   ├── train.py            # Training pipeline + feature engineering
│   └── benchmark.py        # Inference speed: BS vs MC vs MLP
├── notebooks/
│   ├── 01_bs_greeks.ipynb
│   ├── 02_vol_surface.ipynb
│   ├── 03_delta_hedging.ipynb
│   └── 04_ml_pricing.ipynb
├── tests/
│   ├── test_bs_model.py
│   ├── test_binomial_tree.py
│   └── test_implied_vol.py
├── outputs/                # Generated charts (referenced in README)
├── data_loader.py          # Data fetching + filtering + dividend adjustment
├── app.py                  # Streamlit interactive dashboard
└── requirements.txt
```

---

## What This Project Covers

| Module | Description |
|---|---|
| **Black-Scholes** | Analytical pricing for European options with full Greeks (Δ, Γ, ν, Θ, ρ) |
| **Binomial Tree** | CRR numerical model supporting American options with early exercise |
| **Monte Carlo** | GBM path simulation with Antithetic Variates variance reduction |
| **Exotic Options** | Asian (arithmetic average) and Down-and-Out Barrier options |
| **Implied Volatility** | Numerical extraction via Brent's method from real SPY option chain |
| **Volatility Surface** | SVI parametric fitting across strikes and maturities (3D interactive) |
| **Delta Hedging** | Dynamic hedging simulation with Gamma/Vega exposure tracking |
| **ML Pricing** | MLP-based IV prediction with inference speed benchmarking |
| **Dashboard** | Streamlit app for real-time pricing and risk visualization |

---

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the Streamlit dashboard**
```bash
streamlit run app.py
```

**3. Run all notebooks end-to-end**
```bash
jupyter notebook notebooks/
```

**4. Run unit tests**
```bash
pytest tests/ -v
```

---

## Key Results

### Greeks Sensitivity (Black-Scholes)
> *[Insert Greeks sensitivity chart here — outputs/greeks_sensitivity.png]*

### Implied Volatility Surface (SVI Fitted)
> *[Insert 3D volatility surface screenshot here — outputs/vol_surface_3d.png]*

### Delta Hedging P&L Distribution
> *[Insert hedging P&L histogram here — outputs/delta_hedge_pnl.png]*

### Model Comparison

| Model | Price (ATM Call) | Inference Time |
|---|---|---|
| Black-Scholes | — | ~1 μs |
| Binomial Tree (N=200) | — | ~500 μs |
| Monte Carlo (100k paths) | — | ~500,000 μs |
| MLP (inference) | — | ~10 μs |

> *Fill in prices after running the benchmark notebook.*

---

## Data

- **Underlying**: SPY (SPDR S&P 500 ETF Trust)
- **Source**: `yfinance` real market option chain snapshots
- **Filtering applied**:
  - Moneyness: `0.80 < K/S < 1.20`
  - Bid-Ask spread: `(Ask - Bid) / Mid < 15%`
  - Minimum volume: `> 10 contracts`
  - Minimum TTM: `> 7 days`
  - Dividend adjustment: `q ≈ 1.3%` annualized (SPY quarterly dividends)

---

## Technical Notes

- **SVI Fitting**: Uses a two-step calibration approach — polynomial warm-start for initial parameter estimation, followed by constrained global optimization with no-arbitrage bounds.
- **ML Train/Test Split**: Time-ordered split (first 80% train, last 20% test) to prevent data leakage.
- **American Options**: Binomial Tree with backward induction; early exercise premium quantified vs. European equivalent.
- **Variance Reduction**: Antithetic Variates reduce Monte Carlo standard error by ~50% at identical path count.

---

## Requirements

```
numpy==1.26.4
scipy==1.13.0
pandas==2.2.2
matplotlib==3.9.0
plotly==5.22.0
yfinance==0.2.40
torch==2.3.0
scikit-learn==1.5.0
streamlit==1.35.0
pytest==8.2.0
```

---

## License

MIT
