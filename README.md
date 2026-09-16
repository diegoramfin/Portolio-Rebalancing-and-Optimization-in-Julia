# Portfolio Rebalancing Optimization — Julia

Backtesting framework comparing portfolio performance across rebalancing
frequencies, portfolio sizes, allocations, and optimization methods with
realistic transaction costs. Daily reoptimization across all scenarios
showcases Julia's computational efficiency.

## Methodology

- **Data**: daily adjusted closes from Yahoo Finance (cached to `data/`;
  on fetch failure the cache is used and a warning is logged).
- **Universe**: 5 equity ETFs (SPY, QQQ, IWM, EFA, EEM) + 1 bond ETF (AGG).
- **Allocations**: 60/40, 70/30, 80/20 stocks/bonds.
- **Optimizers**:
  - *Markowitz* — long-only mean-variance (max μ′w − γ/2 w′Σw).
  - *Black-Litterman* — equilibrium-implied returns blended with views,
    then mean-variance.
  - *Risk Parity* — equal risk contribution; the stock/bond split is
    optimized on a 2-asset (stocks, bonds) covariance.
- **Rebalancing**: daily, weekly, bi-weekly, monthly. Weights are
  re-estimated *every day* on a trailing 252-day window; trades execute
  only on rebalance dates.
- **Sizes**: $10k, $100k, $1M.
- **Costs**: slippage (5 bps base, scaling as √notional), $1 flat + 1 bp
  fee per trade, 20% tax on realized gains.
- **Baseline**: buy-and-hold at each allocation/size.

## Install

```sh
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Run

```sh
julia --project scripts/run.jl [years] [datadir] [outdir]
```

Outputs land in `output/`: `summary_results.csv` plus PNGs — cumulative
return curves per method × allocation (9 files, frequencies overlaid with
buy-and-hold), cost-vs-return scatter (2×2 grid by frequency; color =
method, marker size = portfolio size), frequency×size Sharpe heatmap
(3×3 grid, one scenario per cell), 3D Sharpe surfaces per method (3
allocations each), and per-method Sharpe and daily-returns histograms.

## Tests

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

## Interpreting results

`summary_results.csv` has one row per scenario: allocation, method,
frequency, size, final return, Sharpe, max drawdown, total costs. Higher
frequency generally improves tracking to target weights but raises costs;
the heatmap and surface plots show where the trade-off breaks even.
