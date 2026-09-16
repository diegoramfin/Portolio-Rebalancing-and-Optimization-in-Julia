# Portfolio Rebalancing Optimization Analysis — Julia Project

## Objective
Build a backtesting framework that compares portfolio performance across different rebalancing frequencies and portfolio sizes, accounting for realistic transaction costs, to identify the optimal rebalancing strategy. Showcase Julia's computational power through daily reoptimization across all scenarios.

---

## Scope

### Data
- **Source:** Free historical price data (Yahoo Finance, Alpha Vantage, or Quandl)
- **Assets:** 5-10 liquid stocks or ETFs, plus bonds
- **Timeframe:** 3-5 years of daily OHLC data
- **Frequency:** Daily candles minimum

### Portfolio Construction
- **Base Allocations:**
  - 60-40 (stocks-bonds)
  - 70-30 (stocks-bonds)
  - 80-20 (stocks-bonds)
- **Optimization Methods:**
  1. Mean-Variance (Markowitz) — maximize return per unit of risk
  2. Black-Litterman — market equilibrium + custom views
  3. Risk Parity — equal risk contribution (stocks & bonds only)
- **Constraints:** Long-only, weights sum to 1

### Rebalancing Frequencies
- Daily
- Weekly
- Bi-weekly
- Monthly

### Portfolio Sizes (to model scaling costs)
- Small: $10,000
- Medium: $100,000
- Large: $1,000,000

### Reoptimization
- **Daily reoptimization** across all scenarios to showcase Julia's computational efficiency

---

## Transaction Cost Model

Implement realistic trading costs that scale with portfolio size and frequency:

- **Slippage:** Percentage of trade size (e.g., 0.05% for small trades, scaling nonlinearly with size)
- **Fees:** Flat per-trade cost + basis points on notional value
- **Taxes:** Capital gains on realized gains (simplified: 20% long-term rate)
- **Cost Function:** Parametric design so costs scale with trade size and rebalancing frequency

---

## Backtest Engine

For each combination of allocation, optimization method, rebalancing frequency, and portfolio size:

1. Calculate optimal weights at each rebalancing date (daily)
2. Simulate trades: record entry price, calculate slippage and fees
3. Deduct costs from portfolio value
4. Track cumulative returns, drawdown, Sharpe ratio, win rate
5. Compare against buy-and-hold baseline (no rebalancing)

---

## Output & Visualization

### Summary Statistics
- Table: allocation, optimization method, frequency, portfolio size, final return, Sharpe ratio, max drawdown, total costs

### Charts
- **Cumulative Return Curves:** One per allocation, showing all rebalancing frequencies overlaid
- **Cost vs. Return Scatter:** X-axis = total transaction costs, Y-axis = final return
- **Heatmap:** Rebalancing frequency vs. portfolio size, color-coded by Sharpe ratio
- **3D Surface:** Show the relationship between frequency, size, and performance (convex, concave, exponential shape)
- **Sharpe Ratio Histogram:** Distribution across all scenarios
- **Daily Returns Histogram:** Distribution across all scenarios

### Write-up
- 1-2 paragraphs on key findings and which strategy (method + frequency + size) performs best

---

## Deliverables

1. **Modular Julia Code**
   - Data fetcher (pull free historical data)
   - Portfolio optimizer (Markowitz, Black-Litterman, Risk Parity)
   - Backtest engine
   - Transaction cost calculator
   - Visualization module

2. **Main Executable Script**
   - Runs all scenarios end-to-end
   - Clean CLI or configuration-driven approach

3. **Output Files**
   - CSV: Summary results table
   - PNG: All plots saved to disk

4. **README**
   - Methodology explanation
   - Dependencies and installation
   - How to run the project
   - Interpretation of results

---

## Notes

- **No Jupyter notebooks** — all professional scripts
- **Julia performance is the story** — daily reoptimization demonstrates computational efficiency
- **Realistic costs matter** — ensure transaction cost model prevents overly optimistic results
- **Risk Parity uses 2 assets** (stocks & bonds only) for simplicity
- **Markowitz and Black-Litterman** use full stock universe (5-10 assets)
