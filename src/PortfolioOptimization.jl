module PortfolioOptimization

include("data.jl")
include("optimize.jl")
include("costs.jl")
include("backtest.jl")
include("visualize.jl")

export fetch_prices, load_or_fetch
export markowitz_weights, black_litterman_weights, risk_parity_weights
export trade_costs, slippage_rate, CostModel
export run_backtest, run_all_scenarios, run_buy_hold, summary_table, BacktestResult
export make_all_plots

end # module
