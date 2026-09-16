# Backtest engine: daily reoptimization, scheduled rebalancing,
# realistic costs, buy-and-hold baseline.
using DataFrames, Statistics, LinearAlgebra, Dates

const TRADING_DAYS = 252
const LOOKBACK = 252          # estimation window for μ, Σ
const FREQ_DAYS = Dict(:daily => 1, :weekly => 5, :biweekly => 10, :monthly => 21)

struct BacktestResult
    allocation::Tuple{Float64,Float64}
    method::Symbol
    frequency::Symbol
    size::Float64
    values::Vector{Float64}
    dates::Vector{Date}
    total_costs::Float64
    final_return::Float64
    sharpe::Float64
    max_drawdown::Float64
    win_rate::Float64
end

"""Every-`step`-th simulated day is a rebalance date (day 1 always is)."""
is_rebalance_day(sim_day::Int, freq::Symbol) =
    (sim_day - 1) % FREQ_DAYS[freq] == 0

"""
    class_weights(method, μ, Σ, stock_idx, bond_idx, target, ctx)

Two-stage weights: optimize within each class, scale by `target`
(stock_weight, bond_weight). For :risk_parity the class split itself is
optimized on the 2-asset (stocks, bonds) covariance; the `target` is then
ignored for the split but still used for the baseline.
"""
function class_weights(method::Symbol, μ, Σ, stock_idx, bond_idx,
                       target::Tuple{Float64,Float64}; ctx=NamedTuple())
    n = length(μ)
    w = zeros(n)
    if method === :risk_parity
        # within-class stock weights via risk parity on the stock block
        ws = risk_parity_weights(Σ[stock_idx, stock_idx])
        wb = length(bond_idx) == 1 ? [1.0] :
             risk_parity_weights(Σ[bond_idx, bond_idx])
        # class-level split: risk parity on 2 assets (stock sub-port, bonds)
        Σs = (ws' * Σ[stock_idx, stock_idx] * ws)[1]
        Σb = (wb' * Σ[bond_idx, bond_idx] * wb)[1]
        Σsb = (ws' * Σ[stock_idx, bond_idx] * wb)[1]
        split = risk_parity_weights([Σs Σsb; Σsb Σb])
        w[stock_idx] .= split[1] .* ws
        w[bond_idx]  .= split[2] .* wb
    else
        ws = optimize_weights(method, μ[stock_idx], Σ[stock_idx, stock_idx]; ctx)
        wb = length(bond_idx) == 1 ? [1.0] :
             optimize_weights(method, μ[bond_idx], Σ[bond_idx, bond_idx]; ctx)
        w[stock_idx] .= target[1] .* ws
        w[bond_idx]  .= target[2] .* wb
    end
    return w ./ sum(w)
end

"""
    daily_weight_path(prices, stock_idx, bond_idx; allocation, method, ctx)

Daily reoptimization: target weights for every simulated day, estimated
from the trailing LOOKBACK window. Returns a sim_T × n matrix.
"""
function daily_weight_path(prices::Matrix{Float64},
                           stock_idx::Vector{Int}, bond_idx::Vector{Int};
                           allocation::Tuple{Float64,Float64}=(0.6, 0.4),
                           method::Symbol=:markowitz,
                           ctx=NamedTuple())
    T, n = size(prices)
    rets = prices[2:end, :] ./ prices[1:end-1, :] .- 1.0
    sim_T = T - LOOKBACK
    W = Matrix{Float64}(undef, sim_T, n)
    for t in 1:sim_T
        i = LOOKBACK + t
        win = rets[i-LOOKBACK:i-1, :]
        μ = vec(mean(win; dims=1)) .* TRADING_DAYS
        Σ = cov(win) .* TRADING_DAYS
        W[t, :] = class_weights(method, μ, Σ, stock_idx, bond_idx,
                                allocation; ctx)
    end
    return W
end

"""
    run_backtest(prices, dates, stock_idx, bond_idx; allocation, method,
                 frequency, capital, cost_model, ctx, weight_path)

Simulate daily against a precomputed daily weight path (or compute one
when `weight_path` is not given). Trades execute only on rebalance dates.
Sells realize gains taxed at `cost_model.tax_rate`.
"""
function run_backtest(prices::Matrix{Float64}, dates::Vector{Date},
                      stock_idx::Vector{Int}, bond_idx::Vector{Int};
                      allocation::Tuple{Float64,Float64}=(0.6, 0.4),
                      method::Symbol=:markowitz,
                      frequency::Symbol=:monthly,
                      capital::Float64=100_000.0,
                      cost_model::CostModel=CostModel(),
                      ctx=NamedTuple(),
                      weight_path::Union{Nothing,Matrix{Float64}}=nothing)
    T, n = size(prices)
    sim_T = T - LOOKBACK
    sim_dates = dates[LOOKBACK+1:end]
    W = weight_path === nothing ?
        daily_weight_path(prices, stock_idx, bond_idx;
                          allocation=allocation, method=method, ctx=ctx) :
        weight_path

    holdings = zeros(n)               # shares
    basis = zeros(n)                  # average cost per share
    cash = capital
    values = Vector{Float64}(undef, sim_T)
    total_costs = 0.0

    for t in 1:sim_T
        i = LOOKBACK + t              # row in `rets` for today's return
        px = vec(prices[i, :])        # close prices, day i
        port_val = cash + dot(holdings, px)
        w_target = vec(W[t, :])

        if is_rebalance_day(t, frequency)
            target_dollars = w_target .* port_val
            cur_dollars = holdings .* px
            deltas = target_dollars .- cur_dollars
            trades = Float64[]
            gains = Float64[]
            for a in 1:n
                d = deltas[a]
                abs(d) < 1e-8 && continue
                push!(trades, d)
                if d < 0  # sell → realize gain vs average basis
                    push!(gains, max(px[a] - basis[a], -Inf) * (-d / px[a]))
                end
            end
            cost, _, _, _ = trade_costs(cost_model, trades, gains)
            # apply trades at close prices
            for a in 1:n
                d = deltas[a]
                abs(d) < 1e-8 && continue
                shares = d / px[a]
                if d > 0  # buy → update average basis
                    tot = holdings[a] + shares
                    basis[a] = tot > 0 ?
                        (basis[a] * holdings[a] + px[a] * shares) / tot : 0.0
                    holdings[a] = tot
                else
                    holdings[a] += shares
                    holdings[a] < 1e-10 && (basis[a] = 0.0; holdings[a] = 0.0)
                end
            end
            cash = port_val - dot(holdings, px) - cost
            total_costs += cost
        end

        values[t] = cash + dot(holdings, px)
    end

    daily_rets = values[2:end] ./ values[1:end-1] .- 1.0
    sharpe = std(daily_rets) > 0 ?
        mean(daily_rets) / std(daily_rets) * sqrt(TRADING_DAYS) : 0.0
    peak = values[1]; mdd = 0.0
    for v in values
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    end
    return BacktestResult(allocation, method, frequency, capital, values,
        sim_dates, total_costs, values[end] / capital - 1, sharpe, mdd,
        mean(>(0), daily_rets))
end

"""Buy-and-hold at `allocation`, no rebalancing, no costs after day 1."""
function run_buy_hold(prices, dates, stock_idx, bond_idx;
                      allocation=(0.6, 0.4), capital=100_000.0,
                      cost_model=CostModel())
    T, n = size(prices)
    sim_T = T - LOOKBACK
    sim_dates = dates[LOOKBACK+1:end]
    w = zeros(n)
    w[stock_idx] .= allocation[1] / length(stock_idx)
    w[bond_idx]  .= allocation[2] / length(bond_idx)
    px0 = vec(prices[LOOKBACK, :])
    holdings = (w .* capital) ./ px0
    cost, _, _, _ = trade_costs(cost_model, w .* capital)
    holdings .*= (1 - cost / capital)
    values = [dot(holdings, vec(prices[LOOKBACK+t, :])) for t in 1:sim_T]
    daily_rets = values[2:end] ./ values[1:end-1] .- 1.0
    sharpe = std(daily_rets) > 0 ?
        mean(daily_rets) / std(daily_rets) * sqrt(TRADING_DAYS) : 0.0
    peak = values[1]; mdd = 0.0
    for v in values
        peak = max(peak, v); mdd = min(mdd, v / peak - 1)
    end
    return BacktestResult(allocation, :buy_hold, :none, capital, values,
        sim_dates, cost, values[end] / capital - 1, sharpe, mdd,
        mean(>(0), daily_rets))
end

"""Run the full PRD scenario sweep; returns Vector{BacktestResult}."""
function run_all_scenarios(prices, dates, stock_idx, bond_idx;
                           allocations=[(0.6,0.4),(0.7,0.3),(0.8,0.2)],
                           methods=[:markowitz,:black_litterman,:risk_parity],
                           frequencies=[:daily,:weekly,:biweekly,:monthly],
                           sizes=[10_000.0,100_000.0,1_000_000.0],
                           cost_model=CostModel(), ctx=NamedTuple())
    results = BacktestResult[]
    # weight paths depend only on (method, allocation) — compute once each
    paths = Dict{Tuple{Tuple{Float64,Float64},Symbol},Matrix{Float64}}()
    for alloc in allocations, m in methods
        paths[(alloc, m)] = daily_weight_path(prices, stock_idx, bond_idx;
            allocation=alloc, method=m, ctx=ctx)
    end
    for alloc in allocations, m in methods, f in frequencies, s in sizes
        push!(results, run_backtest(prices, dates, stock_idx, bond_idx;
            allocation=alloc, method=m, frequency=f, capital=s,
            cost_model=cost_model, weight_path=paths[(alloc, m)]))
    end
    for alloc in allocations, s in sizes
        push!(results, run_buy_hold(prices, dates, stock_idx, bond_idx;
            allocation=alloc, capital=s, cost_model=cost_model))
    end
    return results
end

"""Summary DataFrame per PRD output spec."""
function summary_table(results::Vector{BacktestResult})
    DataFrame(
        allocation = [r.allocation for r in results],
        method = [r.method for r in results],
        frequency = [r.frequency for r in results],
        portfolio_size = [r.size for r in results],
        final_return = [r.final_return for r in results],
        sharpe = [r.sharpe for r in results],
        max_drawdown = [r.max_drawdown for r in results],
        total_costs = [r.total_costs for r in results],
    )
end
