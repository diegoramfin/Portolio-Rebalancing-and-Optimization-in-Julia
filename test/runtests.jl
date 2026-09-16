using PortfolioOptimization
using Test, LinearAlgebra, Statistics, Dates, Random, DataFrames

Random.seed!(42)

# Synthetic price data: 6 assets, 3 years of geometric random walk
function synthetic_prices(T=800, n=6)
    rets = 0.0003 .+ 0.01 .* randn(T, n)
    rets[:, n] .= 0.0001 .+ 0.003 .* randn(T)   # bond-like last column
    prices = 100 .* cumprod(1 .+ rets; dims=1)
    dates = collect(Date(2021, 1, 1):Day(1):Date(2021, 1, 1) + Day(T - 1))
    return prices, dates
end

@testset "Optimizers" begin
    Random.seed!(1)
    n = 5
    A = randn(200, n) .* 0.01
    Σ = cov(A)
    μ = vec(mean(A; dims=1)) .* 252

    for (name, w) in [
        ("markowitz", markowitz_weights(μ, Σ)),
        ("black_litterman", black_litterman_weights(Σ, fill(1 / n, n))),
        ("risk_parity", risk_parity_weights(Σ)),
    ]
        @testset "$name" begin
            @test length(w) == n
            @test sum(w) ≈ 1.0 atol = 1e-6
            @test all(>=(0), w)
        end
    end

    @testset "risk parity equal risk contribution" begin
        w = risk_parity_weights(Σ)
        rc = w .* (Σ * w)
        @test maximum(rc) - minimum(rc) < 0.15 * mean(rc)
    end

    @testset "black-litterman tilts toward view" begin
        P = reshape([1.0, 0, 0, 0, 0], 1, n)
        w_no = black_litterman_weights(Σ, fill(1 / n, n))
        w_v = black_litterman_weights(Σ, fill(1 / n, n);
                                      P=P, Q=[0.5], Ω=fill(0.01, 1, 1))
        @test w_v[1] > w_no[1]
    end
end

@testset "Cost model" begin
    m = CostModel()
    total, slip, fees, tax = trade_costs(m, [10_000.0, -5_000.0], [500.0])
    @test slip > 0 && fees > 0 && tax ≈ 100.0
    @test total ≈ slip + fees + tax
    # slippage scales nonlinearly with size
    @test slippage_rate(m, 1_000_000.0) > slippage_rate(m, 1_000.0)
    # zero trades → zero cost
    @test trade_costs(m, Float64[])[1] == 0.0
end

@testset "Backtest engine" begin
    prices, dates = synthetic_prices()
    stock_idx = collect(1:5); bond_idx = [6]

    r = run_backtest(prices, dates, stock_idx, bond_idx;
                     allocation=(0.6, 0.4), method=:markowitz,
                     frequency=:monthly, capital=100_000.0)
    @test length(r.values) == length(r.dates)
    @test all(>(0), r.values)
    @test r.total_costs > 0
    @test -1 < r.max_drawdown <= 0
    @test 0 <= r.win_rate <= 1

    # daily rebalancing costs more than monthly
    rd = run_backtest(prices, dates, stock_idx, bond_idx;
                      method=:markowitz, frequency=:daily, capital=100_000.0)
    rm = run_backtest(prices, dates, stock_idx, bond_idx;
                      method=:markowitz, frequency=:monthly, capital=100_000.0)
    @test rd.total_costs > rm.total_costs

    # larger portfolio → larger absolute costs
    rl = run_backtest(prices, dates, stock_idx, bond_idx;
                      method=:markowitz, frequency=:monthly, capital=1_000_000.0)
    @test rl.total_costs > rm.total_costs

    # buy-and-hold baseline runs and pays cost only once
    bh = run_buy_hold(prices, dates, stock_idx, bond_idx)
    @test bh.total_costs > 0
    @test bh.total_costs < rm.total_costs

    # class_weights respects long-only, sum-to-1
    rets = prices[2:end, :] ./ prices[1:end-1, :] .- 1.0
    μ = vec(mean(rets; dims=1)) .* 252
    Σ = cov(rets) .* 252
    w = PortfolioOptimization.class_weights(:risk_parity, μ, Σ,
                                            stock_idx, bond_idx, (0.6, 0.4))
    @test sum(w) ≈ 1.0 atol = 1e-6
    @test all(>=(0), w)
end

@testset "Summary table" begin
    prices, dates = synthetic_prices(600)
    res = run_all_scenarios(prices, dates, collect(1:5), [6];
                            allocations=[(0.6, 0.4)],
                            methods=[:markowitz],
                            frequencies=[:weekly, :monthly],
                            sizes=[10_000.0])
    tbl = summary_table(res)
    # 1 alloc × 1 method × 2 freqs × 1 size + 1 baseline
    @test nrow(tbl) == 3
    @test all(in.(tbl.method, Ref([:markowitz, :buy_hold])))
end
