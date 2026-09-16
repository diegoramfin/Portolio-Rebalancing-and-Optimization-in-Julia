#!/usr/bin/env julia
# Main entry point: run the full PRD scenario sweep end-to-end.
# Usage: julia --project scripts/run.jl [years] [datadir] [outdir]
using PortfolioOptimization
using DataFrames, CSV, Printf, Dates

years   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
datadir = length(ARGS) >= 2 ? ARGS[2] : "data"
outdir  = length(ARGS) >= 3 ? ARGS[3] : "output"
mkpath(outdir)

stocks = PortfolioOptimization.DEFAULT_STOCKS
bonds  = PortfolioOptimization.DEFAULT_BONDS
tickers = vcat(stocks, bonds)

println("Loading data for $(join(tickers, ", ")) ($years y)…")
px = load_or_fetch(tickers; years=years, datadir=datadir)
dates = Vector{Date}(px.date)
prices = Matrix(px[:, Not(:date)])
stock_idx = collect(1:length(stocks))
bond_idx  = collect(length(stocks)+1:length(tickers))
println("  $(length(dates)) trading days, $(length(tickers)) assets")

# Black-Litterman context: market-cap-ish prior (equal weights) + one view.
n_s = length(stocks)
ctx = (mkt_w = fill(1 / n_s, n_s),
       P = reshape([i == 1 ? 1.0 : 0.0 for i in 1:n_s], 1, n_s),
       Q = [0.08],                       # view: SPY returns 8%/yr
       Ω = fill(0.02, 1, 1))

println("Running scenario sweep (3 allocations × 3 methods × 4 frequencies × 3 sizes + baselines)…")
t0 = time()
results = run_all_scenarios(prices, dates, stock_idx, bond_idx; ctx=ctx)
@printf("  %d scenarios in %.1fs\n", length(results), time() - t0)

tbl = summary_table(results)
csv_path = joinpath(outdir, "summary_results.csv")
CSV.write(csv_path, tbl)
println("Wrote $csv_path")

println("Rendering plots…")
paths = make_all_plots(results, outdir)
foreach(p -> println("  $p"), paths)

best = sortperm(tbl.sharpe, rev=true)[1]
b = tbl[best, :]
println("\nBest scenario by Sharpe: $(b.method) | $(b.allocation) | " *
        "$(b.frequency) | \$$(Int(b.portfolio_size)) → " *
        "Sharpe $(round(b.sharpe, digits=2)), " *
        "return $(round(100*b.final_return, digits=1))%, " *
        "costs \$$(round(b.total_costs, digits=0))")
