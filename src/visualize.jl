# Visualization: all PRD charts via Plots.jl (GR backend, headless-safe).
# Every chart is broken out per optimization method — no cross-strategy
# aggregation.
using Plots, DataFrames, Statistics, Printf
gr()

const FREQ_ORDER = [:daily, :weekly, :biweekly, :monthly]
const SIZE_ORDER = [10_000.0, 100_000.0, 1_000_000.0]
const METHOD_ORDER = [:markowitz, :black_litterman, :risk_parity]
const METHOD_LABEL = Dict(:markowitz => "Markowitz",
                          :black_litterman => "Black-Litterman",
                          :risk_parity => "Risk Parity")

_alloc_label(a) = "$(round(Int, a[1]*100))/$(round(Int, a[2]*100))"
_size_label(s) = s >= 1e6 ? "\$$(Int(s/1e6))M" : "\$$(Int(s/1e3))k"
_size_ms(s) = 3 + 2 * log10(s / 1e4)   # marker size encodes portfolio size

"""All PRD plots saved as PNGs into `outdir`. Returns file paths."""
function make_all_plots(results::Vector{BacktestResult}, outdir::AbstractString)
    mkpath(outdir)
    paths = String[]
    managed = filter(r -> r.method !== :buy_hold, results)
    allocs = sort(unique(r.allocation for r in managed))

    # ── 1. Cumulative return curves: one PNG per (method, allocation),
    #       all four frequencies overlaid + buy-and-hold baseline ────────
    for m in METHOD_ORDER, alloc in allocs
        sub = filter(r -> r.method == m && r.allocation == alloc &&
                          r.size == 100_000.0, managed)
        isempty(sub) && continue
        p = plot(title="Cumulative returns — $(METHOD_LABEL[m]), " *
                      "$(_alloc_label(alloc)) stocks/bonds, \$100k",
                 xlabel="Date", ylabel="Growth of \$1",
                 legend=:topleft, size=(900, 550))
        for r in sort(sub, by=r -> findfirst(==(r.frequency), FREQ_ORDER))
            plot!(p, r.dates, r.values ./ r.values[1],
                  label="$(r.frequency) rebalance", linewidth=1.5)
        end
        for r in filter(r -> r.method === :buy_hold &&
                             r.allocation == alloc && r.size == 100_000.0,
                        results)
            plot!(p, r.dates, r.values ./ r.values[1],
                  label="buy & hold", linestyle=:dash, color=:black,
                  linewidth=1.5)
        end
        tag = replace(_alloc_label(alloc), "/" => "-")
        f = joinpath(outdir, "cumret_$(m)_$(tag).png")
        savefig(p, f); push!(paths, f)
    end

    # ── 2. Cost vs return: 2×2 grid by frequency; color = method,
    #       marker size = portfolio size ─────────────────────────────────
    subplots = []
    for (fi, freq) in enumerate(FREQ_ORDER)
        p = scatter(title="$(freq) rebalancing",
                    xlabel="Total transaction costs (\$, log)",
                    ylabel="Final return",
                    xscale=:log10, legend=fi == 1 ? :topright : false)
        for m in METHOD_ORDER, s in SIZE_ORDER
            pts = filter(r -> r.method == m && r.frequency == freq &&
                              r.size == s, managed)
            isempty(pts) && continue
            scatter!(p, [r.total_costs for r in pts],
                     [r.final_return for r in pts],
                     label="$(METHOD_LABEL[m]) $(_size_label(s))",
                     markersize=_size_ms(s))
        end
        push!(subplots, p)
    end
    p = plot(subplots...; layout=(2, 2), size=(1300, 950),
             plot_title="Transaction cost vs. final return " *
                        "(color = method, marker size = portfolio size)")
    f = joinpath(outdir, "cost_vs_return.png"); savefig(p, f); push!(paths, f)

    # ── 3. Heatmap grid: rows = method, cols = allocation; each cell is
    #       frequency × size Sharpe for a single scenario (no pooling) ───
    subplots = []
    for m in METHOD_ORDER, alloc in allocs
        z = [only([r.sharpe for r in managed
                   if r.method == m && r.allocation == alloc &&
                      r.frequency == fr && r.size == s])
             for fr in FREQ_ORDER, s in SIZE_ORDER]
        push!(subplots,
            heatmap(_size_label.(SIZE_ORDER), string.(FREQ_ORDER), z;
                    title="$(METHOD_LABEL[m]) $(_alloc_label(alloc))",
                    xlabel="Size", ylabel="Frequency",
                    color=:viridis, colorbar=true, clims=(-1, 2)))
    end
    p = plot(subplots...; layout=(length(METHOD_ORDER), length(allocs)),
             size=(1500, 1200),
             plot_title="Sharpe ratio — frequency × size, per method and allocation")
    f = joinpath(outdir, "heatmap_freq_size_sharpe.png")
    savefig(p, f); push!(paths, f)

    # ── 4. 3D surface per method: one PNG each, allocations side-by-side ─
    for m in METHOD_ORDER
        subplots = []
        for alloc in allocs
            z = [only([r.sharpe for r in managed
                       if r.method == m && r.allocation == alloc &&
                          r.frequency == fr && r.size == s])
                 for fr in FREQ_ORDER, s in SIZE_ORDER]
            xs = collect(1:4); ys = log10.(SIZE_ORDER)
            push!(subplots,
                surface(xs, ys, z';
                        title=_alloc_label(alloc),
                        xlabel="Frequency", ylabel="Size", zlabel="Sharpe",
                        xticks=(xs, string.(FREQ_ORDER)),
                        yticks=(ys, ["10k", "100k", "1M"]),
                        color=:viridis, colorbar=false,
                        camera=(30, 30)))
        end
        p = plot(subplots...; layout=(1, length(allocs)), size=(1600, 500),
                 plot_title="Sharpe surface — $(METHOD_LABEL[m]) " *
                            "(frequency × size, per allocation)")
        f = joinpath(outdir, "surface_$(m).png"); savefig(p, f); push!(paths, f)
    end

    # ── 5. Sharpe histogram per method (overlaid) ───────────────────────
    p = histogram(xlabel="Sharpe ratio", ylabel="Scenario count",
                  title="Sharpe distribution by method",
                  bins=20, alpha=0.55, legend=:topright, size=(900, 550))
    for m in METHOD_ORDER
        vals = [r.sharpe for r in managed if r.method == m]
        histogram!(p, vals; bins=20, alpha=0.55, label=METHOD_LABEL[m])
    end
    f = joinpath(outdir, "sharpe_hist.png"); savefig(p, f); push!(paths, f)

    # ── 6. Daily returns histogram per method (overlaid, density) ───────
    p = histogram(xlabel="Daily return", ylabel="Density",
                  title="Daily return distribution by method",
                  bins=120, alpha=0.55, normalize=:pdf,
                  legend=:topright, size=(900, 550),
                  xlims=(-0.05, 0.05))
    for m in METHOD_ORDER
        vals = Float64[]
        for r in managed
            r.method == m || continue
            append!(vals, r.values[2:end] ./ r.values[1:end-1] .- 1.0)
        end
        histogram!(p, vals; bins=120, alpha=0.55, normalize=:pdf,
                   label=METHOD_LABEL[m])
    end
    f = joinpath(outdir, "daily_returns_hist.png")
    savefig(p, f); push!(paths, f)

    return paths
end
