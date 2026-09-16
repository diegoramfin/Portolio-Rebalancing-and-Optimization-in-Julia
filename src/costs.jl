# Transaction cost model: slippage, fees, capital-gains tax.

"""
    CostModel

Parametric trading-cost model.
- `slippage_bps`: base slippage in basis points of trade notional
- `slippage_size_coef`: nonlinear size penalty — slippage grows as
  `bps * (1 + coef * sqrt(notional / ref_size))`
- `ref_size`: notional at which the size penalty is calibrated
- `flat_fee`: flat dollar cost per executed trade
- `fee_bps`: proportional fee in basis points of notional
- `tax_rate`: capital-gains rate applied to realized gains on sells
"""
Base.@kwdef struct CostModel
    slippage_bps::Float64 = 5.0          # 0.05%
    slippage_size_coef::Float64 = 0.5
    ref_size::Float64 = 100_000.0
    flat_fee::Float64 = 1.0
    fee_bps::Float64 = 1.0
    tax_rate::Float64 = 0.20
end

"""
    slippage_rate(model, notional) -> Float64

Slippage fraction for a trade of `notional` dollars; scales nonlinearly
with size relative to `model.ref_size`.
"""
function slippage_rate(m::CostModel, notional::Float64)
    return (m.slippage_bps / 1e4) *
           (1 + m.slippage_size_coef * sqrt(max(notional, 0.0) / m.ref_size))
end

"""
    trade_costs(model, trades, gains) -> (total, slippage, fees, taxes)

`trades`: vector of signed notionals (buy > 0, sell < 0).
`gains`: realized gain per sell trade (same order as sells in `trades`,
or empty). Returns cost components in dollars.
"""
function trade_costs(m::CostModel, trades::AbstractVector{<:Real},
                     realized_gains::AbstractVector{<:Real}=Float64[])
    slip = 0.0
    fees = 0.0
    for t in trades
        n = abs(t)
        n <= 0 && continue
        slip += n * slippage_rate(m, n)
        fees += m.flat_fee + n * m.fee_bps / 1e4
    end
    taxes = m.tax_rate * sum(max(g, 0.0) for g in realized_gains; init=0.0)
    return (slip + fees + taxes, slip, fees, taxes)
end
