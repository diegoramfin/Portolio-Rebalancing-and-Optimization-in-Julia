# Data fetching: Yahoo Finance chart API with cached-CSV fallback.
using HTTP, JSON3, DataFrames, CSV, Dates, Random

const DEFAULT_STOCKS = ["SPY", "QQQ", "IWM", "EFA", "EEM"]
const DEFAULT_BONDS  = ["AGG"]

"""
    fetch_prices(ticker; years=5) -> DataFrame

Fetch daily adjusted-close prices for `ticker` from Yahoo Finance.
Returns a DataFrame with columns `date` and `adjclose`.
"""
function fetch_prices(ticker::AbstractString; years::Int=5)
    p2 = Int(floor(datetime2unix(now(UTC))))
    p1 = Int(floor(datetime2unix(now(UTC) - Year(years) - Day(10))))
    url = "https://query1.finance.yahoo.com/v8/finance/chart/$(ticker)" *
          "?period1=$(p1)&period2=$(p2)&interval=1d&events=div%2Csplit"
    resp = HTTP.get(url; retry=false, readtimeout=30,
                    headers=["User-Agent" => "Mozilla/5.0"])
    js = JSON3.read(resp.body)
    result = js.chart.result[1]
    ts = result.timestamp
    q = result.indicators.quote[1]
    adj = haskey(result.indicators, :adjclose) ?
          result.indicators.adjclose[1].adjclose : q.close
    dates = [Date(unix2datetime(t)) for t in ts]
    df = DataFrame(date=dates, adjclose=[a === nothing ? NaN : Float64(a) for a in adj])
    dropmissing!(filter!(:adjclose => x -> !isnan(x), df))
    return df
end

"""
    load_or_fetch(tickers; years=5, datadir="data") -> DataFrame

Wide DataFrame (date + one column per ticker) of adjusted closes.
Tries live fetch per ticker; on failure falls back to `datadir/<ticker>.csv`,
logging a warning. Successful fetches refresh the cache.
"""
function load_or_fetch(tickers::Vector{String}; years::Int=5,
                       datadir::AbstractString="data")
    mkpath(datadir)
    frames = DataFrame[]
    for t in tickers
        cache = joinpath(datadir, "$(t).csv")
        df = try
            d = fetch_prices(t; years=years)
            CSV.write(cache, d)
            d
        catch err
            if isfile(cache)
                @warn "Live fetch failed for $t ($(typeof(err))); using cached $cache"
                CSV.read(cache, DataFrame)
            else
                @warn "Live fetch failed for $t and no cache exists; rethrowing"
                rethrow()
            end
        end
        df = select(df, :date, :adjclose => Symbol(t))
        push!(frames, df)
    end
    out = reduce((a, b) -> innerjoin(a, b, on=:date), frames)
    sort!(out, :date)
    return out
end
