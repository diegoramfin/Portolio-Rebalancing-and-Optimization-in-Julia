# Portfolio optimizers: Markowitz, Black-Litterman, Risk Parity.
# All return long-only weight vectors summing to 1.
using JuMP, Ipopt, LinearAlgebra, Statistics

const OPT_SOLVER = optimizer_with_attributes(Ipopt.Optimizer,
    "print_level" => 0, "sb" => "yes", "max_iter" => 3000)

function _regularize(Σ::AbstractMatrix)
    Σs = Symmetric(Matrix(Σ))
    # ensure positive semi-definite for the QP
    if !isposdef(Σs)
        Σs = Σs + 1e-8 * I
    end
    return Matrix(Σs)
end

"""
    markowitz_weights(μ, Σ; risk_aversion=1.0) -> Vector{Float64}

Long-only mean-variance optimum: max μ'w - γ/2 w'Σw s.t. Σw=1, w≥0.
"""
function markowitz_weights(μ::AbstractVector, Σ::AbstractMatrix;
                           risk_aversion::Float64=1.0)
    n = length(μ)
    Σr = _regularize(Σ)
    m = Model(OPT_SOLVER)
    @variable(m, w[1:n] >= 0)
    @constraint(m, sum(w) == 1)
    @objective(m, Max, dot(μ, w) - risk_aversion / 2 * dot(w, Σr * w))
    optimize!(m)
    w = value.(w)
    w = max.(w, 0.0)
    return w ./ sum(w)
end

"""
    black_litterman_weights(Σ, mkt_w; τ=0.05, P, Q, Ω) -> Vector{Float64}

Black-Litterman posterior expected returns fed into mean-variance.
`P` (k×n) picks assets per view, `Q` (k) view returns, `Ω` (k×k) view
uncertainty. With no views, reduces to equilibrium Markowitz.
"""
function black_litterman_weights(Σ::AbstractMatrix, mkt_w::AbstractVector;
                                 τ::Float64=0.05,
                                 P::AbstractMatrix=zeros(0, length(mkt_w)),
                                 Q::AbstractVector=Float64[],
                                 Ω::AbstractMatrix=zeros(0, 0),
                                 risk_aversion::Float64=2.5)
    Σr = _regularize(Σ)
    π_eq = risk_aversion .* (Σr * mkt_w)          # implied equilibrium returns
    if isempty(Q)
        μ_bl = π_eq
    else
        τΣ = τ .* Σr
        M = inv(inv(τΣ) + P' * inv(Ω) * P)
        μ_bl = M * (inv(τΣ) * π_eq + P' * inv(Ω) * Q)
    end
    return markowitz_weights(μ_bl, Σr; risk_aversion=risk_aversion)
end

"""
    risk_parity_weights(Σ) -> Vector{Float64}

Equal risk contribution: w_i (Σw)_i identical across i. Long-only, Σw=1.
"""
function risk_parity_weights(Σ::AbstractMatrix)
    n = size(Σ, 1)
    Σr = _regularize(Σ)
    m = Model(OPT_SOLVER)
    @variable(m, w[1:n] >= 1e-8, start = 1 / n)
    # convex ERC program: min w'Σw - Σ log w_i; ∇=0 ⇒ equal risk contributions
    @objective(m, Min, dot(w, Σr * w) - sum(log(w[i]) for i in 1:n))
    optimize!(m)
    w = max.(value.(w), 0.0)
    return w ./ sum(w)
end

"""
    optimize_weights(method, μ, Σ, ctx) -> Vector{Float64}

Dispatch on `method` (:markowitz | :black_litterman | :risk_parity).
`ctx` carries method extras: `mkt_w`, `P`, `Q`, `Ω`, `risk_aversion`.
"""
function optimize_weights(method::Symbol, μ, Σ; ctx=NamedTuple())
    if method === :markowitz
        return markowitz_weights(μ, Σ;
            risk_aversion=get(ctx, :risk_aversion, 1.0))
    elseif method === :black_litterman
        return black_litterman_weights(Σ, get(ctx, :mkt_w, fill(1 / length(μ), length(μ)));
            P=get(ctx, :P, zeros(0, length(μ))),
            Q=get(ctx, :Q, Float64[]),
            Ω=get(ctx, :Ω, zeros(0, 0)),
            risk_aversion=get(ctx, :risk_aversion, 2.5))
    elseif method === :risk_parity
        return risk_parity_weights(Σ)
    else
        throw(ArgumentError("unknown method $method"))
    end
end
