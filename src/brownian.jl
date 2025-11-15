function wiener_increment(N, s)
    dW = randn(N)
    @. dW *= s
    return dW
end

wiener_process(dW::AbstractVector) = pushfirst!(cumsum(dW), 0.0)

# Sample a Brownian motion or a Wiener process from [0, tmax] with a time step Δt.
# Returns time and the Wiener process.
function wiener_process(h, tmax)
    N = round(Int, tmax / h)
    t = h .* (0:1:N)
    W = wiener_process(wiener_increment(N, sqrt(h)))
    return t, W
end

# Sample an ensemble of size nens of Brownian motions from [0, tmax] with a time step Δt.
# Returns time and a dataframe with all the Wiener processes.
function wiener_process(h, tmax, nens)
    N, sqrth = round(Int, tmax / h), sqrt(h)
    t = h .* (0:1:N)
    df = DataFrame([Symbol("W$e") => wiener_process(wiener_increment(N, sqrth)) for e in 1:nens])
    return t, df
end

function tnWn(t, W, twhen)
    skip = (length(t) - 1) ÷ round(Int, t[end] / twhen)
    @views tn, Wn = t[1:skip:end], W[1:skip:end, :]
    return tn, Wn
end

abstract type AbstractWienerIncrement end

struct InstantWienerIncrement{T} <: AbstractWienerIncrement
    h::T
    sqrth::T
end

InstantWienerIncrement(h::T) where {T} = InstantWienerIncrement{T}(h, sqrt(h))

struct ComputedWienerIncrement{T} <: AbstractWienerIncrement
    h::T
    sqrth::T
    W::Vector{T}
end

function ComputedWienerIncrement(h::T, tmax::T) where {T}
    t, W = wiener_process(h, tmax)
    ComputedWienerIncrement{T}(h, sqrt(h), W)
end

struct SampledWienerIncrement{T} <: AbstractWienerIncrement
    h::T
    sqrth::T
    dW::Vector{T}
end

mutable struct FixedWienerIncrement{T} <: AbstractWienerIncrement
    h::T
    sqrth::T
    const R::Vector{T}
end

function SampledWienerIncrement(h::T, tmax::T) where {T}
    N = round(Int, tmax/h)
    SampledWienerIncrement{T}(h, sqrt(h), wiener_increment(N, sqrt(h)))
end

function FixedWienerIncrement(h::T, tmax::T; force_statistics = false) where {T}
    N = round(Int, tmax/h)
    R = wiener_increment(N, 1.0)
    if force_statistics
        μ, σ = mean(R), std(R)
        @. R = (R - μ) / σ
    end
    FixedWienerIncrement{T}(h, sqrt(h), R)
end

Base.getindex(dW::SampledWienerIncrement, i) = dW.dW[i]
Base.getindex(dW::FixedWienerIncrement, i) = dW.sqrth * dW.R[i]
Base.getindex(dW::InstantWienerIncrement, i) = dW.sqrth * randn()
Base.getindex(dW::ComputedWienerIncrement, i) = dW.W[i+1] - dW.W[i]

wiener_process(dW::SampledWienerIncrement) = wiener_process(dW.dW)

brownian_motion(args...; kwargs...) = wiener_process(args...; kwargs...)
