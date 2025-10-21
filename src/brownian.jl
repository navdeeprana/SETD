function weiner_increment(N, s)
    dW = randn(N)
    @. dW *= s
    return dW
end

weiner_process(dW::AbstractVector) = pushfirst!(cumsum(dW), 0.0)

# Sample a Brownian motion or a Wiener process from [0, tmax] with a time step Δt.
# Returns time and the Wiener process.
function weiner_process(h, tmax)
    N = round(Int, tmax / h)
    t = h .* (0:1:N)
    W = weiner_process(weiner_increment(N, sqrt(h)))
    return t, W
end

# Sample an ensemble of size nens of Brownian motions from [0, tmax] with a time step Δt.
# Returns time and a dataframe with all the Wiener processes.
function weiner_process(h, tmax, nens)
    N, sqrth = round(Int, tmax / h), sqrt(h)
    t = h .* (0:1:N)
    df = DataFrame([Symbol("W$e") => weiner_process(weiner_increment(N, sqrth)) for e in 1:nens])
    return t, df
end

function tnWn(t, W, twhen)
    skip = (length(t) - 1) ÷ round(Int, t[end] / twhen)
    @views tn, Wn = t[1:skip:end], W[1:skip:end, :]
    return tn, Wn
end

abstract type AbstractWeinerIncrement end

struct InstantWeinerIncrement{T} <: AbstractWeinerIncrement
    h::T
    sqrth::T
end

InstantWeinerIncrement(h::T) where {T} = InstantWeinerIncrement{T}(h, sqrt(h))

struct ComputedWeinerIncrement{T} <: AbstractWeinerIncrement
    h::T
    sqrth::T
    W::Vector{T}
end

function ComputedWeinerIncrement(h::T, tmax::T) where {T}
    t, W = brownian_motion(h, tmax)
    ComputedWeinerIncrement{T}(h, sqrt(h), W)
end

struct SampledWeinerIncrement{T} <: AbstractWeinerIncrement
    h::T
    sqrth::T
    dW::Vector{T}
end

function SampledWeinerIncrement(h::T, tmax::T) where {T}
    N = round(Int, tmax/h)
    SampledWeinerIncrement{T}(h, sqrt(h), weiner_increment(N, sqrt(h)))
end

Base.getindex(dW::SampledWeinerIncrement, i) = dW.dW[i]
Base.getindex(dW::InstantWeinerIncrement, i) = dW.sqrth * randn()
Base.getindex(dW::ComputedWeinerIncrement, i) = dW.W[i+1] - dW.W[i]

weiner_process(dW::SampledWeinerIncrement) = weiner_process(dW.dW)

brownian_motion(args...; kwargs...) = weiner_process(args...; kwargs...)
