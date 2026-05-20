using Random, StatsBase

function wiener_increment!(dW, sqrth)
    randn!(dW)
    @. dW *= sqrth
    return nothing
end

function wiener_increment(N, sqrth)
    dW = randn(N)
    @. dW *= sqrth
    return dW
end

function wiener_process!(W::AbstractVector, dW::AbstractVector)
    W[1] = 0
    cumsum!((@views W[2:end]), dW)
    return nothing
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

# Sample an ensemble of size nens of Brownian motions from [0, tmax] with a time step h.
# Returns time and a Vector with all the Wiener processes.
function wiener_process(h, tmax, nens)
    N, sqrth = round(Int, tmax / h), sqrt(h)
    t = h .* (0:1:N)
    dW = zeros(N)

    W = Vector{Float64}[]
    for e in 1:nens
        wiener_increment!(dW, sqrth)
        push!(W, wiener_process(dW))
    end
    return t, W
end

function coarsegrain(t, W::Vector{T}, h) where {T <: Real}
    skip = round(Int, h / (t[2] - t[1]))
    @views Wh = W[1:skip:end]
    return Wh
end

abstract type AbstractWienerIncrement end

struct InstantWienerIncrement{T} <: AbstractWienerIncrement
    h::T
    sqrth::T
    InstantWienerIncrement(h::T) where {T} = new{T}(h, sqrt(h))
end

InstantWienerIncrement(h::T, tmax::T) where {T} = InstantWienerIncrement(h)

Base.getindex(dW::InstantWienerIncrement, i) = dW.sqrth * randn()

function forced_statistics!(Z, μ, σ)
    μc, σc = mean(Z), std(Z)
    @. Z = (σ / σc) * (Z - μc) + μ
    return nothing
end

struct SampledWienerIncrement{T, V} <: AbstractWienerIncrement
    h::T
    sqrth::T
    dW::V
    SampledWienerIncrement(h::T, dW::V) where {T, V} = new{T, V}(h, sqrt(h), dW)
end

function SampledWienerIncrement(h::T, tmax::T; force_statistics = false) where {T}
    N = round(Int, tmax / h)
    dW = wiener_increment(N, sqrt(h))
    if force_statistics
        forced_statistics!(dW, 0.0, sqrt(h))
    end
    return SampledWienerIncrement(h, dW)
end

Base.getindex(dW::SampledWienerIncrement, i) = dW.dW[i]

function resample!(dW::SampledWienerIncrement; force_statistics = false)
    wiener_increment!(dW.dW, dW.sqrth)
    if force_statistics
        forced_statistics!(dW.dW, 0.0, dW.sqrth)
    end
    return nothing
end

function resample!(dW::SampledWienerIncrement, t, W, h)
    Wn = coarsegrain(t, W, h)
    @inbounds for i in eachindex(dW.dW)
        dW.dW[i] = Wn[i + 1] - Wn[i]
    end
    return nothing
end

mutable struct TwoPointWienerIncrement{T, V1, V2} <: AbstractWienerIncrement
    h::T
    sqrth::T
    const dW::V1
    const I10::V2
    TwoPointWienerIncrement(h::T, W::V1, I10::V2) where {T, V1, V2} = new{T, V1, V2}(h, sqrt(h), W, I10)
end

function TwoPointWienerIncrement(h::T, tmax::T) where {T}
    N = round(Int, tmax / h)
    dW = wiener_increment(N, sqrt(h))
    I10 = randn(T, N)
    @. I10 = 0.5 * h^(3 / 2) * (dW + I10 / sqrt(3))
    return TwoPointWienerIncrement(h, dW, I10)
end

Base.getindex(dW::TwoPointWienerIncrement, i) = (dW.dW[i], dW.I10[i])

function resample!(dW::TwoPointWienerIncrement)
    (; h, sqrth, dW, I10) = dW
    wiener_increment!(dW, sqrth)
    randn!(I10)
    @. I10 = 0.5 * h^(3 / 2) * (dW / sqrth + I10 / sqrt(3))
    return nothing
end

function resample!(dW::TwoPointWienerIncrement, t, W, h)
    Wn = coarsegrain(t, W, h)
    @inbounds for i in eachindex(dW.dW)
        dW.dW[i] = Wn[i + 1] - Wn[i]
    end
    integral_I10!(dW.I10, t, W, h)
    return nothing
end

wiener_process(dW::SampledWienerIncrement) = wiener_process(dW.dW)

brownian_motion(args...; kwargs...) = wiener_process(args...; kwargs...)

# Compute the stochastic integral I_(1,0) defined as
# I_(1,0) = \int_{t}^{t+h} [\int_{t}^{s} d W(z)] ds
function integral_I10!(I10, t, W, h)
    δ = t[2] - t[1]
    skip = round(Int, h / δ)
    @inbounds for n in eachindex(I10)
        i = (n - 1) * skip + 1
        W0 = W[i]
        s = 0.0
        for k in 1:skip
            s += W[i + k] - W0
        end
        I10[n] = δ * s
    end
    return nothing
end

function integral_I10(t, W, h)
    N = round(Int, t[end] / h)
    I10 = similar(W, N)
    integral_I10!(I10, t, W, h)
    return I10
end

@inline function three_point_random_number(h)
    r = rand()
    if r < 2 / 3
        return zero(h)
    elseif r < 5 / 6
        return +sqrt(3h)
    else
        return -sqrt(3h)
    end
end
