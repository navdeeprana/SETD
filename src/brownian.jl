function wiener_increment!(dW, s)
    randn!(dW)
    return @. dW *= s
end

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
    W = zeros(N + 1, nens)
    dW = zeros(N)
    for e in 1:nens
        wiener_increment!(dW, sqrth)
        W[1, e] = 0.0
        cumsum!(@view(W[2:end, e]), dW)
    end
    return t, W
end

function coarsegrain(t, W, hcoarse)
    skip = (length(t) - 1) ÷ round(Int, t[end] / hcoarse)
    @views tn, Wn = t[1:skip:end], W[1:skip:end, :]
    return tn, Wn
end

abstract type AbstractWienerIncrement end

struct InstantWienerIncrement{T} <: AbstractWienerIncrement
    h::T
    sqrth::T
    InstantWienerIncrement(h::T) where {T} = new{T}(h, sqrt(h))
end

InstantWienerIncrement(h::T, tmax::T) where {T} = InstantWienerIncrement(h)

struct ComputedWienerIncrement{T, V} <: AbstractWienerIncrement
    h::T
    sqrth::T
    W::V
    ComputedWienerIncrement(h::T, W::V) where {T, V} = new{T, V}(h, sqrt(h), W)
end

function ComputedWienerIncrement(h::T, tmax::T) where {T}
    _, W = wiener_process(h, tmax)
    return ComputedWienerIncrement(h, W)
end

function forced_statistics!(Z, μ, σ)
    μc, σc = mean(Z), std(Z)
    return @. Z = (σ / σc) * (Z - μc) + μ
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

function redraw!(dW::SampledWienerIncrement; force_statistics = false)
    wiener_increment!(dW.dW, dW.sqrth)
    return if force_statistics
        forced_statistics!(dW.dW, 0.0, dW.sqrth)
    end
end

mutable struct SO15WienerIncrement{T, V1, V2} <: AbstractWienerIncrement
    h::T
    sqrth::T
    const dW::V1
    const I10::V2
    SO15WienerIncrement(h::T, W::V1, I10::V2) where {T, V1, V2} = new{T, V1, V2}(h, sqrt(h), W, I10)
end

function SO15WienerIncrement(h::T, tmax::T) where {T}
    N = round(Int, tmax / h)
    dW = wiener_increment(N, h)
    I10 = randn(T, N)
    @. I10 = 0.5 * h^(3 / 2) * (dW + I10 / sqrt(3))
    return SO15WienerIncrement(h, dW, I10)
end

function redraw!(dW::SO15WienerIncrement)
    (; h, dW, I10) = dW
    wiener_increment!(dW, h)
    randn!(I10)
    return @. I10 = 0.5 * h^(3 / 2) * (dW + I10 / sqrt(3))
end

Base.getindex(dW::SampledWienerIncrement, i) = dW.dW[i]
Base.getindex(dW::SO15WienerIncrement, i) = (dW.dW[i], dW.I10[i])
Base.getindex(dW::ComputedWienerIncrement, i) = dW.W[i + 1] - dW.W[i]
Base.getindex(dW::InstantWienerIncrement, i) = dW.sqrth * randn()

wiener_process(dW::SampledWienerIncrement) = wiener_process(dW.dW)

brownian_motion(args...; kwargs...) = wiener_process(args...; kwargs...)

# Compute the stochastic integral I_(1,0) defined as
# I_(1,0) = \int_{t}^{t+h} [\int_{t}^{s} d W(z)] ds
function integral_I10!(I10, t, W, hcoarse)
    skip = (length(t) - 1) ÷ round(Int, t[end] / hcoarse)
    δ = t[2] - t[1]
    return @inbounds for n in 1:length(I10)
        i = (n - 1) * skip + 1
        W0 = W[i]
        s = 0.0
        for k in 1:skip
            s += W[i + k] - W0
        end
        I10[n] = δ * s
    end
end

function integral_I10(t, W, hcoarse)
    N = round(Int, t[end] / hcoarse)
    I10 = similar(W, N)
    integral_I10!(I10, t, W, hcoarse)
    return I10
end
