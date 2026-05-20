# A simple interface for defining integrators for SODEs.
# For a method/algorithm M, we need to define a constructor for the integrator
# and a stepforward function.
# For example, for Euler Maruyama method we have
#       struct EulerMaruyama <: AbstractNumericalMethod end
#       EulerMaruyama(h) = Integrator(EulerMaruyama(), (; h))
#       function stepforward(::EulerMaruyama, q, s::SDE, u0, dW)
#           return u0 + q.h * s.f(u0, s.p) + s.g(u0, s.p) * dW
#       end

abstract type AbstractNumericalMethod end

# Simple explicit methods
struct EulerMaruyama <: AbstractNumericalMethod end
struct Milstein <: AbstractNumericalMethod end
struct StrongOrder15 <: AbstractNumericalMethod end
struct WeakOrder20 <: AbstractNumericalMethod end
struct WeakOrder30 <: AbstractNumericalMethod end

mutable struct ABMaruyama{T} <: AbstractNumericalMethod
    first_step::Bool
    fprev::T
end

# Stochastic Exponential Time Differening (SETD) methods
struct SETDEulerMaruyama <: AbstractNumericalMethod end
struct SETDMilstein <: AbstractNumericalMethod end
struct SETD1 <: AbstractNumericalMethod end

mutable struct SETD2{T} <: AbstractNumericalMethod
    first_step::Bool
    fprev::T
end

# Integrating Factor (IF) methods
struct IFEulerMaruyama <: AbstractNumericalMethod end

# Integrator
struct Integrator{M <: AbstractNumericalMethod, Q}
    m::M # Numerical method
    q::Q # Contains fixed parameters for the integration method.
end

# Constructors for integrators

# Simple explicit methods
EulerMaruyama(h) = Integrator(EulerMaruyama(), (; h))

Milstein(h) = Integrator(Milstein(), (; h))

StrongOrder15(h) = Integrator(StrongOrder15(), (; h, f1 = 0.5 * h, f2 = (0.5 / sqrt(3)) * h^(3 / 2)))
WeakOrder20(h) = Integrator(WeakOrder20(), (; h))
WeakOrder30(h) = Integrator(WeakOrder30(), (; h))

ABMaruyama(h) = Integrator(ABMaruyama(true, 0.0), (; h))

# SETD methods

# For left-point approximation, approx = 1.0 and for mid-point approximation, approx = 0.5.
# By default we choose mid-point approximation as it is more accurate.

function SETDEulerMaruyama(h, c, approx = 0.5)
    fac = (exp(c * h), expm1(c * h) / c, exp(c * h * approx))
    return Integrator(SETDEulerMaruyama(), (; h, c, fac))
end

function SETDMilstein(h, c, approx = 0.5)
    fac = (exp(c * h), expm1(c * h) / c, exp(c * h * approx))
    return Integrator(SETDMilstein(), (; h, c, fac))
end

# Since these integrators operate on Wiener increments, and not directly on the random
# numbers, we need to scale the stochastic integral by sqrt(h). Technically SETD1 should
# only operate on InstantWienerIncrement, but we leave it this way.

function setd1_factors(h, c; threshold = 1.0e-5)
    f1 = exp(h * c)
    if (abs(h * c) <= threshold)
        f2 = h + 0.5e0 * c * h^2
        f3 = 1.0 + 0.5e0 * c * h
    else
        f2 = expm1(c * h) / c
        f3 = sqrt(expm1(2 * c * h) / (2c)) / sqrt(h)
    end
    return f1, f2, f3
end

function SETD1(h, c; threshold = 1.0e-5)
    fac = setd1_factors(h, c; threshold)
    return Integrator(SETD1(), (; h, c, fac))
end

function setd2_factors(h, c; threshold = 1.0e-5)
    f1 = exp(h * c)
    if (abs(h * c) <= threshold)
        f2 = 1.5e0 * h + (2.0e0 / 3.0e0) * c * h^2
        f3 = -0.5e0 * h - (1.0e0 / 6.0e0) * c * h^2
        f4 = 1.0 + 0.5e0 * c * h
    else
        f2 = ((1 + c * h) * expm1(c * h) - c * h) / (c^2 * h)
        f3 = -(expm1(c * h) - c * h) / (c^2 * h)
        f4 = sqrt(expm1(2 * h * c) / (2c)) / sqrt(h)
    end
    return f1, f2, f3, f4
end

function SETD2(h, c; threshold = 1.0e-5)
    fac = setd2_factors(h, c; threshold)
    return Integrator(SETD2(true, 0.0), (; h, c, fac))
end

# Integrating Factor (IF) methods
function IFEulerMaruyama(h, c)
    fac = (
        exp(c * h),
        sqrt(expm1(2 * h * c) / (2c)) / sqrt(h),
    )
    return Integrator(IFEulerMaruyama(), (; h, c, fac))
end

reset!(m::AbstractNumericalMethod) = nothing

function reset!(m::Union{ABMaruyama, SETD2})
    m.first_step = true
    m.fprev = 0.0
    return nothing
end

abstract type AbstractSDE end

# Define a multiplicative SDE du = f(u,p) + g(u,p) dW, where dg = ∂g/∂u.
struct MultiplicativeSDE{F, G, DG, P} <: AbstractSDE
    f::F
    g::G
    dg::DG
    p::P
end

# Define an additive SDE du = f(u,p) + σ dW, where df = ∂f/∂u and d2f = ∂^2f/∂u^2.
struct AdditiveSDE{F, DF, D2F, D3F, D4F, S, P} <: AbstractSDE
    f::F
    df::DF
    d2f::D2F
    d3f::D3F
    d4f::D4F
    σ::S
    p::P
end

stepforward(int::Integrator, s::AbstractSDE, u0, dW) = stepforward(int.m, int.q, s, u0, dW)

# Methods for additive SDEs.

function stepforward(::EulerMaruyama, q, s::AdditiveSDE, u0, dW)
    return u0 + q.h * s.f(u0, s.p) + s.σ * dW
end

function stepforward(m::StrongOrder15, q, s::AdditiveSDE, u0, dW)
    (; p, f, df, d2f, σ) = s
    h, a, da, d2a = q.h, f(u0, p), df(u0, p), d2f(u0, p)
    return (
        u0 + h * a + σ * (dW[1] + da * dW[2])
            + 0.5 * h^2 * (a * da + 0.5 * σ^2 * d2a)
    )
end

function stepforward(m::WeakOrder20, q, s::AdditiveSDE, u0, dW)
    (; p, f, df, d2f, σ) = s
    h, a, da, d2a = q.h, f(u0, p), df(u0, p), d2f(u0, p)
    return (
        u0 + h * a + σ * (1 + 0.5 * h * da) * dW
            + 0.5 * h^2 * (a * da + 0.5 * σ^2 * d2a)
    )
end

function stepforward(m::WeakOrder30, q, s::AdditiveSDE, u0, dW)
    (; p, f, df, d2f, d3f, d4f, σ) = s
    h, a, da, d2a, d3a, d4a = q.h, f(u0, p), df(u0, p), d2f(u0, p), d3f(u0, p), d4f(u0, p)
    return (
        u0 + h * a + σ * dW[1] + σ * da * dW[2]
            + 0.5 * h^2 * (a * da + 0.5 * σ^2 * d2a)
            + (h^2 / 6) * σ * (da^2 + 2 * a * d2a + σ^2 * d3a) * dW[1]
            + (h / 6) * σ^2 * d2a * (dW[1]^2 - h)
            + (h^3 / 6) * (a * da^2 + a^2 * d2a + σ^2 * (1.5 * da * d2a + a * d3a) + 0.25 * σ^4 * d4a)
    )
end

function stepforward(int::ABMaruyama, q, s::AdditiveSDE, u0, dW)
    (; p, f, σ) = s
    h, fnow = q.h, f(u0, p)
    if int.first_step
        un = u0 + h * fnow + σ * dW
        int.first_step = false
    else
        un = u0 + h * (1.5 * fnow - 0.5 * int.fprev) + σ * dW
    end
    int.fprev = fnow
    return un
end

# Methods for multiplicative SDEs

function stepforward(::EulerMaruyama, q, s::MultiplicativeSDE, u0, dW)
    return u0 + q.h * s.f(u0, s.p) + s.g(u0, s.p) * dW
end

function stepforward(::Milstein, q, s::MultiplicativeSDE, u0, dW)
    return (
        u0 + q.h * s.f(u0, s.p)
            + s.g(u0, s.p) * (dW + 0.5 * s.dg(u0, s.p) * (dW^2 - q.h))
    )
end

# SETD Methods

# SETD Methods for additive SDEs

function stepforward(::SETDEulerMaruyama, q, s::AdditiveSDE, u0, dW)
    (; p, f, σ) = s
    return q.fac[1] * u0 + q.fac[2] * f(u0, p) + q.fac[3] * σ * dW
end

function stepforward(::SETD1, q, s::AdditiveSDE, u0, dW)
    (; p, f, σ) = s
    return q.fac[1] * u0 + q.fac[2] * f(u0, p) + q.fac[3] * σ * dW
end

function stepforward(int::SETD2, q, s::AdditiveSDE, u0, dW)
    (; p, f, σ) = s
    fnow = f(u0, s.p)
    if int.first_step
        fac1 = setd1_factors(q.h, q.c)
        un = fac1[1] * u0 + fac1[2] * fnow + fac1[3] * σ * dW
        int.first_step = false
    else
        un = q.fac[1] * u0 + q.fac[2] * fnow + q.fac[3] * int.fprev + q.fac[4] * σ * dW
    end
    int.fprev = fnow
    return un
end

function stepforward(::IFEulerMaruyama, q, s::AdditiveSDE, u0, dW)
    (; p, f, σ) = s
    return q.fac[1] * (u0 + q.h * f(u0, p)) + q.fac[2] * σ * dW
end

# SETD Methods for multiplicative SDEs.

function stepforward(::SETDEulerMaruyama, q, s::MultiplicativeSDE, u0, dW)
    (; p, f, g) = s
    return q.fac[1] * u0 + q.fac[2] * f(u0, p) + q.fac[3] * g(u0, p) * dW
end

function stepforward(::SETDMilstein, q, s::MultiplicativeSDE, u0, dW)
    (; p, f, g, dg) = s
    return (
        q.fac[1] * u0
            + q.fac[2] * f(u0, p)
            + q.fac[3] * g(u0, p) * (dW + 0.5 * dg(u0, p) * (dW^2 - q.h))
    )
end

function solve(s::AbstractSDE, int::Integrator, dW::AbstractWienerIncrement, u0, tmax, saveat; save_after = 0.0)
    (; h) = int.q
    niters, nsave = @. round(Int, (tmax, saveat) / h)
    sol = (; t = Float64[], u = Float64[])
    if save_after == 0.0
        push!(sol.t, 0.0)
        push!(sol.u, u0)
    end
    ui = u0
    for i in 1:niters
        dWi = dW[i]
        ui = stepforward(int, s, ui, dWi)
        if (mod(i, nsave) == 0) && (i * h >= save_after)
            push!(sol.t, i * h)
            push!(sol.u, ui)
        end
    end
    return sol
end
