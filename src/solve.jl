abstract type AbstractNumericalMethod end
struct EulerMaruyama <: AbstractNumericalMethod end
struct Milstein <: AbstractNumericalMethod end
struct SETDEulerMaruyama <: AbstractNumericalMethod end
struct SETDMilstein <: AbstractNumericalMethod end
struct SETD1 <: AbstractNumericalMethod end

mutable struct SETD2{T} <: AbstractNumericalMethod
    first_step::Bool
    fprev::T
end

struct Integrator{M<:AbstractNumericalMethod,Q}
    m::M # Integration method
    q::Q # Contains fixed parameters for the integration method.
end

EulerMaruyama(h) = Integrator(EulerMaruyama(), (; h))

Milstein(h) = Integrator(Milstein(), (; h))

# For left-point approximation, approx = 1.0 and for mid-point approximation, approx = 0.5.
# By default we choose mid-point approximation as it is more accurate.

function SETDEulerMaruyama(h, c, approx = 0.5)
    fac = (exp(c*h), expm1(c*h)/c, exp(c*h*approx))
    Integrator(SETDEulerMaruyama(), (; h, c, fac))
end

function SETDMilstein(h, c, approx = 0.5)
    fac = (exp(c*h), expm1(c*h)/c, exp(c*h*approx))
    Integrator(SETDMilstein(), (; h, c, fac))
end

# Since these integrators operate on Wiener increments, and not directly on the random
# numbers, we need to scale the stochastic integral by sqrt(h). Technically SETD1 should
# only operate on InstantWienerIncrement, but we leave it this way.
function SETD1(h, c)
    fac = (
        exp(c*h),
        expm1(c*h)/c,
        sqrt(expm1(2*c*h)/(2c))/sqrt(h)
    )
    Integrator(SETD1(), (; h, c, fac))
end

function SETD2(h, c)
    fac = (
        exp(c*h),
        ((1 + c*h) * expm1(c*h) - c*h) / (c^2*h),
        -(expm1(c*h) - c*h) / (c^2*h),
        sqrt(expm1(2*h*c)/(2c))/sqrt(h)
    )
    Integrator(SETD2(true, 0.0), (; h, c, fac))
end

# Define a SDE du = f(u,p) + g(u,p) dW, where dg  = ∂g/∂u.
struct SDE{F,G,DG,P}
    f::F
    g::G
    dg::DG
    p::P
end

stepforward(int::Integrator, s::SDE, u0, dW) = stepforward(int.m, int.q, s, u0, dW)

function stepforward(::EulerMaruyama, q, s::SDE, u0, dW)
    return u0 + q.h * s.f(u0, s.p) + s.g(u0, s.p) * dW
end

function stepforward(::Milstein, q, s::SDE, u0, dW)
    return (
        u0 + q.h * s.f(u0, s.p)
        + s.g(u0, s.p) * (dW + 0.5 * s.dg(u0, s.p) * (dW^2-q.h))
    )
end

function stepforward(::SETDEulerMaruyama, q, s::SDE, u0, dW)
    return q.fac[1] * u0 + q.fac[2] * s.f(u0, s.p) + q.fac[3] * s.g(u0, s.p) * dW
end

function stepforward(::SETDMilstein, q, s::SDE, u0, dW)
    return (
        q.fac[1] * u0
        + q.fac[2] * s.f(u0, s.p)
        + q.fac[3] * s.g(u0, s.p) * (dW + 0.5 * s.dg(u0, s.p) * (dW^2 - q.h))
    )
end

function stepforward(::SETD1, q, s::SDE, u0, dW)
    return q.fac[1] * u0 + q.fac[2] * s.f(u0, s.p) + q.fac[3] * s.g(u0, s.p) * dW
end

function stepforward(int::SETD2, q, s::SDE, u0, dW)
    fnow = s.f(u0, s.p)
    if int.first_step
        c, h = q.c, q.h
        fac = (exp(c*h), expm1(c*h)/c, sqrt(expm1(2*c*h)/(2c))/sqrt(h))
        un = fac[1] * u0 + fac[2] * fnow + q.fac[3] * s.g(u0, s.p) * dW
        int.first_step = false
    else
        un = q.fac[1] * u0 + q.fac[2] * fnow + q.fac[3] * int.fprev + q.fac[4] * s.g(u0, s.p) * dW
    end
    int.fprev = fnow
    return un
end

function solve(s::SDE, int::Integrator, dW::AbstractWienerIncrement, u0, tmax, saveat; save_after = 0.0)
    (; h) = int.q
    niters, nsave = @. round(Int, (tmax, saveat)/h)
    sol = (; t = Float64[], u = Float64[])
    if save_after == 0.0
        push!(sol.t, 0.0);
        push!(sol.u, u0);
    end
    ui = u0
    for i in 1:niters
        dWi = dW[i]
        ui = stepforward(int, s, ui, dWi)
        if (mod(i, nsave) == 0) && (i*h >= save_after)
            push!(sol.t, i*h)
            push!(sol.u, ui)
        end
    end
    return sol
end

function compute_convergence!(cvg, h, sol, sol_an)
    for (i, ti) in enumerate(sol[1].t)
        ti == 0 ? continue : nothing
        u = [s.u[i] for s in sol]
        uan = [s.u[i] for s in sol_an]
        es = mean(@. abs(u - uan))
        ew = abs(mean(u) - mean(uan))
        push!(cvg, (ti, h, es, ew))
    end
end

function convergence(s, int_constructor, p, h_cvg, t, W, sol_an)
    cvg = DataFrame(t = Float64[], h = Float64[], es = Float64[], ew = Float64[])
    for h in h_cvg
        tn, Wn = tnWn(t, W, h)
        sol = map(
            Wni -> solve(
                s, int_constructor(h),
                ComputedWienerIncrement{typeof(h)}(h, sqrt(h), Wni),
                p.u0, p.tmax, p.saveat
            ),
            eachcol(Wn)
        )
        compute_convergence!(cvg, h, sol, sol_an)
    end
    return cvg
end

function _tweak_dW(dW, h, s)
    foreach(dW) do x
        x.h = h
        x.sqrth = s * sqrt(h)
    end
end

# Weak convergence does not require access to the Wiener process so we can compute it in a simpler way.
# To reduce the variance, we use Antithetic variates, which allows for smaller ensemble size.
function weak_convergence(s, seuler, int_constructor, p, h_cvg; scale = 4)
    cvg = DataFrame(t = Float64[], h = Float64[], es = Float64[], ew = Float64[])

    h_small = minimum(h_cvg)/scale
    dW = [FixedWienerIncrement(h_small, p.tmax; force_statistics = true) for _ in 1:p.nens]
    sol_anp = map(dWi -> solve(seuler, EulerMaruyama(h_small), dWi, p.u0, p.tmax, p.saveat), dW);
    _tweak_dW(dW, h_small, -1)
    sol_ann = map(dWi -> solve(seuler, EulerMaruyama(h_small), dWi, p.u0, p.tmax, p.saveat), dW);
    sol_an = vcat(sol_anp, sol_ann)
    for h in h_cvg
        dW = [FixedWienerIncrement(h, p.tmax; force_statistics = true) for _ in 1:p.nens]
        _tweak_dW(dW, h, +1)
        solp = map(dWi -> solve(s, int_constructor(h), dWi, p.u0, p.tmax, p.saveat), dW)
        _tweak_dW(dW, h, -1)
        soln = map(dWi -> solve(s, int_constructor(h), dWi, p.u0, p.tmax, p.saveat), dW)
        sol = vcat(solp, soln)
        compute_convergence!(cvg, h, sol, sol_an)
    end
    cvg.es .= 0.0 # Strong convergence is wrong for this method.
    return cvg
end
