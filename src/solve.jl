abstract type AbstractNumericalMethod end
struct EulerMaruyama <: AbstractNumericalMethod end
struct Milstein <: AbstractNumericalMethod end
struct SETDEulerMaruyama <: AbstractNumericalMethod end
struct SETDMilstein <: AbstractNumericalMethod end
struct SETD1 <: AbstractNumericalMethod end

struct Integrator{M<:AbstractNumericalMethod,Q}
    m::M # Integration method
    q::Q # Contains fixed parameters for the integration method.
end

EulerMaruyama(h) = Integrator(EulerMaruyama(), (h = h, sqrth = sqrt(h)))

Milstein(h) = Integrator(Milstein(), (h = h, sqrth = sqrt(h)))

# For left-point approximation, approx = 1.0 and for mid-point approximation, approx = 0.5.
# By default we choose mid-point approximation as it is more accurate.

function SETDEulerMaruyama(h, c, approx = 0.5)
    fac = (exp(c*h), expm1(c*h)/c, exp(c*h*approx))
    Integrator(SETDEulerMaruyama(), (; h, sqrth = sqrt(h), fac))
end

function SETDMilstein(h, c, approx = 0.5)
    fac = (exp(c*h), expm1(c*h)/c, exp(c*h*approx))
    Integrator(SETDMilstein(), (; h, sqrth = sqrt(h), fac))
end

# Since these integrators operate on Weiner increments, and not directly on the random
# numbers, we need to scale the stochastic integral by sqrt(h). Technically SETD1 should
# only operate on InstantWeinerIncrement, but we leave it this way.
function SETD1(h, c)
    fac = (exp(c*h), expm1(c*h)/c, sqrt(expm1(2*c*h)/(2c))/sqrt(h))
    Integrator(SETD1(), (; h, sqrth = sqrt(h), fac))
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

function solve(s::SDE, int::Integrator, dW::AbstractWeinerIncrement, u0, tmax, saveat; save_after = 0.0)
    (; h, sqrth) = int.q
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

function compute_convergence!(cvg, h, sol, sol_an, mean_an)
    for (i, ti) in enumerate(sol[1].t)
        ti == 0 ? continue : nothing
        u = [s.u[i] for s in sol]
        uan = [s[i] for s in sol_an]
        es = mean(@. abs(u - uan))
        ew = abs(mean(u) - mean(uan))
        ew_an = abs(mean(u) - mean_an[i])
        push!(cvg, (ti, h, es, ew, ew_an))
    end
end

function tnWn(t, W, twhen)
    skip = (length(t) - 1) ÷ round(Int, t[end] / twhen)
    @views tn, Wn = t[1:skip:end], W[1:skip:end, :]
    return tn, Wn
end

function convergence(s, int_constructor, p, h_cvg, t, W, sol_an, mean_an)
    cvg = DataFrame(t = Float64[], h = Float64[], es = Float64[], ew = Float64[], ew_an = Float64[])
    for h in h_cvg
        tn, Wn = tnWn(t, W, h)
        sol = map(
            Wni -> solve(
                s, int_constructor(h),
                ComputedWeinerIncrement{typeof(h)}(h, sqrt(h), Wni),
                p.u0, p.tmax, p.saveat
            ),
            eachcol(Wn)
        )
        compute_convergence!(cvg, h, sol, sol_an, mean_an)
    end
    return cvg
end
