using OnlineStats, Measurements, ProgressMeter

function resetdW!(dW::SampledWienerIncrement, t, W, h)
    Wn = coarsegrain(t, W, h)
    @inbounds for i in eachindex(dW.dW)
        dW.dW[i] = Wn[i + 1] - Wn[i]
    end
    return nothing
end

function resetdW!(dW::SO15WienerIncrement, t, W, h)
    Wn = coarsegrain(t, W, h)
    @inbounds for i in eachindex(dW.dW)
        dW.dW[i] = Wn[i + 1] - Wn[i]
    end
    integral_I10!(dW.I10, t, W, h)
    return nothing
end

function wiener_increment_for_convergence(::AbstractNumericalMethod, h, tmax)
    return SampledWienerIncrement(h, tmax)
end

function wiener_increment_for_convergence(::StrongOrder15, h, tmax)
    return SO15WienerIncrement(h, tmax)
end

# Simple solve for convergence
@inline function simple_solve(s::AbstractSDE, int::Integrator, dW::AbstractWienerIncrement, u0, tmax)
    (; h) = int.q
    niters = @. round(Int, tmax / h)
    ui = u0
    for i in 1:niters
        dWi = dW[i]
        ui = stepforward(int, s, ui, dWi)
    end
    return ui
end

function solve_for_convergence_old(sde, int_constructor::F, p, h_cvg; scale = 32, scale_an = 4) where {F}
    h_small = minimum(h_cvg) / scale
    h_exact = scale_an * h_small
    int = int_constructor(h_exact)
    t, W = wiener_process(h_small, p.tmax, p.nens)
    dW = wiener_increment_for_convergence(int.m, h_exact, p.tmax)
    u_an = @time @showprogress desc = "Solving" map(
        Wn -> begin
            int = int_constructor(h_exact)
            resetdW!(dW, t, Wn, h_exact)
            simple_solve(sde, int, dW, p.u0, p.tmax)
        end,
        W
    )
    return t, W, u_an
end

function solve_for_convergence(sde, int_constructor::F, p, h_cvg; noise_scale = 4, h_exact_scale = 32, return_noise_scale = noise_scale) where {F}
    h_exact = minimum(h_cvg) / h_exact_scale
    int = int_constructor(h_exact)
    dW = wiener_increment_for_convergence(int.m, h_exact, p.tmax)

    h_noise_exact = h_exact / noise_scale
    te, We = wiener_process(h_noise_exact, p.tmax)
    dWe = zeros(eltype(We), length(We) - 1)

    h_noise_return = return_noise_scale * h_noise_exact

    t = 0.0:h_noise_return:p.tmax
    W = [zeros(length(t)) for n in 1:p.nens]
    u_an = zeros(p.nens)
    @time @showprogress desc = "Solving" for n in 1:p.nens
        int = int_constructor(h_exact)
        wiener_increment!(dWe, sqrt(h_noise_exact))
        wiener_process!(We, dWe)
        resetdW!(dW, te, We, h_exact)
        un = simple_solve(sde, int, dW, p.u0, p.tmax)
        u_an[n] = un
        W[n] .= coarsegrain(te, We, h_noise_return)
    end
    return t, W, u_an
end

weak_g(x) = x^3

cvg_stats() = OnlineStats.Series(Mean(), Variance())

function create_measurement(s)
    v, N = value(s), nobs(s)
    return measurement(v[1], v[2] / sqrt(N))
end

function convergence(sde, int_constructor::F, p, h_cvg, t, W, u_an) where {F}
    cvg = (; h = Float64[], es = Measurement{Float64}[], ew = Measurement{Float64}[])

    @showprogress desc = "Convergence" for h in h_cvg
        os = OnlineStats.Group(cvg_stats(), cvg_stats(), cvg_stats())
        dW = wiener_increment_for_convergence(int_constructor(h).m, h, p.tmax)

        for (Wn, un_an) in zip(W, u_an)
            int = int_constructor(h)
            resetdW!(dW, t, Wn, h)
            un = simple_solve(sde, int, dW, p.u0, p.tmax)
            fit!(os, (abs(un - un_an), weak_g(un), weak_g(un_an)))
        end

        es, un, un_an = (create_measurement(s) for s in os.stats)
        ew = abs(un - un_an)
        addto!(cvg, (h, es, ew))
    end
    return cvg
end

# Weak convergence does not require access to the Wiener process so we can compute it in a simpler way.
# To reduce the variance, we use Antithetic variates, which allows for smaller ensemble size.

function antithetic!(dW::SampledWienerIncrement, sign)
    @. dW.dW = sign * dW.dW
    return nothing
end

function antithetic!(dW::SO15WienerIncrement, sign)
    @. dW.dW = sign * dW.dW
    @. dW.I10 = sign * dW.I10
    return nothing
end

function solve_for_weak_convergence(sde, int_constructor::F, dW, p, h; show_progress = true) where {F}
    os = cvg_stats()

    prog = Progress(p.nens; desc = "Solving", enabled = show_progress)
    for _ in 1:p.nens
        redraw!(dW)

        int = int_constructor(h)
        un = simple_solve(sde, int, dW, p.u0, p.tmax, saveat; save_after)
        fit!(os, weak_g(un))

        int = int_constructor(h)
        antithetic!(dW, -1)
        un = simple_solve(sde, int, dW, p.u0, p.tmax, saveat; save_after)
        fit!(os, weak_g(un))
        next!(prog)
    end
    return create_measurement(os)
end

function weak_convergence(sde, int_constructor, dW_constructor, p, h_cvg, stats_an)
    cvg = (; h = Float64[], ew = Measurement{Float64}[])

    @showprogress desc = "Weak convergence" for h in h_cvg
        dW = dW_constructor(h, p.tmax)
        stats = solve_for_weak_convergence(sde, int_constructor, dW, p, h; show_progress = false)
        ew = abs(stats - stats_an)
        addto!(cvg, (h, ew))
    end
    return cvg
end
