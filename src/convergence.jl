using Random, OnlineStats, Measurements, LsqFit

# function resetdW!(dW::ComputedWienerIncrement, t, W, h)
#     tn, Wn = coarsegrain(t, W, h)
#     @. dW.W = Wn
#     return nothing
# end

function resetdW!(dW::SampledWienerIncrement, t, W, h)
    tn, Wn = coarsegrain(t, W, h)
    # Unroll the loop because the @views below allocate
    # @views @. dW.dW = Wn[2:end] - Wn[1:(end - 1)]
    @inbounds for i in eachindex(dW.dW)
        dW.dW[i] = Wn[i+1] - Wn[i]
    end
    return nothing
end

function resetdW!(dW::SO15WienerIncrement, t, W, h)
    tn, Wn = coarsegrain(t, W, h)
    @inbounds for i in eachindex(dW.dW)
        dW.dW[i] = Wn[i+1] - Wn[i]
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

function save_params_for_convergence(p)
    saveat, save_after = p.tmax, 0.5 * p.tmax
    return saveat, save_after
end

function solve_for_convergence(sde, int_constructor :: F, p, h_cvg; scale = 32, scale_an = 4) where {F}
    h_small = minimum(h_cvg) / scale
    h_analytical = scale_an * h_small
    saveat, save_after = save_params_for_convergence(p)
    int = int_constructor(h_analytical)
    t, W = wiener_process(h_small, p.tmax, p.nens)
    dW = wiener_increment_for_convergence(int.m, h_analytical, p.tmax)
    sol_an = @time @showprogress map(
        Wi -> begin
            resetdW!(dW, t, Wi, h_analytical)
            solve(sde, int, dW, p.u0, p.tmax, saveat; save_after)
        end,
        eachcol(W)
        );
    return t, W, sol_an
end

function create_measurement(s)
    v, N = value(s), nobs(s)
    return measurement(v[1], v[2]/sqrt(N))
end

test_function(x) = x^3

function convergence(sde, int_constructor::F, p, h_cvg, t, W, sol_an) where {F}

    cvg = (; h = Float64[], es = Measurement{Float64}[], ew = Measurement{Float64}[])

    saveat, save_after = save_params_for_convergence(p)

    _stats() = OnlineStats.Series(Mean(), Variance())

    @showprogress for h in h_cvg
        os = OnlineStats.Group(_stats(), _stats(), _stats())
        dW = wiener_increment_for_convergence(int_constructor(h).m, h, p.tmax)
        for (Wi, sa) in zip(eachcol(W), sol_an)
            int = int_constructor(h)
            resetdW!(dW, t, Wi, h)
            sol = solve(sde, int, dW, p.u0, p.tmax, saveat; save_after)
            u, uan = sol.u[end], sa.u[end]
            fit!(os, (abs(u - uan), test_function(u), test_function(uan)))
        end
        es, u, uan = (create_measurement(s) for s in os.stats)
        ew = abs(u - uan)
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

function solve_for_weak_convergence(sde, int_constructor :: F, dW, p, h) where {F}
    sqrth = sqrt(h)
    saveat, save_after = save_params_for_convergence(p)
    sol = solve(sde, int_constructor(h), dW, p.u0, p.tmax, saveat; save_after)

    os = OnlineStats.Series(Mean(), Variance(), Extrema())

    for _ in 1:p.nens
        redraw!(dW)

        int = int_constructor(h)
        sol = solve(sde, int, dW, p.u0, p.tmax, saveat; save_after)
        fit!(os, sol.u[end])

        # int = int_constructor(h)
        # antithetic!(dW, -1)
        # sol = solve(sde, int, dW, p.u0, p.tmax, saveat; save_after)
        # fit!(os, sol.u[end])
    end
    mean, std, ext = value(os.stats[1]), sqrt(value(os.stats[2])), value(os.stats[3])
    return (; mean, std, ext)
end

function weak_convergence(sde, int_constructor, dW_constructor, p, h_cvg, stats_an; scale = 32)
    cvg = (; h = Float64[], ew = Float64[], ewe = Float64[])

    @showprogress for h in h_cvg
        dW = dW_constructor(h, p.tmax)
        stats = solve_for_weak_convergence(sde, int_constructor, dW, p, h)
        ew = abs(stats.mean - stats_an.mean)
        err = sqrt(stats.std^2 + stats_an.std^2)
        addto!(cvg, (h, ew, err))
    end
    return cvg
end
