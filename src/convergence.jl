function resetdW!(dW::ComputedWienerIncrement, t, W, h)
    tn, Wn = coarsegrain(t, W, h)
    @. dW.W = Wn
    return nothing
end

function resetdW!(dW::SO15WienerIncrement, t, W, h)
    tn, Wn = coarsegrain(t, W, h)
    # Unroll the loop because the @views below allocate
    # @views @. dW.dW = Wn[2:end] - Wn[1:(end - 1)]
    @inbounds for i in eachindex(dW.dW)
        dW.dW[i] = Wn[i+1] - Wn[i]
    end
    integral_I10!(dW.I10, t, W, h)
    return nothing
end

function wiener_increment_for_convergence(::AbstractNumericalMethod, h, tmax)
    return ComputedWienerIncrement(h, tmax)
end

function wiener_increment_for_convergence(::StrongOrder15, h, tmax)
    return SO15WienerIncrement(h, tmax)
end

function save_params_for_convergence(p)
    saveat, save_after = p.tmax, 0.5 * p.tmax
    return saveat, save_after
end

function convergence(sde, int_constructor::F, p, h_cvg, t, W, sol_an) where {F}
    cvg = (; h = Float64[], es = Float64[], ew = Float64[])


    saveat, save_after = save_params_for_convergence(p)

    @showprogress for h in h_cvg
        os = OnlineStats.Group(Mean(), Mean(), Mean())
        dW = wiener_increment_for_convergence(int_constructor(h).m, h, p.tmax)
        for (Wi, sa) in zip(eachcol(W), sol_an)
            int = int_constructor(h)
            resetdW!(dW, t, Wi, h)
            sol = solve(sde, int, dW, p.u0, p.tmax, saveat; save_after)
            u, uan = sol.u[end], sa.u[end]
            fit!(os, (abs(u - uan), u, uan))
        end
        es, u, uan = value.(os.stats)
        addto!(cvg, (h, es, abs(u - uan)))
    end
    return cvg
end

# Weak convergence does not require access to the Wiener process so we can compute it in a simpler way.
# To reduce the variance, we use Antithetic variates, which allows for smaller ensemble size.

function antithesis!(dW::SampledWienerIncrement, sign, sqrth)
    @. dW.dW = sign * dW.dW
    return nothing
end

function antithesis!(dW::SO15WienerIncrement, sign, sqrth)
    @. dW.dW = sign * dW.dW
    @. dW.I10 = sign * dW.I10
    return nothing
end

function solve_for_weak_convergence(sde, int_constructor :: F, dW, p, h; ϕ = identity) where {F}
    sqrth = sqrt(h)
    saveat, save_after = save_params_for_convergence(p)
    sol = solve(sde, int_constructor(h), dW, p.u0, p.tmax, saveat; save_after)

    os = OnlineStats.Series(Mean(), Variance())


    for _ in 1:p.nens
        redraw!(dW)

        int = int_constructor(h)
        solp = solve(sde, int, dW, p.u0, p.tmax, saveat; save_after)

        int = int_constructor(h)
        antithesis!(dW, -1, sqrth)
        solm = solve(sde, int, dW, p.u0, p.tmax, saveat; save_after)

        fit!(os, 0.5 * (ϕ(solp.u[end]) + ϕ(solm.u[end])))
    end
    mean, std = value(os.stats[1]), sqrt(value(os.stats[2]))
    return (; mean, std)
end

function weak_convergence(sde, int_constructor, dW_constructor, p, h_cvg, stats_an; scale = 32, ϕ = identity)
    cvg = (; h = Float64[], ew = Float64[], ewe = Float64[])

    @showprogress for h in h_cvg
        dW = dW_constructor(h, p.tmax)
        stats = solve_for_weak_convergence(sde, int_constructor, dW, p, h; ϕ)
        ew = abs(stats.mean - stats_an.mean)
        err = sqrt(stats.std^2 + stats_an.std^2)
        addto!(cvg, (h, ew, err))
    end
    return cvg
end
