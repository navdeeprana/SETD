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

function convergence(sde, int_constructor, p, h_cvg, t, W, sol_an)
    cvg = (; t = Float64[], h = Float64[], es = Float64[], ew = Float64[])

    t_an = sol_an[1].t

    @showprogress for h in h_cvg
        os = [OnlineStats.Group(Mean(), Mean(), Mean()) for _ in t_an]
        dW = wiener_increment_for_convergence(int_constructor(h).m, h, p.tmax)
        for (Wi, sa) in zip(eachcol(W), sol_an)
            int = int_constructor(h)
            resetdW!(dW, t, Wi, h)
            sol = solve(sde, int, dW, p.u0, p.tmax, p.saveat, save_after = p.save_after)
            u_an = sa.u
            for (n, tn) in enumerate(t_an)
                fit!(os[n], (abs(sol.u[n] - u_an[n]), sol.u[n], u_an[n]))
            end
        end
        for (n, tn) in enumerate(t_an)
            es, u, uan = value.(os[n].stats)
            addto!(cvg, (tn, h, es, abs(u - uan)))
        end
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
    sol = solve(sde, int_constructor(h), dW, p.u0, p.tmax, p.saveat, save_after = p.save_after)

    os = [OnlineStats.Series(Mean(), Variance()) for _ in sol.t]

    for _ in 1:p.nens
        redraw!(dW)

        int = int_constructor(h)
        solp = solve(sde, int, dW, p.u0, p.tmax, p.saveat, save_after = p.save_after)

        int = int_constructor(h)
        antithesis!(dW, -1, sqrth)
        solm = solve(sde, int, dW, p.u0, p.tmax, p.saveat, save_after = p.save_after)
        for n in eachindex(solp.t)
            fit!(os[n], 0.5 * (ϕ(solp.u[n]) + ϕ(solm.u[n])))
        end
    end
    mean = [value(o.stats[1]) for o in os]
    std = [sqrt(value(o.stats[2])) for o in os]
    return (; sol.t, mean, std)
end

function weak_convergence(sde, int_constructor, dW_constructor, p, h_cvg, stats_an; scale = 32, ϕ = identity)
    cvg = (; t = Float64[], h = Float64[], ew = Float64[], ewe = Float64[])

    @showprogress for h in h_cvg
        dW = dW_constructor(h, p.tmax)
        stats = solve_for_weak_convergence(sde, int_constructor, dW, p, h; ϕ)
        for (n, tn) in enumerate(stats.t)
            ew = abs(stats.mean[n] - stats_an.mean[n])
            err = sqrt(stats.std[n]^2 + stats_an.std[n]^2)
            addto!(cvg, (tn, h, ew, err))
        end
    end
    return cvg
end
