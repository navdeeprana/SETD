using OnlineStats, Measurements, ProgressMeter, OhMyThreads


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
    @inbounds for i in 1:niters
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
            resample!(dW, t, Wn, h_exact)
            simple_solve(sde, int, dW, p.u0, p.tmax)
        end,
        W
    )
    return t, W, u_an
end

function solve_for_convergence(
        sde, int_constructor::F, p, h_cvg;
        noise_scale = 4, h_exact_scale = 32, return_noise_scale = noise_scale,
        max_threads = Base.Threads.nthreads()
    ) where {F}
    h_exact = minimum(h_cvg) / h_exact_scale
    h_noise_exact = h_exact / noise_scale

    h_noise_return = return_noise_scale * h_noise_exact

    t = 0.0:h_noise_return:p.tmax
    W = [zeros(length(t)) for n in 1:p.nens]
    u_an = zeros(p.nens)

    prog = Progress(p.nens; desc = "Solving on $(max_threads) threads.")
    @tasks for chunk in chunks(1:p.nens; n = max_threads)
        int = int_constructor(h_exact)
        dW = wiener_increment_for_convergence(int.m, h_exact, p.tmax)
        te, We = wiener_process(h_noise_exact, p.tmax)
        dWe = zeros(eltype(We), length(We) - 1)

        for n in chunk
            int = int_constructor(h_exact)
            wiener_increment!(dWe, sqrt(h_noise_exact))
            wiener_process!(We, dWe)
            resample!(dW, te, We, h_exact)
            un = simple_solve(sde, int, dW, p.u0, p.tmax)
            u_an[n] = un
            W[n] .= coarsegrain(te, We, h_noise_return)
            next!(prog)
        end
    end
    return t, W, u_an
end

test_f(x) = x^4

cvg_stats() = OnlineStats.Series(Mean(), Variance())

function create_measurement(s)
    v, N = value(s), nobs(s)
    return measurement(v[1], sqrt(v[2] / N))
end

function convergence(sde, int_constructor::F, p, h_cvg, t, W, u_an) where {F}
    cvg = (; h = Float64[], es = Measurement{Float64}[], ew = Measurement{Float64}[], f2 = Measurement{Float64}[])

    @showprogress desc = "Convergence" for h in h_cvg
        os = OnlineStats.Group(cvg_stats(), cvg_stats(), cvg_stats(), cvg_stats())
        dW = wiener_increment_for_convergence(int_constructor(h).m, h, p.tmax)

        for (Wn, un_an) in zip(W, u_an)
            int = int_constructor(h)
            resample!(dW, t, Wn, h)
            un = simple_solve(sde, int, dW, p.u0, p.tmax)
            fit!(os, (abs(un - un_an), test_f(un), test_f(un_an), 6 * un^2 * cos(un)))
        end

        es, un, un_an, f2 = (create_measurement(s) for s in os.stats)
        ew = abs(un - un_an)
        addto!(cvg, (h, es, ew, f2))
    end
    return cvg
end

# Weak convergence does not require access to the Wiener process so we can compute it in a simpler way.
# To reduce the variance, we optionally use Antithetic variates, which allows for smaller ensemble size.

function antithetic!(dW::SampledWienerIncrement, sign)
    @. dW.dW = sign * dW.dW
    return nothing
end

function antithetic!(dW::SO15WienerIncrement, sign)
    @. dW.dW = sign * dW.dW
    @. dW.I10 = sign * dW.I10
    return nothing
end

function solve_for_weak_convergence(
        sde, int_constructor::F1, dW_constructor::F2, p, h;
        show_progress = true, anti = false, max_threads = Base.Threads.nthreads()
    ) where {F1, F2}

    chunks = OhMyThreads.chunks(1:p.nens; n = max_threads)
    os_all = [cvg_stats() for _ in chunks]

    prog = Progress(p.nens; desc = "Solving on $(max_threads) threads.", enabled = show_progress)
    @tasks for (nc, chunk) in enumerate(chunks)
        dW = dW_constructor(h, p.tmax)
        for n in chunk
            resample!(dW)
            int = int_constructor(h)
            un = simple_solve(sde, int, dW, p.u0, p.tmax)
            fit!(os_all[nc], test_f(un))

            if anti
                int = int_constructor(h)
                antithetic!(dW, -1)
                un = simple_solve(sde, int, dW, p.u0, p.tmax)
                fit!(os_all[nc], test_f(un))
            end
            next!(prog)
        end
    end
    os = cvg_stats()
    for osc in os_all
        merge!(os, osc)
    end
    return create_measurement(os)
end

function weak_convergence(sde, int_constructor, dW_constructor, p, h_cvg, stats_an; kwargs...)
    cvg = (; h = Float64[], ew = Measurement{Float64}[])

    @showprogress desc = "Weak convergence" for h in h_cvg
        stats = solve_for_weak_convergence(sde, int_constructor, dW_constructor, p, h; show_progress = false, kwargs...)
        ew = abs(stats - stats_an)
        addto!(cvg, (h, ew))
    end
    return cvg
end

function single_step_error_threaded(sde, int_constructor::F, p, h; max_threads = Base.Threads.nthreads()) where {F}
    chunks = OhMyThreads.chunks(1:p.nens; n = max_threads)
    os_all = [OnlineStats.Group(cvg_stats(), cvg_stats(), cvg_stats()) for _ in chunks]
    prog = Progress(p.nens; desc = "Solving on $(max_threads) threads.")
    @tasks for (nc, chunk) in enumerate(chunks)
        sqrth = sqrt(h)
        int = int_constructor(h)
        int_an = WeakOrder20(h)
        for n in chunk
            reset!(int.m)
            dWi = sqrth * randn()
            un_an = stepforward(int_an, sde, p.u0, dWi)
            un = stepforward(int, sde, p.u0, dWi)

            local u0_an = un_an
            # dWi = sqrth * randn()
            # un_an = stepforward(int_an, sde, un_an, dWi)
            # un = stepforward(int, sde, un, dWi)

            to_fit = (u0_an, test_f(un_an), test_f(un))
            fit!(os_all[nc], to_fit)
            next!(prog)
        end
    end
    os = OnlineStats.Group(cvg_stats(), cvg_stats(), cvg_stats())
    for osc in os_all
        merge!(os, osc)
    end
    u0, avg_an, avg = (create_measurement(s) for s in os)
    return u0, abs.(avg_an - avg)
end

draw(m::AbstractNumericalMethod, sqrth) = sqrth * randn()

function draw(m::WeakOrder30, sqrth)
    U1, U2 = randn(), randn()
    return sqrth * U1, 0.5 * sqrth^3 * (U1 + U2 / sqrt(3))
end

function single_step_error(sde, int_constructor::F1, err_fun1::F2, err_fun2::F3, p, h; multi_step = false, nsteps = 5) where {F1, F2, F3}
    os = OnlineStats.Group(cvg_stats(), cvg_stats(), cvg_stats(), cvg_stats())

    sqrth = sqrt(h)
    int, int_an = int_constructor(h), WeakOrder30(h)

    multi_step = multi_step || (int.m isa ABMaruyama)

    prog = Progress(p.nens)
    for n in 1:p.nens
        reset!(int.m)
        un_an = stepforward(int_an, sde, p.u0, draw(int_an.m, sqrth))
        un = stepforward(int, sde, p.u0, draw(int.m, sqrth))

        utmh_an, ut_an = p.u0, un_an
        err_an = err_fun1(h, ut_an, utmh_an)
        if multi_step
            for _ in 1:nsteps
                err_an += err_fun2(h, ut_an, utmh_an)
                un_an = stepforward(int_an, sde, un_an, draw(int_an.m, sqrth))
                un = stepforward(int, sde, un, draw(int.m, sqrth))
                utmh_an, ut_an = ut_an, un_an
            end
        end
        fit!(os, (ut_an, err_an, test_f(un_an), test_f(un)))
        next!(prog)
    end
    u0, err_an, avg_an, avg = (create_measurement(s) for s in os)
    return u0, abs(err_an), abs(avg_an - avg)
end
