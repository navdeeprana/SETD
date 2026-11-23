using OnlineStats, Measurements, ProgressMeter, OhMyThreads


function wiener_increment_for_convergence(::AbstractNumericalMethod, h, tmax)
    return SampledWienerIncrement(h, tmax)
end

function wiener_increment_for_convergence(::Union{StrongOrder15, WeakOrder30}, h, tmax)
    return TwoPointWienerIncrement(h, tmax)
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

struct TestFunc{G, DG, D2G}
    g::G
    dg::DG
    d2g::D2G
end

function test_poly(z)
    g(x) = x^z
    dg(x) = z * x^(z - 1)
    d2g(x) = z * (z - 1) * x^(z - 2)
    return TestFunc(g, dg, d2g)
end

function test_atan()
    g(x) = (atan(x))^2
    dg(x) = (2 * atan(x)) / (1 + x^2)
    d2g(x) = (2 - 4 * x * atan(x)) / (1 + x^2)^2
    return TestFunc(g, dg, d2g)
end

meanvar() = OnlineStats.Series(Mean(), Variance())

function create_measurement(s)
    v, N = value(s), nobs(s)
    return measurement(v[1], sqrt(v[2] / N))
end

function convergence(sde, int_constructor::F1, test_func::F2, p, h_cvg, t, W, u_an) where {F1, F2}
    cvg = (; h = Float64[], es = Measurement{Float64}[], ew = Measurement{Float64}[])

    func = test_func.g

    @showprogress desc = "Convergence" for h in h_cvg
        os = OnlineStats.Group(meanvar(), meanvar(), meanvar())

        int = int_constructor(h)
        dW = wiener_increment_for_convergence(int.m, h, p.tmax)

        for (Wn, un_an) in zip(W, u_an)
            reset!(int.m)
            resample!(dW, t, Wn, h)
            un = simple_solve(sde, int, dW, p.u0, p.tmax)
            fit!(os, (abs(un - un_an), func(un), func(un_an)))
        end

        es, un, un_an = (create_measurement(s) for s in os.stats)
        ew = abs(un - un_an)
        addto!(cvg, (h, es, ew))
    end
    return cvg
end

# Weak convergence does not require access to the Wiener process so we can compute it in a simpler way.
# To reduce the variance, we optionally use Antithetic variates, which allows for smaller ensemble size.

function antithetic!(dW::SampledWienerIncrement, sign)
    @. dW.dW = sign * dW.dW
    return nothing
end

function antithetic!(dW::TwoPointWienerIncrement, sign)
    @. dW.dW = sign * dW.dW
    @. dW.I10 = sign * dW.I10
    return nothing
end

function solve_for_weak_convergence(
        sde, int_constructor::F1, dW_constructor::F2, test_func::F3, p, h;
        show_progress = true, anti = false, max_threads = Base.Threads.nthreads()
    ) where {F1, F2, F3}

    func = test_func.g

    chunks = OhMyThreads.chunks(1:p.nens; n = max_threads)
    os_chunks = [meanvar() for _ in chunks]

    prog = Progress(p.nens; desc = "Solving on $(max_threads) threads.", enabled = show_progress)
    @tasks for (nc, chunk) in enumerate(chunks)
        local dW = dW_constructor(h, p.tmax)
        local int = int_constructor(h)
        for n in chunk
            reset!(int.m)
            resample!(dW)
            un = simple_solve(sde, int, dW, p.u0, p.tmax)
            fit!(os_chunks[nc], func(un))

            if anti
                reset!(int.m)
                antithetic!(dW, -1)
                un = simple_solve(sde, int, dW, p.u0, p.tmax)
                fit!(os_chunks[nc], func(un))
            end
            next!(prog)
        end
    end
    os = meanvar()
    for osc in os_chunks
        merge!(os, osc)
    end
    return create_measurement(os)
end

function weak_convergence(sde, int_constructor, dW_constructor, test_func, p, h_cvg, stats_an; kwargs...)
    cvg = (; h = Float64[], ew = Measurement{Float64}[])

    @showprogress desc = "Weak convergence" for h in h_cvg
        stats = solve_for_weak_convergence(sde, int_constructor, dW_constructor, test_func, p, h; show_progress = false, kwargs...)
        ew = abs(stats - stats_an)
        addto!(cvg, (h, ew))
    end
    return cvg
end

@inline draw(m::AbstractNumericalMethod, sqrth) = sqrth * randn()

@inline function draw(m::WeakOrder30, sqrth)
    U1, U2 = randn(), randn()
    return sqrth * U1, 0.5 * sqrth^3 * (U1 + U2 / sqrt(3))
end

function theoretical_error(::EulerMaruyama, sde, h, p, u0, um, test_func::F) where {F}
    (; f, df, d2f, σ) = sde
    at, dat, d2at = f(u0, p), df(u0, p), d2f(u0, p)
    dgt, d2gt = test_func.dg(u0), test_func.d2g(u0)
    return (
        0.5 * h^2 * ((at * dat + 0.5 * σ^2 * d2at) * dgt + σ^2 * dat * d2gt)
    )
end

function theoretical_error(int::ABMaruyama, sde, h, p, u0, um, test_func::F) where {F}
    (; df, d2f, σ) = sde
    dat, d2at, d2atm = df(u0, p), d2f(u0, p), d2f(um, p)
    dgt, d2gt = test_func.dg(u0), test_func.d2g(u0)
    return (
        0.5 * h^2 * σ^2 * ((d2at - d2atm) * dgt + dat * d2gt)
    )
end

function theoretical_error(::BrokenWeakOrder20, sde, h, p, u0, um, test_func::F) where {F}
    (; df, d2f, σ) = sde
    dat = df(u0, p)
    d2gt = test_func.d2g(u0)
    return (
        0.5 * h^2 * σ^2 * (dat * d2gt)
    )
end

function nsteps_error(m::Union{EulerMaruyama, BrokenWeakOrder20}, sde, h, p, uf, test_func, nsteps)
    um = 0.5 * (p.u0 + uf)
    return (nsteps + 1) * abs(theoretical_error(m, sde, h, p, um, p.u0, test_func))
end

function nsteps_error(m::ABMaruyama, sde, h, p, uf, test_func, nsteps)
    um = 0.5 * (p.u0 + uf)
    return (
        nsteps * abs(theoretical_error(m, sde, h, p, um, p.u0, test_func))
            + abs(theoretical_error(EulerMaruyama(), sde, h, p, um, p.u0, test_func))
    )
end

function local_error(
        sde, int_constructor::F1, test_func::F2, p, h; nens = p.nens, nsteps = 5, show_progress = true,
    ) where {F1, F2}

    func = test_func.g
    os = OnlineStats.Group(meanvar(), meanvar(), meanvar(), meanvar())
    sqrth = sqrt(h)
    int, int_th = int_constructor(h), WeakOrder30(h)

    prog = Progress(nens; enabled = show_progress)
    for n in 1:nens
        reset!(int_th.m)
        un_th = stepforward(int_th, sde, p.u0, draw(int_th.m, sqrth))

        # Step 0 error
        reset!(int.m)
        err_path = theoretical_error(int.m, sde, h, p, p.u0, p.u0, test_func)
        un = stepforward(int, sde, p.u0, draw(int.m, sqrth))

        unm_path, un_path = p.u0, un
        for _ in 1:nsteps
            un_th = stepforward(int_th, sde, un_th, draw(int_th.m, sqrth))

            # Add error at nth step
            err_path += theoretical_error(int.m, sde, h, p, un_path, unm_path, test_func)

            un = stepforward(int, sde, un, draw(int.m, sqrth))
            unm_path, un_path = un_path, un
        end

        fit!(os, (un, err_path, func(un_th), func(un)))
        next!(prog)
    end
    ut, err_path, avg_th, avg = (create_measurement(s) for s in os)
    return ut, abs(err_path), abs(avg_th - avg)
end
