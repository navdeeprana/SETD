# Implements CGLE.
# ∂_t u = u - (1+ic) |p|^2 p + (1+ib) ∇^2 p + sqrt(2D) eta.
#
# Abuses namedtuples to create structures on the fly.

function create_grid(nx, lx)
    Δx = lx / nx
    Δk = 2π / lx
    x = Δx .* (0:1:(nx-1))
    k = fftfreq(nx, Δk * nx)
    k2 = @. k^2
    kcut = abs((2/3) * minimum(k))
    kin = @. abs(k) < kcut
    arr = Vector{ComplexF64}(undef, nx)
    pfor, pinv = plan_fft(arr; flags = FFTW.MEASURE), plan_ifft(arr; flags = FFTW.MEASURE)
    return (; nx, lx, Δx, Δk, x, k, k2, kcut, kin, pfor, pinv)
end

@inline function setd1_factors(pars, grid, h)
    (; k2, nx, lx) = grid
    G0 = sqrt(2 * pars.D * nx^2 / lx)
    f1, f2, f3 = zero(k2), zero(k2), zero(k2)
    for i in eachindex(k2)
        ci, hci = 1 - k2[i], h * (1 - k2[i])
        f1[i] = exp(hci)
        if abs(hci) <= 1.e-5
            f2[i], f3[i] = h + 0.5 * ci * h^2, G0 * sqrt(h)
        else
            f2[i], f3[i] = expm1(hci) / ci, G0 * sqrt(expm1(2hci) / (2ci))
        end
    end
    (; f1, f2, f3)
end

function nonlinear!(sim)
    (; pars, vars, grid) = sim
    (; u, uk, duk, nl) = vars

    mul!(u, grid.pinv, uk)
    @. nl = -complex(1, pars.c) * abs2(u) * u
    mul!(duk, grid.pfor, nl)
    @. duk = duk - 1im * pars.b * grid.k2 * uk
end

function setd1!(sim, int)
    (; pars, vars, grid) = sim
    (; uk, duk, nl) = vars
    (; f1, f2, f3) = int
    @inbounds for i in 1:length(grid.k)
        if grid.kin[i]
            uk[i] = f1[i]*uk[i] + f2[i]*duk[i] + f3[i]*randn(ComplexF64)
        else
            uk[i] = 0.0
        end
    end
end

function wave(q, x)
    @assert(q < 1, "q should always be smaller than 1.")
    (@. sqrt(1-q^2) * exp(1im * q * x))
end

function savedata!(fout, sim, tsave, dset_counter)
    (; grid, pars, vars, int) = sim

    dset = @sprintf "data/u_%04d" dset_counter
    mul!(vars.u, grid.pinv, vars.uk)
    jldopen(fout, "r+") do file
        file[dset] = vars.u
    end

    push!(sim.tsave, tsave)
    push!(sim.dsave, dset)
    return dset_counter + 1
end

function loaddata(fout)
    t = load(fout, "tsave")
    dsets = load(fout, "dsave")
    return (; t, u = [load(fout, dset) for dset in dsets])
end

function solve(sim, tmax, h, saveat, save_after)
    (; grid, pars, vars, int, fout) = sim
    niters, nsave = @. round(Int, (tmax, saveat)/h)
    mul!(vars.uk, grid.pfor, vars.u)

    dset_counter = 0
    @showprogress for i in 1:(niters+1)
        isave, tsave = i-1, h*(i-1)
        if (mod(isave, nsave) == 0) && (tsave >= save_after)
            dset_counter = savedata!(fout, sim, tsave, dset_counter)
        end
        nonlinear!(sim)
        setd1!(sim, int)
    end
    jldopen(fout, "r+") do file
        file["tsave"] = sim.tsave
        file["dsave"] = sim.dsave
    end
    nothing
end

# Define a custom jldsave function create groups.
function myjldsave(fout, group; kwargs...)
    jldsave(fout; (Symbol("$(group)/$(k)") => v for (k, v) in pairs(kwargs))...)
end

cgle_vars(u0) = (; u = copy(u0), uk = zero(u0), duk = zero(u0), nl = zero(u0))

function cgle_simulation(nx, lx, pars, tmax, h, saveat, save_after, seed, fout)
    # Set random seed
    Random.seed!(seed)
    # Store parameters, simulations can be reproduced exactly from the parameters.
    myjldsave(fout, "parameters"; nx, lx, tmax, h, saveat, save_after, seed, pars...)
    grid = create_grid(nx, lx)
    u0 = wave(pars.q, grid.x)
    vars = cgle_vars(u0);
    pars = (; G0 = sqrt(2 * pars.D * grid.nx^2 / grid.lx), pars...)
    int = setd1_factors(pars, grid, h)
    sim = (; grid, pars, vars, int, fout, tsave = Float64[], dsave = String[]);
    return sim
end

# Analytical expressions for equal time correlations
function Cet(k, q, D, c)
    R = sqrt(1-q^2)
    num = @. (k^2 - 2k*q + R^2)*(k^4+2k^2*R^2+(1+c^2)R^4)
    den = @. (2k^2)*(k^6 + 4k^4*(R^2-q^2) + k^2*R^2*(5R^2-8q^2)-4(1+c^2)*q^2*R^4+2R^6)
    Ck = @. (2D) * num / den
    return Ck
end

function equal_time_correlator(grid, vars, sol; skip = 0)
    Ck = zeros(grid.nx)
    for ui in sol.u[skip:end]
        mul!(vars.uk, grid.pfor, ui)
        @. Ck = Ck + abs2(vars.uk)
    end
    N = length(sol.u[skip:end])
    @. Ck = grid.lx * Ck / (grid.nx^2 * N)

    ks, Cks, kins = fftshift(copy(grid.k)), fftshift(Ck), fftshift(grid.kin)
    return ks[kins], Cks[kins]
end
