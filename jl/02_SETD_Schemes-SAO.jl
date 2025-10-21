# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: julia 1.10.10
#     language: julia
#     name: julia-1.10
# ---

# %%
# Imports and setup
import Pkg;
Pkg.activate(".");
Pkg.instantiate();

# %%
using Revise, Printf, CairoMakie, DataFrames, StatsBase, Random, FFTW, ProgressMeter
includet("src/plotting.jl")
includet("src/brownian.jl")
includet("src/sde_examples.jl")
includet("src/solve.jl")
colors = Makie.wong_colors();
set_theme!(merge(theme_latexfonts(), makietheme()))
CairoMakie.enable_only_mime!("html")
Random.seed!(42);

# %% [markdown]
# # Stochastic Anharmonic Oscillator

# %%
function SAO(p)
    f(u, p) = -p.Γ * (u + p.b * u^p.z)
    g(u, p) = sqrt(2*p.Γ*p.T)
    dg(u, p) = 0
    return SDE(f, g, dg, p)
end

function SAO_SETD(p)
    f(u, p) = - p.Γ * p.b * u^p.z
    g(u, p) = sqrt(2*p.Γ*p.T)
    dg(u, p) = 0
    return SDE(f, g, dg, p)
end

# %%
p_rest = (; u0 = 0.0, tmax = 10.0, nens = 10000, T = 6.0, Γ = 5.0, b = 1.e-2, z = 3, saveat = 0.1, save_after = 2.0);

# %%
p = (; dt = 1.e-1, p_rest...)
dW = [SampledWeinerIncrement(p.dt, p.tmax) for _ in 1:p.nens]
args, kwargs = (p.u0, p.tmax, p.saveat), (; save_after = p.save_after)
sol_em1 = map(dWi -> solve(SAO(p), EulerMaruyama(p.dt), dWi, args...; kwargs...), dW);
sol_et1 = map(dWi -> solve(SAO_SETD(p), SETDEulerMaruyama(p.dt, -p.Γ, 0.5), dWi, args...; kwargs...), dW);
sol_ex1 = map(dWi -> solve(SAO_SETD(p), SETD1(p.dt, -p.Γ), dWi, args...; kwargs...), dW);

# %%
p = (; dt = 1.e-2, p_rest...)
dW = [SampledWeinerIncrement(p.dt, p.tmax) for _ in 1:p.nens]
args, kwargs = (p.u0, p.tmax, p.saveat), (; save_after = p.save_after)
sol_em2 = map(dWi -> solve(SAO(p), EulerMaruyama(p.dt), dWi, args...; kwargs...), dW);
sol_et2 = map(dWi -> solve(SAO_SETD(p), SETDEulerMaruyama(p.dt, -p.Γ, 0.5), dWi, args...; kwargs...), dW);
sol_ex2 = map(dWi -> solve(SAO_SETD(p), SETD1(p.dt, -p.Γ), dWi, args...; kwargs...), dW);

# %%
using OnlineStats, StatsBase, LinearAlgebra
function probability_distribution(sol; bins = -9:0.2:9)
    h = OnlineStats.Hist(bins)
    for si in sol
        @views u = si.u[2:end]
        fit!(h, u)
    end
    hn = normalize(Histogram(h.edges, h.counts))
    return (; x = midpoints(h), P = hn.weights)
end

# %%
fig, axes = figax(nx = 3, xlabel = L"$u$", limits = (-9, 9, -0.01, 0.23), yticks = [0.0, 0.1, 0.2])
axes[1].ylabel = L"$P(u)$"
for ax in axes
    plot_boltzmann_distribution!(ax, p, 9.0; color = (:black, 0.15), linewidth = 10)
end

axes[1].title = ("EM")
P = probability_distribution(sol_em1)
lines!(axes[1], P.x, P.P; linewidth = 5, label = "h=0.1")
P = probability_distribution(sol_em2)
lines!(axes[1], P.x, P.P; linewidth = 5, label = "h=0.01")

axes[2].title = ("SETD-EM")
P = probability_distribution(sol_et1)
lines!(axes[2], P.x, P.P; linewidth = 5, label = "h=0.1")
P = probability_distribution(sol_et2)
lines!(axes[2], P.x, P.P; linewidth = 5, label = "h=0.01")

axes[3].title = ("SETD1")
P = probability_distribution(sol_ex1)
lines!(axes[3], P.x, P.P; linewidth = 5, label = "h=0.1")
P = probability_distribution(sol_ex2)
lines!(axes[3], P.x, P.P; linewidth = 5, label = "h=0.01")

axislegend.(axes)
resize_to_layout!(fig)
save("figs/SAO_probability.pdf", fig)
fig

# %% [markdown]
# # Correlators and asymptotic variance for the OU-process

# %%
function OU(p)
    f(u, p) = p.k * u
    g(u, p) = sqrt(2*p.D)
    dg(u, p) = 0
    return SDE(f, g, dg, p)
end

function OU_SETD(p)
    f(u, p) = p.δ * u
    g(u, p) = sqrt(2*p.D)
    dg(u, p) = 0
    return SDE(f, g, dg, p)
end

# %%
p_rest = (; u0 = 0.0, tmax = 250.0, nens = 5000, k = -5.0, D = 5.0, δ = 0.0, saveat = 0.1, save_after = 5.0);
p = (; dt = 1.e-1, p_rest...)
dW = [SampledWeinerIncrement(p.dt, p.tmax) for _ in 1:p.nens]
args, kwargs = (p.u0, p.tmax, p.saveat), (; save_after = p.save_after)
sol_em = map(dWi -> solve(OU(p), EulerMaruyama(p.dt), dWi, args...; kwargs...), dW);
sol_et = map(dWi -> solve(OU_SETD(p), SETDEulerMaruyama(p.dt, p.k-p.δ), dWi, args...; kwargs...), dW);
sol_ex = map(dWi -> solve(OU_SETD(p), SETD1(p.dt, p.k-p.δ), dWi, args...; kwargs...), dW);

# %%
function correlator(sol)
    hanning(N) = @. sin(π*(1:N)/N)^2
    t = sol[1].t
    tspan, NT = t[end] - t[1], length(t)
    w = hanning(NT)
    # uw2 = map(s -> abs2.(fft(w .* s.u)), sol);
    # C = fftshift(mean(uw2))
    uw2 = map(s -> abs2.(fft(s.u)), sol);
    Cun = fftshift(mean(uw2))
    ω = fftshift(fftfreq(NT, (2π/tspan)*NT))
    scale = tspan/NT^2
    # return ω, (@. 1.63^2 * scale * C), (@. scale * Cun)
    return ω, (@. scale * Cun)
end

correlator_analytical(ω, D, c) = @. (2D)/(ω^2+c^2)
function correlator_discrete(ω, D, c, dt)
    a = exp(-c*dt)
    return @. dt*(D/c)*((1-a^2)/(1+a^2-2*a*cos(w*dt)))
end

# %%
fig, axes = figax(nx = 3, yscale = log10, xlabel = L"\omega")
axes[1].ylabel=L"C(\omega)"

w, C = correlator(sol_em)
lines!(axes[1], w, C, label = "EM")
w, C = correlator(sol_et)
lines!(axes[2], w, C, label = "SETD-EM")
w, C = correlator(sol_ex)
lines!(axes[3], w, C, label = "SETD1")

for ax in axes
    Can = correlator_analytical(w, p.D, p.k)
    lines!(ax, w, Can, color = (:black, 0.5), linewidth = 8, label = "Continuum")
    Can = correlator_discrete(w, p.D, p.k, p.dt)
    lines!(ax, w, Can, color = (colors[2], 0.5), linewidth = 8, label = "Discrete")
end

axislegend.(axes, position = :cb)

save("figs/OU_correlators.pdf")
resize_to_layout!(fig)
fig

# %%
OU_EM_var(k, D, h) = 2*D*h/(1-(1+k*h)^2)

function OU_SETDEM_var(k, D, h, δ)
    c = k-δ
    fac = (exp(c*h), (exp(c*h)-1)/c, exp(c*h/2))
    return 2*D*h*fac[3]^2/(1 - (fac[1]+δ*fac[2])^2)
end

function OU_SETD1_var(k, D, h, δ)
    c = (k-δ)
    fac = (exp(c*h), (exp(c*h)-1)/c, sqrt((exp(2*c*h)-1)/(2c)))
    return 2*D*fac[3]^2/(1 - (fac[1]+δ*fac[2])^2)
end

function varerr!(vs, sol)
    uall = vcat([s.u for s in sol]...)
    u = Iterators.partition(uall, length(uall)÷40)
    vi = [var(ui) for ui in u]
    push!(vs.m, mean(vi))
    push!(vs.s, std(vi))
end

# %%
p_rest = (; u0 = 1.0, tmax = 50.0, nens = 50000, k = -2.0, D = 1.0, saveat = 0.1, save_after = 10.0);
dt_var = @. 1/2^(3:0.5:8)

var1 = (; m = Float64[], s = Float64[])
var2 = (; m = Float64[], s = Float64[])
var3 = (; m = Float64[], s = Float64[])

@showprogress for dt in dt_var
    p = (; p_rest...)
    dW = [SampledWeinerIncrement(dt, p.tmax) for _ in 1:p.nens]

    sol = map(dWi -> solve(OU(p), EulerMaruyama(dt), dWi, p.u0, p.tmax, p.saveat), dW);
    varerr!(var1, sol)

    p = (; δ = -0.5, p_rest...)
    sol = map(dWi -> solve(OU_SETD(p), SETDEulerMaruyama(dt, p.k - p.δ), dWi, p.u0, p.tmax, p.saveat), dW);
    varerr!(var2, sol)

    p = (; δ = -0.5, p_rest...)
    sol = map(dWi -> solve(OU_SETD(p), SETD1(dt, p.k - p.δ), dWi, p.u0, p.tmax, p.saveat), dW);
    varerr!(var3, sol)
end

# %%
fig, ax = figax(xscale = log2, xlabel = L"h", ylabel = "Asymptotic variance")
errorscatter!(ax, dt_var, var1.m, var1.s, marker = :circle, label = "Euler")
errorscatter!(ax, dt_var, var2.m, var2.s, marker = :rect, label = "SETD-EM")
errorscatter!(ax, dt_var, var3.m, var3.s, marker = :diamond, label = "SETD1")
x = @. 1/2^(3:0.1:8)
lines!(ax, x, OU_EM_var.(p.k, p.D, x), color = (colors[1], 0.5), linewidth = 5)
lines!(ax, x, OU_SETDEM_var.(p.k, p.D, x, -0.5), color = (colors[2], 0.5), linewidth = 5)
lines!(ax, x, OU_SETD1_var.(p.k, p.D, x, -0.5), color = (colors[3], 0.5), linewidth = 5)
lines!(ax, x, fill(0.5, length(x)), color = (:black, 0.5), linewidth = 5, label = "Analytical")
axislegend(ax, position = :lt, nbanks = 2)
save("figs/OU_asymptotic_variance.pdf")
fig
