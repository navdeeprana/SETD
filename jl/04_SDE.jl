# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Julia 1.10
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
using OnlineStats, StatsBase, LinearAlgebra
includet("src/plotting.jl")
includet("src/brownian.jl")
includet("src/sde_examples.jl")
includet("src/solve.jl")
includet("src/convergence.jl")
includet("src/utils.jl")
colors = Makie.wong_colors();
set_theme!(makietheme())
CairoMakie.enable_only_mime!("html")
Random.seed!(42);

# %% [markdown]
# # Simple SDE

# %%
function AdditiveSDE(p)
    f(u, p) = p.a * u - p.b * u^2
    df(u, p) = p.a - 2 * p.b * u
    d2f(u, p) = - 2 * p.b
    g(u, p) = sqrt(2 * p.D)
    return AdditiveSDE(f, df, d2f, g, p)
end

# %% [markdown]
# # Convergence for the Additive SDE

# %%
p_rest = (; u0 = 0.0, tmax = 1.0, D = 0.1, a = 0.2, b = 0.2, saveat = 1.0)
scale = 64
h_cvg = @. 1 / 2^(2:8)

# First use a StrongOrder15 solution as an approximation
h_small = minimum(h_cvg) / scale
h_analytical = 2 * h_small
p = (nens = 20000, p_rest...)
args = (p.u0, p.tmax, p.saveat)
t, W = wiener_process(h_small, p.tmax, p.nens)

sde, int = AdditiveSDE(p), StrongOrder15(h_analytical)
dW = wiener_increment_for_convergence(int.m, h_analytical, p.tmax)
sol_an = @showprogress map(
    Wi -> begin
        resetdW!(dW, t, Wi, h_analytical)
        solve(sde, int, dW, args..., save_after = 0.5 * p.saveat)
    end,
    eachcol(W)
);

# %%
sde = AdditiveSDE(p)
cvg = (
    em = convergence(sde, EulerMaruyama, p, h_cvg, t, W, sol_an),
    so15 = convergence(sde, StrongOrder15, p, h_cvg, t, W, sol_an),
    wo2 = convergence(sde, WeakOrder20, p, h_cvg, t, W, sol_an),
);

# %%
fig, axes = figax(nx = 2, ny = 1, xscale = log2, s = 130, yscale = log2, xlabel = L"h")
axes[1].yticks = (collect(2.0 .^ (-10:2:4)), [L"2^{%$i}" for i in -10:2:4])
axes[2].yticks = (collect(2.0 .^ (-10:2:4)), [L"2^{%$i}" for i in -10:2:4])
axes[1].title = "Strong convergence"
axes[2].title = "Weak convergence"
plot_convergence(fig, axes[1], axes[2], cvg.em, marker = :circle, label = "EM")
plot_convergence(fig, axes[1], axes[2], cvg.so15, marker = :circle, label = "SO1.5")
plot_convergence(fig, axes[1], axes[2], cvg.wo2, marker = :circle, label = "WO2")
lines!(axes[1], h_cvg, (@. 1.0e-1 * (h_cvg)^1.0), linewidth = 3, color = :black)
lines!(axes[1], h_cvg, (@. 1.0e-2 * (h_cvg)^1.5), linewidth = 3, color = :black)

lines!(axes[2], h_cvg, (@. 1.0e-1 * (h_cvg)^1.0), linewidth = 3, color = :black)
lines!(axes[2], h_cvg, (@. 1.0e-2 * (h_cvg)^2.0), linewidth = 3, color = :black)
axislegend.(axes[2], position = :rb)
resize_to_layout!(fig)
fig

# %% [markdown]
# # Weak convergence for the Additive SDE

# %%
p = (dt = h_small, nens = 10000, p_rest...)
sde = AdditiveSDE(p)
weak_cvg = (
    em = weak_convergence(sde, sde, EulerMaruyama, SampledWienerIncrement, p, h_cvg; ϕ = u -> u^2),
    so15 = weak_convergence(sde, sde, StrongOrder15, SO15WienerIncrement, p, h_cvg; ϕ = u -> u^2),
    wo2 = weak_convergence(sde, sde, WeakOrder20, SampledWienerIncrement, p, h_cvg; ϕ = u -> u^2),
);

# %%
fig, axes = figax(nx = 2, ny = 1, xscale = log2, s = 130, yscale = log2, xlabel = L"h")
axes[1].yticks = (collect(2.0 .^ (-10:2:4)), [L"2^{%$i}" for i in -10:2:4])
axes[2].yticks = (collect(2.0 .^ (-10:2:4)), [L"2^{%$i}" for i in -10:2:4])
axes[1].title = "Strong convergence"
axes[2].title = "Weak convergence"
# plot_convergence(fig, axes[1], axes[2], cvg.em, marker = :circle, label = "EM")
plot_convergence(fig, axes[1], axes[2], cvg.so15, marker = :circle, label = "SO1.5")
# plot_convergence(fig, axes[1], axes[2], weak_cvg.em; ignore_es = true, marker = :circle, label = "EM")
plot_convergence(fig, axes[1], axes[2], weak_cvg.so15; ignore_es = true, marker = :circle, label = "SO1.5 W")

lines!(axes[1], h_cvg, (@. 1.0e-1 * (h_cvg)^1.0), linewidth = 3, color = :black)
lines!(axes[1], h_cvg, (@. 1.0e-2 * (h_cvg)^1.5), linewidth = 3, color = :black)

lines!(axes[2], h_cvg, (@. 1.0e-1 * (h_cvg)^1.0), linewidth = 3, color = :black)
lines!(axes[2], h_cvg, (@. 1.0e-2 * (h_cvg)^2.0), linewidth = 3, color = :black)
axislegend.(axes[2], position = :rb)
resize_to_layout!(fig)
# save("figs/SAO_convergence.pdf", fig)
save("Untitled.png", fig)
fig

# %%
