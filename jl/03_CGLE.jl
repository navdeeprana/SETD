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
using Revise, Printf, LinearAlgebra, FFTW, JLD2, Random, ProgressMeter
using CairoMakie, StatsBase
includet("src/cgle.jl")
includet("src/plotting.jl")
colors = Makie.wong_colors()
set_theme!(merge(theme_latexfonts(), makietheme()))

# %% [markdown]
# # Test Simulation

# %%
# Define parameters
nx, lx = 1024, 320π
tmax, h, saveat, save_after = 1.e+3, 1.e-1, 1.e+1, 0.e+0
seed = 42
pars = (b = 0.0, c = 0.5, D = 1.e-2, q = 16 * (2π / lx));

# %%
fout = @sprintf "data/scratch_nx_%04d_ln_%04d.jld2" nx round(Int, lx/(2π))
sim = cgle_simulation(nx, lx, pars, tmax, h, saveat, save_after, seed, fout);

# %%
@time solve(sim, tmax, h, saveat, save_after);

# %%
sol = loaddata(sim.fout);
fig, ax = figax(a = 3)
lines!(ax, real.(sol.u[1]); linewidth = 2)
lines!(ax, real.(sol.u[end]); linewidth = 3)
fig

# %% [markdown]
# # Analysis

# %%
# To generate the data, run CGLE_variousD.jl for different values of D.

# %%
# Analysis
nx, lx = 1024, 320π
tmax, h, saveat, save_after = 2.e+5, 2.e-2, 2.e+2, 0.0
pars = (b = 0.0, c = 0.1, q = 16 * (2π / lx));

grid = create_grid(nx, lx)
vars = cgle_vars(zeros(ComplexF64, nx));

# %%
function logspaced(x; base = 1.2)
    N = log(length(x)) / log(base)
    n = unique(floor.(Int, base .^ (1:N)))
    return x[n]
end

# %%
fig, axes = figax(nx = 2, yscale = log10, xlabel = L"k", ylabel = L"C_{\text{ET}}(k)")

axes[1].limits = (-1, 1, 5.e-3, 1.e+3)
axes[1].xscale = Makie.pseudolog10

ax = axes[2]
ax.limits = (1.5π/grid.lx, 1.8, 1.e-2, 5.e+2)
ax.xscale = log10

labels = [L"D=0.01", L"D=0.05", L"D=0.10"]
for (D, m, c, l) in zip([0.01, 0.05, 0.10], [:circle, :rect, :diamond], colors, labels)
    fdata = @sprintf "data/cgleD_nx_%04d_ln_%04d_D_%.2f.jld2" nx round(Int, lx/(2π)) D
    sol = loaddata(fdata);
    k, Ck = equal_time_correlator(grid, vars, sol; skip = 500);
    Ck_an = Cet(k, pars.q, D, pars.c)

    if D == 0.01
        kp = @. abs(k) > 0
        lines!(axes[1], k[kp], Ck_an[kp]; linewidth = 7, color = (colors[4], 0.4), label = "Analytical")
        lines!(axes[1], k .- pars.q, Ck; linewidth = 3, label = L"Numerics, $D=0.01$")
    end

    lines!(ax, k[k .> 0], Ck_an[k .> 0]; linewidth = 7, color = (c, 0.4))
    kq = k .- pars.q
    kq, Ckq = kq[kq .> 0], Ck[kq .> 0]
    # lines!(ax, kq, Ckq, linewidth=2)
    kp, Ckp = logspaced(kq), logspaced(Ckq)
    scatterlines!(ax, kp, Ckp, marker = m, linewidth = 3, markersize = 20, label = l)
end
k = 0.02:0.1:1.0
lines!(k, (@. 1.e-1/k^2), color = :black)
text!(ax, L"$k^{-2}$"; position = (1.5e-1, 1.e+1), fontsize = 30)
axislegend.(axes, position = :rt)
resize_to_layout!(fig)
save("figs/CGLE_correlators.pdf", fig)
fig
