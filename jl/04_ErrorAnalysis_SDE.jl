# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Julia 1.13
#     language: julia
#     name: julia-1.13
# ---

# %%
# Imports and setup
import Pkg;
Pkg.activate(".");
Pkg.instantiate();

# %%
using Revise, Printf, CairoMakie, Random, ProgressMeter
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
# # One-step error for Sine

# %%
h_err = [5.0e-3, 1.0e-2, 2.0e-2, 4.0e-2, 8.0e-2];
nens_err = [51200000, 12800000, 12800000, 12800000, 6400000];
var_err = zip(h_err, nens_err)

# %%
p = (; u0 = -2.5, σ = 1.0, k = 1.0)
sde = Sine(p);
nsteps = 4
test_func = test_atan()
ut, err_an, err = zip(
    [local_error(sde, EulerMaruyama, test_func, p, h; nens, nsteps) for (h, nens) in var_err]...
) .|> collect;

# %%
fig, ax = figax(yscale = log10, xscale = log10, title = "Local error for fixed nsteps varying h")
scatterlines!(ax, h_err, err; linewidth = 3, markersize = 25, linestyle = :dash, label = "Err")
scatterlines!(ax, h_err, err_an; linewidth = 3, markersize = 25, linestyle = :dash, label = "Err an")
err_n = [nsteps_error(EulerMaruyama(), sde, h, p, uti, test_func, nsteps) for (h, uti) in zip(h_err, ut)]
lines!(ax, h_err, err_n, label = "Absolute Sum")
axislegend(ax, position = :rb)
fig

# %%
sde = Sine(p);
nsteps = 4
test_func = test_atan()
ut, err_an, err = zip(
    [local_error(sde, BrokenWeakOrder20, test_func, p, h; nens, nsteps) for (h, nens) in var_err]...
) .|> collect;

# %%
fig, ax = figax(yscale = log10, xscale = log10, title = "Local error for fixed nsteps varying h")
scatterlines!(ax, h_err, err; linewidth = 3, markersize = 25, linestyle = :dash, label = "Err")
scatterlines!(ax, h_err, err_an; linewidth = 3, markersize = 25, linestyle = :dash, label = "Err an")
err_n = [nsteps_error(BrokenWeakOrder20(), sde, h, p, uti, test_func, nsteps) for (h, uti) in zip(h_err, ut)]
lines!(ax, h_err, err_n, label = "Sum Less")
axislegend(ax, position = :rb)
fig

# %%
sde = Sine(p);
nsteps = 6
test_func = test_atan()
ut, err_an, err = zip(
    [local_error(sde, ABMaruyama, test_func, p, h; nens, nsteps) for (h, nens) in var_err]...
) .|> collect;

# %%
fig, ax = figax(yscale = log10, xscale = log10, title = "Local error for fixed nsteps varying h")
scatterlines!(ax, h_err, err; linewidth = 3, markersize = 25, linestyle = :dash, label = "Err")
scatterlines!(ax, h_err, err_an; linewidth = 3, markersize = 25, linestyle = :dash, label = "Err an")
err_n = [nsteps_error(ABMaruyama(), sde, h, p, uti, test_func, nsteps) for (h, uti) in zip(h_err, ut)]
lines!(ax, h_err, err_n, label = "Sum Less")
axislegend(ax, position = :rb)
fig

# %%
p_rest = (; u0 = -2.5, σ = 1.0, k = 1.0)
h = 2.0e-2
p = (nens = 10000000, p_rest...)
sde = Sine(p);
step_range = 1:1:6
ut, err_an, err = zip([local_error(sde, EulerMaruyama, test_func, p, h; nsteps) for nsteps in step_range]...) .|> collect;

# %%
fig, ax = figax(yscale = log10, xscale = log10, title = "Local error for fixed h varying nsteps")
x = step_range
scatterlines!(ax, x, err; linewidth = 3, markersize = 25, linestyle = :dash, label = "Err")
scatterlines!(ax, x, err_an; linewidth = 3, markersize = 25, linestyle = :dash, label = "Err an")
err_n = [nsteps_error(EulerMaruyama(), sde, h, p, uti, test_func, nsteps) for (nsteps, uti) in zip(x, ut)]
lines!(ax, x, err_n, label = "Sum Less")
axislegend(ax, position = :rb)
fig

# %%
p_rest = (; u0 = -2.5, σ = 1.0, k = 1.0)
h = 2.0e-2
p = (nens = 10000000, p_rest...)
sde = Sine(p);
step_range = 1:1:6
ut, err_an, err = zip([local_error(sde, ABMaruyama, test_func, p, h; nsteps) for nsteps in step_range]...) .|> collect;

# %%
fig, ax = figax(yscale = log10, xscale = log10, title = "Local error for fixed h varying nsteps")
x = step_range
scatterlines!(ax, x, err; linewidth = 3, markersize = 25, linestyle = :dash, label = "Err")
scatterlines!(ax, x, err_an; linewidth = 3, markersize = 25, linestyle = :dash, label = "Err an")
err_n = [nsteps_error(ABMaruyama(), sde, h, p, uti, test_func, nsteps) for (nsteps, uti) in zip(x, ut)]
lines!(ax, x, err_n, label = "Sum Less")
axislegend(ax, position = :rb)
fig

# %% [markdown]
# # One-step error for Phi4

# %%
p_rest = (; u0 = 0.9, σ = 1.0, a = 1.0, b = 1.0)
p = (nens = 5000000, p_rest...)
sde = Phi4(p);
nsteps = 3
ut, err_an, err = zip(
    [local_error(sde, EulerMaruyama, test_func, p, h; nsteps) for h in h_err]...
) .|> collect;

# %%
fig, ax = figax(yscale = log10, xscale = log10)
scatterlines!(ax, h_err, err; linewidth = 3, markersize = 25, linestyle = :dash, label = "Err")
scatterlines!(ax, h_err, err_an; linewidth = 3, markersize = 25, linestyle = :dash, label = "Err an")
err_n = [nsteps_error(EulerMaruyama(), sde, h, p, uti, test_func, nsteps) for (h, uti) in zip(h_err, ut)]
lines!(ax, h_err, err_n, label = "Absolute Sum")
fig

# %%
p_rest = (; u0 = 0.9, σ = 1.0, a = 1.0, b = 1.0)
p = (nens = 5000000, p_rest...)
sde = Phi4(p);
nsteps = 5
ut, err_an, err = zip(
    [local_error(sde, ABMaruyama, test_func, p, h; nsteps) for h in h_err]...
) .|> collect;

# %%
fig, ax = figax(yscale = log10, xscale = log10)
scatterlines!(ax, h_err, err; linewidth = 3, markersize = 25, linestyle = :dash, label = "Err")
scatterlines!(ax, h_err, err_an; linewidth = 3, markersize = 25, linestyle = :dash, label = "Err an")
err_n = [nsteps_error(ABMaruyama(), sde, h, p, uti, test_func, nsteps) for (h, uti) in zip(h_err, ut)]
lines!(ax, h_err, err_n, label = "Absolute Sum")
axislegend(ax, position = :rb)
fig
