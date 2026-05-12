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
# # Convergence for Additive SDEs

# %%
function Sine(p)
    f(u, p) = sin(u)
    df(u, p) = cos(u)
    d2f(u, p) = -sin(u)
    σ = p.σ
    return AdditiveSDE(f, df, d2f, σ, p)
end

# %% [markdown]
# # Convergence for the Additive SDE

# %%
h_cvg = @. 1 / 2^(3:6)
p_rest = (; u0 = 0.5, tmax = 1.0, σ = 1.0)
p = (nens = 50000, p_rest...)
sde = Sine(p);

# %%
t, W, u_an = solve_for_convergence(sde, StrongOrder15, p, h_cvg; noise_scale = 4, h_exact_scale = 32, return_noise_scale = 1);
@time cvg_so = convergence(sde, StrongOrder15, p, h_cvg, t, W, u_an)

# %%
fig, axes = figax(nx = 2, ny = 1, xscale = log2, yscale = log10, s = 130, xlabel = L"h")
axes[1].yticks = logticks(10.0, -7:1:1)
axes[2].yticks = logticks(10.0, -7:1:1)
axes[1].title = "Strong convergence"
axes[2].title = "Weak convergence"
plot_convergence_both(axes[1], axes[2], cvg_so; marker = :circle, color = colors[1], label = "SO1.5")
fit_and_plot(axes[1], cvg_so, :es, colors[1])
fit_and_plot(axes[2], cvg_so, :ew, colors[1])
axislegend.(axes, position = :rb, nbanks = 3)
resize_to_layout!(fig)
fig

# %%
h_cvg = @. 1 / 2^(3:6)
p_rest = (; u0 = 0.5, tmax = 1.0, σ = 1.0)
p = (nens = 400000, p_rest...)
sde = Sine(p);

# %%
t, W, u_an = solve_for_convergence(sde, StrongOrder15, p, h_cvg; noise_scale = 4, h_exact_scale = 32, return_noise_scale = 128);

# %%
@time cvg = (
    em = convergence(sde, EulerMaruyama, p, h_cvg, t, W, u_an),
    ab = convergence(sde, ABMaruyama, p, h_cvg, t, W, u_an),
    wo = convergence(sde, WeakOrder20, p, h_cvg, t, W, u_an),
);

# %%
fig, axes = figax(nx = 2, ny = 1, xscale = log2, yscale = log10, s = 130, xlabel = L"h")
axes[1].yticks = logticks(10.0, -7:1:1)
axes[2].yticks = logticks(10.0, -7:1:1)
axes[1].title = "Strong convergence"
axes[2].title = "Weak convergence"
plot_convergence_both(axes[1], axes[2], cvg.em, marker = :circle, color = colors[1], label = "EM")
plot_convergence_both(axes[1], axes[2], cvg.ab; marker = :circle, color = colors[2], label = "AB")
plot_convergence_both(axes[1], axes[2], cvg.wo; marker = :circle, color = colors[3], label = "WO")

fit_and_plot(axes[1], cvg.em, :es, colors[1])
fit_and_plot(axes[2], cvg.em, :ew, colors[1])

fit_and_plot(axes[1], cvg.ab, :es, colors[2])
fit_and_plot(axes[2], cvg.ab, :ew, colors[2])

fit_and_plot(axes[1], cvg.wo, :es, colors[3])
fit_and_plot(axes[2], cvg.wo, :ew, colors[3])

axislegend.(axes, position = :rb, nbanks = 3)
# save("Sine.png", fig)
resize_to_layout!(fig)
fig

# %%
p_rest = (; u0 = 0.5, tmax = 1.0, σ = 0.05)
p = (nens = 400000, p_rest...)
sde = Sine(p);

# %%
t, W, u_an = solve_for_convergence(sde, StrongOrder15, p, h_cvg; noise_scale = 4, h_exact_scale = 32, return_noise_scale = 128);

# %%
@time cvg = (
    ab = convergence(sde, ABMaruyama, p, h_cvg, t, W, u_an),
    wo = convergence(sde, WeakOrder20, p, h_cvg, t, W, u_an),
);

# %%
fig, axes = figax(nx = 2, ny = 1, xscale = log2, yscale = log10, s = 130, xlabel = L"h")
axes[1].yticks = logticks(10.0, -7:1:1)
axes[2].yticks = logticks(10.0, -7:1:1)
axes[1].title = "Strong convergence"
axes[2].title = "Weak convergence"
plot_convergence_both(axes[1], axes[2], cvg.ab; marker = :circle, color = colors[2], label = "AB")
plot_convergence_both(axes[1], axes[2], cvg.wo; marker = :circle, color = colors[3], label = "WO")

fit_and_plot(axes[1], cvg.ab, :es, colors[2])
fit_and_plot(axes[2], cvg.ab, :ew, colors[2])

fit_and_plot(axes[1], cvg.wo, :es, colors[3])
fit_and_plot(axes[2], cvg.wo, :ew, colors[3])

axislegend.(axes, position = :rb, nbanks = 3)
# save("Sine.png", fig)
resize_to_layout!(fig)
fig

# %%
function Phi4(p)
    f(u, p) = p.a * u - p.b * u^3
    df(u, p) = p.a - 3 * p.b * u^2
    d2f(u, p) = - 6 * p.b * u
    σ = p.σ
    return AdditiveSDE(f, df, d2f, σ, p)
end

# %%
h_cvg = @. 1 / 2^(3:6)

p_rest = (; u0 = 0.5, tmax = 1.0, a = 0.5, b = 0.2, σ = 1.0)
p = (nens = 100000, p_rest...)
sde = Phi4(p);

# %%
t, W, u_an = solve_for_convergence(sde, StrongOrder15, p, h_cvg; noise_scale = 4, h_exact_scale = 32, return_noise_scale = 128);

# %%
@time cvg = (
    em = convergence(sde, EulerMaruyama, p, h_cvg, t, W, u_an),
    ab = convergence(sde, ABMaruyama, p, h_cvg, t, W, u_an),
    wo = convergence(sde, WeakOrder20, p, h_cvg, t, W, u_an),
);

# %%
fig, axes = figax(nx = 2, ny = 1, xscale = log2, yscale = log10, s = 130, xlabel = L"h")
axes[1].yticks = logticks(10.0, -7:1:1)
axes[2].yticks = logticks(10.0, -7:1:1)
axes[1].title = "Strong convergence"
axes[2].title = "Weak convergence"
plot_convergence_both(axes[1], axes[2], cvg.em, marker = :circle, color = colors[1], label = "EM")
plot_convergence_both(axes[1], axes[2], cvg.ab, marker = :circle, color = colors[2], label = "AB")
plot_convergence_both(axes[1], axes[2], cvg.wo, marker = :circle, color = colors[3], label = "WO")

fit_and_plot(axes[1], cvg.em, :es, colors[1])
fit_and_plot(axes[2], cvg.em, :ew, colors[1])

fit_and_plot(axes[1], cvg.ab, :es, colors[2])
fit_and_plot(axes[2], cvg.ab, :ew, colors[2])

fit_and_plot(axes[1], cvg.wo, :es, colors[3])
fit_and_plot(axes[2], cvg.wo, :ew, colors[3])

axislegend.(axes, position = :rb)
# save("Phi4.png", fig)
resize_to_layout!(fig)
fig

# %%
p = (nens = 10000, p_rest...)
sde = Phi4(p);
t, W, u_an = solve_for_convergence(sde, StrongOrder15, p, h_cvg; noise_scale = 4, h_exact_scale = 32, return_noise_scale = 128);

# %% [markdown]
# # Weak convergence

# %%
p = (nens = 200000, p_rest...)
sde = Sine(p)
scale = 64
h_small = minimum(h_cvg) / scale
dW = wiener_increment_for_convergence(StrongOrder15(h_small).m, h_small, p.tmax)
stats_an = solve_for_weak_convergence(sde, StrongOrder15, dW, p, h_small);

# %%
args = (p, h_cvg, stats_an)
weak = (
    em = weak_convergence(sde, EulerMaruyama, SampledWienerIncrement, args...),
    ab = weak_convergence(sde, ABMaruyama, SampledWienerIncrement, args...),
    wo = weak_convergence(sde, WeakOrder20, SampledWienerIncrement, args...),
);

# %%
fig, ax = figax(nx = 1, ny = 1, xscale = log2, yscale = log10, s = 130, xlabel = L"h")
plot_weak_convergence(ax, weak.em; label = "EM")
plot_weak_convergence(ax, weak.ab; label = "AB")
plot_weak_convergence(ax, weak.wo; label = "WO20")
lines!(ax, h_cvg, (@. 1.0e-1 * h_cvg^1), linewidth = 3, color = :black)
# lines!(ax, h_cvg, (@. 1.e-2 * h_cvg^2), linewidth = 3, color = :black)
axislegend(ax, position = :rb)
fig

# %%
