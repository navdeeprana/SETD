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
# # Convergence for Additive SDEs

# %%
h_cvg = @. 1 / 2^(3:6)
p_rest = (; u0 = 0.5, tmax = 1.0, σ = 1.0, k = 1.0)
p = (nens = 50000, p_rest...)
test_func = test_atan()
sde = Sine(p);

# %%
t, W, u_an = solve_for_convergence(sde, StrongOrder15, p, h_cvg; noise_scale = 4, h_exact_scale = 32, return_noise_scale = 1);
@time cvg_so = convergence(sde, StrongOrder15, test_func, p, h_cvg, t, W, u_an);

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
p_rest = (; u0 = 0.5, tmax = 1.0, σ = 1.0, k = 1.0)
p = (nens = 1000000, p_rest...)
sde = Sine(p);

# %%
kwargs = (; noise_scale = 4, h_exact_scale = 32, return_noise_scale = 128)
t, W, u_an = solve_for_convergence(sde, StrongOrder15, p, h_cvg; kwargs...);

# %%
@show extrema(u_an)
P = pdf(u_an, -5:0.1:6)
lines(P.x, P.P)

# %%
@time cvg = (
    em = convergence(sde, EulerMaruyama, test_func, p, h_cvg, t, W, u_an),
    ab = convergence(sde, ABMaruyama, test_func, p, h_cvg, t, W, u_an),
    wo = convergence(sde, WeakOrder20, test_func, p, h_cvg, t, W, u_an),
);

# %%
fig, axes = figax(nx = 2, ny = 1, xscale = log2, yscale = log10, s = 130, xlabel = L"h")
axes[1].yticks = logticks(10.0, -7:1:1)
axes[2].yticks = logticks(10.0, -7:1:1)
axes[1].title = "Strong convergence"
axes[2].title = "Weak convergence"
plot_convergence_both(axes[1], axes[2], cvg_so, marker = :circle, color = :black, label = "SO15")
plot_convergence_both(axes[1], axes[2], cvg.em, marker = :circle, color = colors[1], label = "EM")
plot_convergence_both(axes[1], axes[2], cvg.ab; marker = :circle, color = colors[2], label = "AB")
plot_convergence_both(axes[1], axes[2], cvg.wo; marker = :circle, color = colors[3], label = "WO")

fit_and_plot(axes[1], cvg_so, :es, :black)
fit_and_plot(axes[2], cvg_so, :ew, :black)

fit_and_plot(axes[1], cvg.em, :es, colors[1])
fit_and_plot(axes[2], cvg.em, :ew, colors[1])

fit_and_plot(axes[1], cvg.ab, :es, colors[2])
fit_and_plot(axes[2], cvg.ab, :ew, colors[2])

fit_and_plot(axes[1], cvg.wo, :es, colors[3])
fit_and_plot(axes[2], cvg.wo, :ew, colors[3])

axislegend.(axes, position = :rb, nbanks = 4)
save("Sine.png", fig)
resize_to_layout!(fig)
fig

# %% [markdown]
# # Phi4

# %%
h_cvg = @. 1 / 2^(5:8)
p_rest = (; u0 = 0.9, tmax = 0.125, a = 1.0, b = 0.5, σ = 1.0)
p = (nens = 1000000, p_rest...)
test_func = test_atan();
sde = Phi4(p);

# %%
t, W, u_an = solve_for_convergence(sde, StrongOrder15, p, h_cvg; noise_scale = 4, h_exact_scale = 32, return_noise_scale = 128);

# %%
@show extrema(u_an)
P = pdf(u_an, -5:0.1:5)
lines(P.x, P.P)

# %%
@time cvg = (
    em = convergence(sde, EulerMaruyama, test_func, p, h_cvg, t, W, u_an),
    ab = convergence(sde, ABMaruyama, test_func, p, h_cvg, t, W, u_an),
    wo = convergence(sde, WeakOrder20, test_func, p, h_cvg, t, W, u_an),
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
save("Phi4.png", fig)
resize_to_layout!(fig)
fig

# %% [markdown]
# # Weak convergence

# %%
h_cvg = @. 1 / 2^(2:5)
p_rest = (; u0 = 0.5, tmax = 0.25, σ = 1.0, k = 1.0)
p = (nens = 10000000, p_rest...)
sde = Sine(p);
test_func = test_atan();

scale = 64
h_small = minimum(h_cvg) / scale
stats_an = solve_for_weak_convergence(sde, WeakOrder30, TwoPointWienerIncrement, test_func, (nens = 1000000, p_rest...), h_small);
stats_an

# %%
# For test function u^4, u0=0.5, sine
# stats_th = measurement(14.5488017, 0.000006)
# For test function atan(u)^2, u0=0.5, sine
# stats_th = measurement(0.80535759, 0.00000010)
# For test function atan(u)^2, u0=0.5, sine, tmax = 0.125
stats_th = measurement(0.38827177, 0.00001)

# %%
stats_ch = stats_th
weak = (
    # em = weak_convergence(sde, EulerMaruyama, SampledWienerIncrement, p, h_cvg, stats_ch),
    ab = weak_convergence(sde, ABMaruyama, SampledWienerIncrement, test_func, p, h_cvg, stats_ch),
    wo = weak_convergence(sde, WeakOrder20, SampledWienerIncrement, test_func, p, h_cvg, stats_ch),
    wo3 = weak_convergence(sde, WeakOrder30, TwoPointWienerIncrement, test_func, p, h_cvg, stats_ch),
);

# %%
# using JLD2
# jldsave("weakcvg.jld2"; weak)

# %%
fig, ax = figax(nx = 1, ny = 1, xscale = log2, yscale = log10, s = 130, xlabel = L"h")
# plot_weak_convergence(ax, weak.em; label = "EM")
plot_weak_convergence(ax, weak.ab; label = "AB")
# plot_weak_convergence(ax, weak.ab2; label = "AB2")
plot_weak_convergence(ax, weak.wo; label = "WO20")
plot_weak_convergence(ax, weak.wo3; label = "WO30")
lines!(ax, h_cvg, (@. 2 * h_cvg^1), linewidth = 3, color = :black)
lines!(ax, h_cvg, (@. 4 * h_cvg^2), linewidth = 3, color = :black)
axislegend(ax, position = :rb)
# save("Sine_Weak_5e7.pdf", fig)
fig

# %%
function ito_taylor(sde, test_func, p, h, un, dW; sign = 1)
    f, σ = sde.f, p.σ
    (; g, dg, d2g) = test_func
    fn, gn, dgn, d2gn = f(un, p), g(un), dg(un), d2g(un)
    return (
        gn + sign * (h * fn + σ * dW) * dgn + 0.5 * σ^2 * dW^2 * d2gn
    )
end

function ito_taylor_full(sde, test_func, p, h, un, du; sign = 1)
    f, σ = sde.f, p.σ
    (; g, dg, d2g) = test_func
    gn, dgn, d2gn = g(un), dg(un), d2g(un)
    return (
        gn + sign * du * dgn + 0.5 * du^2 * d2gn
    )
end

# %%
# Ito-Taylor expansion
h = 1.0e-4
p = (; u0 = 0.2, σ = 1.0, k = 1.0)
sde = Sine(p);
# test_func = test_poly(3);
test_func = test_atan();

int = EulerMaruyama(h)
dW = sqrt(h) * randn()
un = stepforward(int, sde, p.u0, dW)

ftrue = test_func.g(p.u0), test_func.g(un)

# %%
e1 = ito_taylor(sde, test_func, p, h, p.u0, dW)
e2 = ito_taylor(sde, test_func, p, h, un, dW; sign = -1)
fito = (e2, e1)

# %%
du = un - p.u0
e1 = ito_taylor_full(sde, test_func, p, h, p.u0, du)
e2 = ito_taylor_full(sde, test_func, p, h, un, du; sign = -1)
ffull = (e2, e1)

# %%
@. abs(fito - ftrue) |> println
@. abs(ffull - ftrue) |> println

# %%
h^(3 / 2), h^2
