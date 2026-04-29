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

# %% [markdown]
# # Stochastic Anharmonic Oscillator

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

# %%
function SAO(p)
    f(u, p) = -p.Γ * (u + p.b * u^p.z)
    df(u, p) = -p.Γ * (1 + p.z * p.b * u^(p.z - 1))
    d2f(u, p) = -p.Γ * p.z * (p.z - 1) * p.b * u^(p.z - 2)
    g(u, p) = sqrt(2 * p.Γ * p.T)
    return AdditiveSDE(f, df, d2f, g, p)
end

function SAO_SETD(p)
    f(u, p) = -p.Γ * p.b * u^p.z
    df(u, p) = -p.Γ * p.z * p.b * u^(p.z - 1)
    d2f(u, p) = -p.Γ * p.z * (p.z - 1) * p.b * u^(p.z - 2)
    g(u, p) = sqrt(2 * p.Γ * p.T)
    return AdditiveSDE(f, df, d2f, g, p)
end

# %%
p_rest = (; u0 = 0.0, tmax = 20.0, nens = 50000, T = 6.0, Γ = 5.0, b = 1.0e-2, z = 3, saveat = 0.2, save_after = 2.0);

# %%
h_scan = [1.0e-2, 2.0e-2, 5.0e-2, 1.0e-1, 2.0e-1]
data = Dict()
for (nh, h) in enumerate(h_scan)
    p = (; dt = h, p_rest...)
    dW = [SampledWienerIncrement(p.dt, p.tmax) for _ in 1:p.nens]
    args, kwargs = (p.u0, p.tmax, p.saveat), (; save_after = p.save_after)
    data[nh] = (;
        em = map(dWi -> solve(SAO(p), EulerMaruyama(p.dt), dWi, args...; kwargs...), dW),
        setdem = map(dWi -> solve(SAO_SETD(p), SETDEulerMaruyama(p.dt, -p.Γ, 0.5), dWi, args...; kwargs...), dW),
        setd1 = map(dWi -> solve(SAO_SETD(p), SETD1(p.dt, -p.Γ), dWi, args...; kwargs...), dW),
        ifem = map(dWi -> solve(SAO_SETD(p), IFEulerMaruyama(p.dt, -p.Γ), dWi, args...; kwargs...), dW),
    )
end

# %%
function _plot_this(ax, d1, d2)
    P = probability_distribution(d1)
    lines!(ax, P.x, P.P; linewidth = 5, label = "h=0.2")
    P = probability_distribution(d2)
    return lines!(ax, P.x, P.P; linewidth = 5, label = "h=0.01")
end

fig, axes = figax(
    nx = 3, xlabel = L"$u$",
    limits = (-9, 9, -0.01, 0.23),
    xticks = -8:4:8, yticks = [0.0, 0.1, 0.2]
)

axes[1].title = ("EM")
axes[1].ylabel = L"$P(u)$"
_plot_this(axes[1], data[5].em, data[1].em)

axes[2].title = ("SETD-EM")
axes[2].yticklabelsvisible = false
_plot_this(axes[2], data[5].setdem, data[1].setdem)

axes[3].title = ("SETD1")
axes[3].yticklabelsvisible = false
_plot_this(axes[3], data[5].setd1, data[1].setd1)

for ax in axes
    plot_boltzmann_distribution!(ax, p_rest, 9.0; color = :black, linewidth = 3, linestyle = :dash)
end

axislegend.(axes; patchsize = (35, 20))
resize_to_layout!(fig)
# save("figs/SAO_probability.pdf", fig)
fig

# %%
fig, axes = figax(
    nx = 2, ny = 2, xlabel = L"$u$",
    limits = (-9, 9, -0.01, 0.23),
    xticks = -8:4:8, yticks = [0.0, 0.1, 0.2]
)

axes[1].title = ("EM")
axes[1].ylabel = L"$P(u)$"
_plot_this(axes[1], data[5].em, data[1].em)

axes[2].title = ("SETD-EM")
axes[2].yticklabelsvisible = false
_plot_this(axes[2], data[5].setdem, data[1].setdem)

axes[3].title = ("IF-EM")
axes[3].ylabel = L"$P(u)$"
_plot_this(axes[3], data[5].ifem, data[1].ifem)

axes[4].title = ("SETD1")
axes[4].yticklabelsvisible = false
_plot_this(axes[4], data[5].setd1, data[1].setd1)

for ax in axes
    plot_boltzmann_distribution!(ax, p_rest, 9.0; color = :black, linewidth = 3, linestyle = :dash)
end

axislegend.(axes; patchsize = (35, 20))
resize_to_layout!(fig)
fig

# %%
function error_in_distribution(sol, pars)
    P = probability_distribution(sol)
    B = boltzmann_distribution(P.x, pars)
    dP = @. P.P - B
    rmse = sqrt(mean(dP .^ 2))
    return dP, rmse
end

kw = (markersize = 25, linestyle = :dash, linewidth = 3)

fig, ax = figax(xscale = log10, yscale = log10, xlabel = L"h")

sols = [data[n].em for n in 1:5]
rmse = [error_in_distribution(sol, p_rest)[2] for sol in sols]
scatterlines!(ax, h_scan, rmse; kw..., label = "EM")

sols = [data[n].setdem for n in 1:5]
rmse = [error_in_distribution(sol, p_rest)[2] for sol in sols]
scatterlines!(ax, h_scan, rmse; kw..., label = "SETD-EM")

sols = [data[n].setd1 for n in 1:5]
rmse = [error_in_distribution(sol, p_rest)[2] for sol in sols]
scatterlines!(ax, h_scan, rmse; kw..., label = "SETD1")

sols = [data[n].ifem for n in 1:5]
rmse = [error_in_distribution(sol, p_rest)[2] for sol in sols]
scatterlines!(ax, h_scan, rmse; kw..., label = "IF-EM")

ax.title = "RMS error for the probability distribution"
axislegend(ax, position = :lt)

fig

# %% [markdown]
# # Convergence for the SAO

# %%
p_rest = (; u0 = 3.0, tmax = 1.0, T = 0.1, Γ = 1.0, b = 1.0e-1, z = 2);
scale = 32
h_cvg = @. 1 / 2^(2:8)

# First use a StrongOrder15 solution as an approximation
h_small = minimum(h_cvg) / scale
h_analytical = 2 * h_small
p = (; nens = 50000, p_rest..., saveat = p_rest.tmax)
args = (p.u0, p.tmax, p.saveat)
t, W = wiener_process(h_small, p.tmax, p.nens)

sde_an, int = SAO(p), StrongOrder15(h_analytical)
dW = wiener_increment_for_convergence(int.m, h_analytical, p.tmax)
sol_an = @showprogress map(
    Wi -> begin
        resetdW!(dW, t, Wi, h_analytical)
        solve(sde_an, int, dW, args..., save_after = 0.5 * p.saveat)
    end,
    eachcol(W)
);

# %%
_SETDEulerMaruyama(h) = SETDEulerMaruyama(h, -p.Γ)
_SETD1(h) = SETD1(h, -p.Γ)
_SETD2(h) = SETD2(h, -p.Γ)
_IFEulerMaruyama(h) = IFEulerMaruyama(h, -p.Γ)

sde = SAO_SETD(p)
args = (p, h_cvg, t, W, sol_an)
cvg = (
    # em = convergence(sde_an, EulerMaruyama, args...),
    ab = convergence(sde_an, ABMaruyama, args...),
    wo2 = convergence(sde_an, WeakOrder20, args...),
    # so15 = convergence(sde_an, StrongOrder15, args...),
    # etdem = convergence(sde, _SETDEulerMaruyama, args...),
    # ifem = convergence(sde, _IFEulerMaruyama, args...),
    setd1 = convergence(sde, _SETD1, args...),
    setd2 = convergence(sde, _SETD2, args...),
);

# %%
fig, axes = figax(nx = 2, ny = 1, xscale = log2, s = 130, yscale = log2, xlabel = L"h")
axes[1].yticks = (collect(2.0 .^ (-10:2:4)), [L"2^{%$i}" for i in -10:2:4])
axes[2].yticks = (collect(2.0 .^ (-10:2:4)), [L"2^{%$i}" for i in -10:2:4])
axes[1].title = "Strong convergence for SAO"
axes[2].title = "Weak convergence for SAO"
# plot_convergence(fig, axes[1], axes[2], cvg.em, marker = :circle, label = "EM")
plot_convergence(fig, axes[1], axes[2], cvg.ab, marker = :circle, label = "AB")
plot_convergence(fig, axes[1], axes[2], cvg.wo2, marker = :circle, label = "WO2")
# plot_convergence(fig, axes[1], axes[2], cvg.so15, marker = :circle, label = "SO1.5")
# plot_convergence(fig, axes[1], axes[2], cvg.etdem, marker = :circle, label = "SETD-EM")
# plot_convergence(fig, axes[1], axes[2], cvg.ifem; ignore_es = true, marker = :circle, label = "IF-EM")
plot_convergence(fig, axes[1], axes[2], cvg.setd1; ignore_es = true, marker = :circle, label = "SETD1")
plot_convergence(fig, axes[1], axes[2], cvg.setd2; ignore_es = true, marker = :circle, label = "SETD2")
for (ax, a, n, text, yf) in zip(axes, [1.5, 1.5], [1.0, 1.0], [L"h", L"h"], [1.2, 1.2])
    lines!(ax, h_cvg, (@. a * h_cvg^n), linewidth = 3, color = :black)
    x = (h_cvg[3] + h_cvg[4]) / 2
    text!(ax, x, yf * a * (x^n); text, fontsize = 30)
end
lines!(axes[2], h_cvg, (@. 2.0e-1 * (h_cvg)^2.0), linewidth = 3, color = :black)
axislegend.(axes, position = :rb)
resize_to_layout!(fig)
save("Untitled.png", fig)
fig

# %%
# p = (nens = 10000, p_rest...)
# weak_cvg = (;
#     ifem = weak_convergence(sde, sde_an, _IFEulerMaruyama, FixedWienerIncrement, p, h_cvg),
#     etd1 = weak_convergence(sde, sde_an, _SETD1, FixedWienerIncrement, p, h_cvg),
#     etd2 = weak_convergence(sde, sde_an, _SETD2, FixedWienerIncrement, p, h_cvg),
# );

# %%
# fig, ax = figax(nx = 1, ny = 1, xscale = log2, s = 130, yscale = log2, xlabel = L"h")
# # plot_convergence(fig, ax, ax, weak_cvg.etd1; ignore_es = true, marker = :circle, label = "SETD1")
# plot_convergence(fig, ax, ax, weak_cvg.etd2; ignore_es = true, marker = :circle, label = "SETD2")
# lines!(ax, h_cvg, (@. 1 * h_cvg^1), linewidth = 3, color = :black)
# lines!(ax, h_cvg, (@. 1 * h_cvg^2), linewidth = 3, color = :black)
# axislegend(ax, position = :rb)
# fig
