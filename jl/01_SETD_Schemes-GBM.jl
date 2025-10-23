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
using Revise, Printf, CairoMakie, DataFrames, StatsBase, Random
includet("src/plotting.jl")
includet("src/brownian.jl")
includet("src/sde_examples.jl")
includet("src/solve.jl")
colors = Makie.wong_colors();
set_theme!(makietheme())
CairoMakie.enable_only_mime!("html")
Random.seed!(42);

# %%
function GBM(p)
    f(u, p) = p.a * u
    g(u, p) = p.b * u
    dg(u, p) = p.b
    return SDE(f, g, dg, p)
end

function GBM_SETD(p)
    f(u, p) = p.δ * u
    g(u, p) = p.b * u
    dg(u, p) = p.b
    return SDE(f, g, dg, p)
end

GBM_ANALYTICAL(p, t, W) = (; t, u = gbm_analytical.(p.u0, p.a, p.b, t, W))

# %%
p_rest = (u0 = 1.0, tmax = 2.0, a = 2.0, b = 1.0, nens = 1024, dt = 1/2^4)
p = (δ = 0.5, p_rest...)
dW = [SampledWeinerIncrement(p.dt, p.tmax) for _ in 1:p.nens]
t = 0:p.dt:p.tmax
W = map(dWi -> brownian_motion(dWi), dW);
sol_an = map(Wi -> GBM_ANALYTICAL(p, t, Wi), W);

# %%
sol_em = map(dWi -> solve(GBM(p), EulerMaruyama(p.dt), dWi, p.u0, p.tmax, p.dt), dW);
sol_ml = map(dWi -> solve(GBM(p), Milstein(p.dt), dWi, p.u0, p.tmax, p.dt), dW);
sol_ea = map(dWi -> solve(GBM_SETD(p), SETDEulerMaruyama(p.dt, p.a - p.δ), dWi, p.u0, p.tmax, p.dt), dW);
sol_eb = map(dWi -> solve(GBM_SETD(p), SETDMilstein(p.dt, p.a - p.δ), dWi, p.u0, p.tmax, p.dt), dW);

# %%
function trajectory_rms_error(sol, sol_an)
    dx2 = zero(sol[1].u)
    for (sa, sb) in zip(sol, sol_an)
        @. dx2 = dx2 + (sa.u - sb.u)^2
    end
    return sqrt.(dx2 ./ length(sol))
end
function _plot_sol!(ax, sol, n; kwargs...)
    lines!(ax, sol[n].t, sol[n].u; linewidth = 4, kwargs...)
    nothing
end
function _plot_rms_error!(ax, sol, sol_an; kwargs...)
    dx2 = trajectory_rms_error(sol, sol_an)
    lines!(ax, sol[1].t, dx2; kwargs...)
    nothing
end

# %%
n=10
fig, axes = figax(nx = 3, xlabel = L"t", xticks = 0:1:2)
axes[2].yticklabelsvisible = false
ax = axes[1]
ax.title="GBM trajectories"
ax.ylabel=L"u(t)"
_plot_sol!(ax, sol_em, n; color = colors[1], label = "EM")
_plot_sol!(ax, sol_ml, n; color = colors[2], label = "Milstein")
_plot_sol!(ax, sol_an, n; color = (:black, 0.7), label = "Analytical")

ax = axes[2]
ax.title="GBM trajectories"
_plot_sol!(ax, sol_ea, n; color = colors[3], label = "SETD-EM")
_plot_sol!(ax, sol_eb, n; color = colors[4], label = "SETD-Milstein")
_plot_sol!(ax, sol_an, n; color = (:black, 0.7), label = "Analytical")

ax = axes[3]
ax.limits = (nothing, nothing, 1.e-2, 1.e+2)
ax.yscale = log10
ax.title="RMS error with time"
ax.ylabel=L"\Delta(t)"
_plot_rms_error!(ax, sol_em, sol_an, label = "EM")
_plot_rms_error!(ax, sol_ml, sol_an, label = "Milstein")
_plot_rms_error!(ax, sol_ea, sol_an, label = "SETD-EM")
_plot_rms_error!(ax, sol_eb, sol_an, label = "SETD-Milstein")

axislegend(axes[1], position = :lb, nbanks = 3, labelsize = 28)
axislegend(axes[2], position = :lb, nbanks = 3, labelsize = 28)
axislegend(axes[3], position = :rb, nbanks = 2, labelsize = 28)
resize_to_layout!(fig)
save("figs/GBM_algorithm_comparison.pdf", fig)
fig

# %%
# fig, axes = figax(nx=4, ny=4, h=3)
# for (i, ax) in enumerate(axes)
#     n = 1+i
#     lines!(ax, sol_ml[n].t, sol_ml[n].u, )
#     lines!(ax, sol_ea[n].t, sol_ea[n].u, )
#     lines!(ax, sol_an[n].t, sol_an[n].u, color=:black)
#     ax.title = string(n)
# end
# resize_to_layout!(fig)
# fig

# %% [markdown]
# ## Convergence for the GBM

# %%
p_rest = (u0 = 1.0, tmax = 1.0, a = 2.0, b = 1.0, nens = 50000, saveat = 1/2^1)
p = (δ = 0.5, p_rest...)
h_cvg = @. 1 / 2^(5:10)
t, W = brownian_motion(minimum(h_cvg), p.tmax, p.nens);
tn, Wn = tnWn(t, W, p.saveat)
sol_an = map(Wni -> gbm_analytical.(p.a, p.b, tn, Wni), eachcol(Wn));
mean_an = @. p.u0 * exp(p.a*tn);

# %%
_SETDEulerMaruyama(h) = SETDEulerMaruyama(h, p.a-p.δ)
_SETDMilstein(h) = SETDMilstein(h, p.a-p.δ)
cvg1 = (
    # em = convergence(GBM(p), EulerMaruyama, p, h_cvg, t, W, sol_an, mean_an),
    # ml = convergence(GBM(p), Milstein, p, h_cvg, t, W, sol_an, mean_an),
    etdem = convergence(GBM_SETD(p), _SETDEulerMaruyama, p, h_cvg, t, W, sol_an, mean_an),
    etdml = convergence(GBM_SETD(p), _SETDMilstein, p, h_cvg, t, W, sol_an, mean_an)
);

# %%
p = (δ = 1.0, p_rest...)
_SETDEulerMaruyama(h) = SETDEulerMaruyama(h, p.a-p.δ)
_SETDMilstein(h) = SETDMilstein(h, p.a-p.δ)
cvg2 = (
    # em = convergence(GBM(p), EulerMaruyama, p, h_cvg, t, W, sol_an),
    # ml = convergence(GBM(p), Milstein, p, h_cvg, t, W, sol_an),
    etdem = convergence(GBM_SETD(p), _SETDEulerMaruyama, p, h_cvg, t, W, sol_an, mean_an),
    etdml = convergence(GBM_SETD(p), _SETDMilstein, p, h_cvg, t, W, sol_an, mean_an)
);

# %%
function plot_convergence(fig, ax1, ax2, cvg; kwargs...)
    g = groupby(cvg, :t)[end]
    (; h, es, ew, ew_an) = g
    kw = (markersize = 25, linestyle = :dash, linewidth = 3)
    scatterlines!(ax1, h, es; kw..., kwargs...)
    scatterlines!(ax2, h, ew; kw..., kwargs...)
end

# %%
fig, axes = figax(nx = 2, ny = 2, xscale = log2, s = 130, yscale = log2, xlabel = L"h")
axes[1].title = "Strong convergence for SETD-EM"
axes[2].title = "Weak convergence for SETD-EM"
axes[3].title = "Strong convergence for SETD-Milstein"
axes[4].title = "Weak convergence for SETD-Milstein"
plot_convergence(fig, axes[1], axes[2], cvg1.etdem, marker = :circle, label = L"$\delta = 0.5$")
plot_convergence(fig, axes[1], axes[2], cvg2.etdem, marker = :rect, label = L"\delta = 1.0")
plot_convergence(fig, axes[3], axes[4], cvg1.etdml, marker = :circle, label = L"\delta = 0.5")
plot_convergence(fig, axes[3], axes[4], cvg2.etdml, marker = :rect, label = L"\delta = 1.0")
# axes[3].yticks = @. 2^(-8:-2:-2)
for (ax, a, n, text, yf) in zip(axes, [3, 2.5, 5, 2.5], [0.5, 1.0, 1.0, 1.0], [L"h^{1/2}", L"h", L"h", L"h"], [0.75, 0.55, 0.55, 0.55])
    lines!(ax, h_cvg, (@. a * h_cvg^n), linewidth = 3, color = :black)
    x = (h_cvg[3] + h_cvg[4])/2
    text!(ax, x, yf*a*(x^n); text, fontsize = 30)
end
axislegend.(axes, position = :rb)
resize_to_layout!(fig)
save("figs/GBM_convergence.pdf", fig)
fig

# %% [markdown]
# # Stability

# %%
dt = 1.e-2
p_rest = (u0 = 1.0, tmax = 3.0, a = -3.0, b = sqrt(2.0), δ = 0.5, nens = 10000)
p = (; dt = dt, p_rest...)
t, W = brownian_motion(dt, p.tmax, p.nens);
uvar = gbm_var.(p.u0, p.a, p.b, t);
uan = map(Wi -> gbm_analytical.(p.u0, p.a, p.b, t, Wi), eachcol(W));

# %%
fig, ax = figax(h = 5)
lines!(ax, t, uvar)
lines!(ax, t, var(uan))
fig

# %%
isGBMstable(p, h) = p.a + 0.5*p.b^2 < 0
isEMstableforGBM(p, h) = abs(1+h*p.a)^2 + h*abs(p.b)^2 < 1
function isSETDEMstableforGBM(p, h)
    (; a, b, δ) = p
    c = a - δ
    return (exp(c*h) + (δ/c)*(exp(c*h)-1))^2 + h*(b*exp(c*h))^2 < 1
end

# %%
fig, ax = figax(yscale = Makie.pseudolog10, xlabel = "t", ylabel = "Var[u(t)]")
ax.limits = (nothing, nothing, -0.1, 2)
for dt in [1.e-2, 1.e-1, 4.5e-1, 5.0e-1]
    p = (; dt = dt, p_rest...)
    dW = [SampledWeinerIncrement(p.dt, p.tmax) for _ in 1:p.nens]
    sol = map(dWi -> solve(GBM(p), EulerMaruyama(p.dt), dWi, p.u0, p.tmax, p.dt), dW);
    l = @sprintf "h=%.2f, Stable=%s" dt string(isEMstableforGBM(p, dt))
    lines!(ax, sol[1].t, var([s.u for s in sol]), label = l)
end
t = 0.0:0.01:p_rest.tmax
uvar = gbm_var.(p_rest.u0, p_rest.a, p_rest.b, t)
lines!(ax, t, uvar, color = :black)
axislegend(ax)
fig

# %%
fig, ax = figax(yscale = Makie.pseudolog10)
for dt in [1.e-2, 1.e-1, 4.5e-1, 5.0e-1]
    p = (; dt = dt, p_rest...)
    dW = [SampledWeinerIncrement(p.dt, p.tmax) for _ in 1:p.nens]
    sol = map(dWi -> solve(GBM_SETD(p), SETDEulerMaruyama(p.dt, p.a - p.δ), dWi, p.u0, p.tmax, p.dt), dW);
    lines!(ax, sol[1].t, var([s.u for s in sol]))
end
t = 0.0:0.01:p_rest.tmax
uvar = gbm_var.(p_rest.u0, p_rest.a, p_rest.b, t)
lines!(ax, t, uvar, color = :black)
fig

# %%
stability_GBM(x, y, p) = @. 1+x+y/2
stability_EM(x, y, p) = @. (1+x)^2 + y
stability_Milstein(x, y, p) = @. (1+x)^2 + y + y^2/2

function stability_terms(x, y, p)
    z = p.z
    c, δ = z * p.a, (1-z) * p.a
    t1 = @. ((z-1+exp(z*x))^2)/z^2
    t2 = @. y * exp(z*x)
    t3 = @. y^2 * exp(z*x)/2
    t4 = @. (exp(2*z*x)-1)*(y/(2*z*x))
    return t1, t2, t3, t4
end

function stability_SETDEM(x, y, p)
    t = stability_terms(x, y, p)
    return @. t[1] + t[2]
end

function stability_SETDMilstein(x, y, p)
    t = stability_terms(x, y, p)
    return @. t[1] + t[2] + t[3]
end

function stability_SETD1(x, y, p)
    t = stability_terms(x, y, p)
    return @. t[1] + t[4]
end

# %%
function stability_region_plot!(ax, f, x, y, p, color; alpha = 0.1, kw...)
    F = f(x, y', p)
    Z = @. F < 1
    contour!(ax, x, y, F, levels = [1]; linewidth = 5, color, kw...)
    contourf!(ax, x, y, Z; colormap = [:transparent, (color, alpha)])
end

# %%
fig, axes = figax(nx = 3, ny = 1, xlabel = L"$\lambda h$", xticks = -5:1:1, yticks = 0:2:6)
axes[1].ylabel = L"$\mu^2 h$"
axes[2].yticklabelsvisible=false
axes[3].yticklabelsvisible=false

x = -6:0.05:1.0
y = 0:0.05:7

ax = axes[1]
stability_region_plot!(ax, stability_GBM, x, y, p, :black; alpha = 0.05)
stability_region_plot!(ax, stability_EM, x, y, p, colors[1])
stability_region_plot!(ax, stability_Milstein, x, y, p, colors[2])

for (ax, z) in zip(axes[2:end], [0.3, 0.5])
    p = (; a = -2.0, z = z)
    stability_region_plot!(ax, stability_GBM, x, y, p, :black; alpha = 0.05)
    stability_region_plot!(ax, stability_SETDEM, x, y, p, colors[1]; label = "SETD-EM")
    stability_region_plot!(ax, stability_SETDMilstein, x, y, p, colors[2]; label = "SETD-Milstein")
    stability_region_plot!(ax, stability_SETD1, x, y, p, colors[3]; label = "SETD1")
end

points = Point2f[(0, 0.25), (1, 0.25), (1, 0.75), (0, 0.75)]
e1 = PolyElement(; color = (:black, 0.05), strokecolor = (:black), strokewidth = 3, points)
e2 = PolyElement(; color = (colors[1], 0.05), strokecolor = (colors[1]), strokewidth = 3, points)
e3 = PolyElement(; color = (colors[2], 0.05), strokecolor = (colors[2]), strokewidth = 3, points)
e4 = PolyElement(; color = (colors[3], 0.05), strokecolor = (colors[3]), strokewidth = 3, points)

kw = (; tellheight = false, tellwidth = false, halign = :right, valign = :top, patchsize = (35, 35), rowgap = 10, margin = (10, 10, 10, 10))

Legend(fig[1, 1], [e1, e2, e3], ["Analytical", "EM", "Milstein"]; kw...)
Legend(fig[1, 2], [e2, e3, e4], ["SETD-EM", "SETD-Milstein", "SETD1"], L"$c=0.3\lambda$"; kw...)
Legend(fig[1, 3], [e2, e3, e4], ["SETD-EM", "SETD-Milstein", "SETD1"], L"$c=0.5\lambda$"; kw...)
resize_to_layout!(fig)
save("figs/GBM_stability.pdf", fig)
fig

# %%
