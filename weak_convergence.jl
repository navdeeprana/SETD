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

# %%
function Sine(p)
    f(u, p) = sin(u)
    df(u, p) = cos(u)
    d2f(u, p) = -sin(u)
    d3f(u, p) = -cos(u)
    d4f(u, p) = sin(u)
    σ = p.σ
    return AdditiveSDE(f, df, d2f, d3f, d4f, σ, p)
end

# # Weak convergence

# %%
h_cvg = @. 1 / 2^(2:5)
p_rest = (; u0 = 0.5, tmax = 1.0, σ = 1.0)
p = (nens = 1000000000, p_rest...)
sde = Sine(p);

scale = 32
h_small = minimum(h_cvg) / scale
stats_an = solve_for_weak_convergence(sde, WeakOrder30, TwoPointWienerIncrement, p, h_small);

# %%
weak = (
    em = weak_convergence(sde, EulerMaruyama, SampledWienerIncrement, p, h_cvg, stats_an),
    ab = weak_convergence(sde, ABMaruyama, SampledWienerIncrement, p, h_cvg, stats_an),
    ab2 = weak_convergence(sde, ABMaruyama, SampledWienerIncrement, p, h_cvg, stats_an; anti = true),
    wo2 = weak_convergence(sde, WeakOrder20, SampledWienerIncrement, p, h_cvg, stats_an),
    wo3 = weak_convergence(sde, WeakOrder30, TwoPointWienerIncrement, p, h_cvg, stats_an),
);

# %%
using JLD2
jldsave("weakcvg_big.jld2"; weak)

# %%
fig, ax = figax(nx = 1, ny = 1, xscale = log2, yscale = log10, s = 130, xlabel = L"h")
plot_weak_convergence(ax, weak.em; label = "EM")
plot_weak_convergence(ax, weak.ab; label = "AB")
plot_weak_convergence(ax, weak.ab2; label = "AB2")
plot_weak_convergence(ax, weak.wo; label = "WO20")
lines!(ax, h_cvg, (@. 2 * h_cvg^1), linewidth = 3, color = :black)
lines!(ax, h_cvg, (@. 2 * h_cvg^2), linewidth = 3, color = :black)
axislegend(ax, position = :rb)
save("Sine_Weak_big.pdf", fig)
