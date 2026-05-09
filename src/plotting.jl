using CairoMakie, Measurements, LsqFit, Printf

# Create a makie theme for plotting.
function makietheme()
    theme = Theme(
        fontsize = 30,
        Axis = (
            backgroundcolor = :transparent,
            xgridvisible = false,
            ygridvisible = false,
            xlabelpadding = 3,
            ylabelpadding = 3,
            xtickalign = 1,
            ytickalign = 1,
            xminorticksvisible = true,
            yminorticksvisible = true,
            xminortickalign = 1,
            yminortickalign = 1,
            titlefont = :regular,
        ),
        Lines = (linewidth = 3.0,),
        Scatter = (markersize = 20,)
    )
    return merge(theme_latexfonts(), theme)
end

# Create a grid of axis to plot into.
function figax(; nx = 1, ny = 1, h = 5, a = 1.6, s = 100, sharex = false, sharey = false, kwargs...)
    (a > 1) ? size = (a * s * h * nx, s * h * ny) : size = (s * h * nx, s * h * ny / a)
    fig = Figure(; size = round.(Int, size))
    ax = [Axis(fig[j, i]; aspect = AxisAspect(a), kwargs...) for i in 1:nx, j in 1:ny]
    for i in 1:nx
        colsize!(fig.layout, i, Aspect(1, a))
    end
    resize_to_layout!(fig)
    (nx * ny == 1) ? ax = ax[1] : nothing
    return fig, ax
end

function errorscatter!(ax, x, y, dy; kw...)
    p = scatter!(ax, x, y; kw...)
    errorbars!(ax, x, y, dy; color = p.color, whiskerwidth = 0.5 * to_value(p.markersize)[1])
end

# General power law
power_law(x, p, y0; x0 = minimum(x)) = @. y0 * (x / x0)^p

plot_probability_distribution!(ax, X; bins = 256, kw...) = stephist!(ax, X; normalization = :pdf, bins, kw...)

function plot_normal_distribution!(ax, xm; μ = 0.0, σ = 1.0, kw...)
    x = LinRange(-xm, xm, 1000)
    P = @. exp(-((x - μ)^2 / (2 * σ^2))) / sqrt(2π * σ^2)
    return lines!(ax, x, P; label = "Normal", kw...)
end

function plot_boltzmann_distribution!(ax, pars, xm; kw...)
    x = LinRange(-xm, xm, 1000)
    P = boltzmann_distribution(x, pars)
    return lines!(ax, x, P; label = "Boltzmann", kw...)
end

function _plot_sol!(ax, sol, n; kwargs...)
    lines!(ax, sol[n].t, sol[n].u; linewidth = 4, kwargs...)
    return nothing
end
function _plot_rms_error!(ax, sol, sol_an; kwargs...)
    dx2 = trajectory_rms_error(sol, sol_an)
    lines!(ax, sol[1].t, dx2; kwargs...)
    return nothing
end

function abc(axes)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return [L"\textbf{(%$(l))}" for (_, l) in zip(axes, alphabet)]
end

function logticks(base, range)
    return collect(float(base) .^ range), [L"{%$base}^{%$i}" for i in range]
end

function fitxy(x, y, model, p0)
    f = curve_fit(model, x, y, p0)
    c, e = coef(f), standard_errors(f)
    xu = sort(unique(x))
    yu = model(xu, c)
    return (; f, c, e, x = xu, y = yu)
end

# Some common functions to fit
_linefunc(x, p) = @. p[1] + p[2] * x
_powerlawfunc(x, p) = @. p[1] * x^p[2]

fitline(x, y; p0 = [minimum(y), 1.0]) = fitxy(x, y, _linefunc, p0);
fitpowerlaw(x, y; p0 = [minimum(y), 1.0]) = fitxy(x, y, _powerlawfunc, p0);

function fit_and_plot(ax, cvg, s, color)
    f = fitpowerlaw(cvg.h, Measurements.value.(cvg[s]))
    lines!(ax, f.x, f.y; linewidth = 3, color, label=(@sprintf "%.2f" f.c[2]))
end

function plot_convergence(ax, h, c; error = false, kwargs...)
    kw = (markersize = 25, linestyle = :dash, linewidth = 3)
    y, dy = Measurements.value.(c), Measurements.uncertainty.(c)
    scatterlines!(ax, h, y; kw..., kwargs...)
    if error
        rangebars!(ax, h, y .- dy, y .+ dy)
    end
end

plot_strong_convergence(ax, cvg; kwargs...) = plot_convergence(ax, cvg.h, cvg.es; kwargs...)
plot_weak_convergence(ax, cvg; kwargs...) = plot_convergence(ax, cvg.h, cvg.ew; kwargs...)

function plot_convergence_both(ax1, ax2, cvg; kwargs...)
    (; h, es, ew) = cvg
    plot_convergence(ax1, h, es; kwargs...)
    plot_convergence(ax2, h, ew; kwargs...)
end