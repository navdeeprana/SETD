# Create a makie theme for plotting.
function makietheme()
    return Theme(
        fontsize = 25,
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
            titlefont = :regular
        ),
        Lines = (linewidth = 3.0,),
        Scatter = (markersize = 20,)
    )
end

# Create a grid of axis to plot into.
function figax(; nx = 1, ny = 1, h = 5, a = 1.6, s = 100, fontsize = 24, sharex = false, sharey = false, kwargs...)
    (a > 1) ? size = (a * s * h * nx, s * h * ny) : size = (s * h * nx, s * h * ny / a)
    fig = Figure(; size = round.(Int, size), fontsize)
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
    errorbars!(ax, x, y, dy; color = p.color, whiskerwidth = 0.5*to_value(p.markersize)[1])
end

# General power law
power_law(x, x0, p, a) = @. a * (x / x0)^p

function plot_convergence(fig, ax, cvg, ps, pw; f = minimum)
    (; Δt, N, es, ew) = cvg
    scatterlines!(ax, Δt, es, label = "Strong", markersize = 20, linestyle = :dash)
    scatterlines!(ax, Δt, ew, label = "Weak", markersize = 20, linestyle = :dash)
    lines!(ax, Δt, power_law(Δt, f(Δt), ps, f(es)), color = :black)
    lines!(ax, Δt, power_law(Δt, f(Δt), pw, f(ew)), color = :red)
    axislegend(ax, position = :rb)
    ax.xlabel = "Δt"
end

plot_probability_distribution!(ax, X; bins = 256, kw...) = stephist!(ax, X; normalization = :pdf, bins, kw...)

function plot_normal_distribution!(ax, xm; μ = 0.0, σ = 1.0, kw...)
    x = LinRange(-xm, xm, 1000)
    P = @. exp(-((x-μ)^2 / (2*σ^2))) / sqrt(2π*σ^2)
    lines!(ax, x, P; label = "Normal", kw...)
end

function plot_boltzmann_distribution!(ax, pars, xm; kw...)
    x = LinRange(-xm, xm, 1000)
    P = @. exp(-(x^2 / 2 + pars.b * x^4 / 4) / pars.T)
    P = P / sum(P * (x[2] - x[1]))
    lines!(ax, x, P; label = "Boltzmann", kw...)
end
