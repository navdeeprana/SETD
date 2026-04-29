function probability_distribution(sol; bins = -9:0.2:9)
    h = OnlineStats.Hist(bins)
    for si in sol
        @views u = si.u[2:end]
        fit!(h, u)
    end
    hn = normalize(Histogram(h.edges, h.counts))
    return (; x = midpoints(h), P = hn.weights)
end

function boltzmann_distribution(x, pars)
    P = @. exp(-(x^2 / 2 + pars.b * x^4 / 4) / pars.T)
    P = P / sum(P * (x[2] - x[1]))
    return P
end

function trajectory_rms_error(sol, sol_an)
    dx2 = zero(sol[1].u)
    for (sa, sb) in zip(sol, sol_an)
        @. dx2 = dx2 + (sa.u - sb.u)^2
    end
    return sqrt.(dx2 ./ length(sol))
end
