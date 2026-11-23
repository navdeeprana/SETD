# Some common SDE examples and algorithms.
# Equations are described in Kloeden 1999 Section 4.4 and the code follows the same notation.

############## Linear SDEs with Additive Noise (Equation 4.2) ##############
# dx = ax(t)dt + bdt + cdW

function linearsde_analytical!(x, x0, a, b, c, t, W)
    x[1] = x0
    ito = 0.0
    for i in 2:length(W)
        at = a * t[i]
        ito = ito + exp(-at) * (W[i] - W[i - 1])
        x[i] = exp(at) * (x0 + (b / a) * (1 - exp(-at)) + c * ito)
    end
    return nothing
end

function linearsde_analytical(x0, a, b, c, t, W)
    x = zero(W)
    linearsde_analytical!(x, x0, a, b, c, t, W)
    return x
end

############## OU process (maps to 4.2 with a=-k, b=0, and c=sqrt(2D)) ##############
# dx = -kx(t)dt + sqrt(2D)dW

OU_analytical(x0, k, D, t, W) = linearsde_analytical(x0, -k, 0, sqrt(2D), t, W)

############## Geometric Brownian Motion (Equation 4.6) ##############
# dx = ax(t)dt + bx(t)dW

gbm_analytical(x0, a, b, t, W) = x0 * exp((a - b^2 / 2) * t + b * W)
gbm_analytical(a, b, t, W) = gbm_analytical(1, a, b, t, W)

gbm_var(x0, a, b, t) = x0^2 * exp(2 * a * t) * (exp(b^2 * t) - 1)

## Additive SDES ##

# du = sin(ku) dt + σ dW(t)
function Sine(p)
    f(u, p) = sin(p.k * u)
    df(u, p) = p.k * cos(p.k * u)
    d2f(u, p) = -p.k^2 * sin(p.k * u)
    d3f(u, p) = -p.k^3 * cos(p.k * u)
    d4f(u, p) = +p.k^4 * sin(p.k * u)
    σ = p.σ
    return AdditiveSDE(f, df, d2f, d3f, d4f, σ, p)
end

# du = (au-bu^3) dt + σ dW(t)
function Phi4(p)
    f(u, p) = p.a * u - p.b * u^3
    df(u, p) = p.a - 3 * p.b * u^2
    d2f(u, p) = - 6 * p.b * u
    d3f(u, p) = - 6 * p.b
    d4f(u, p) = 0
    σ = p.σ
    return AdditiveSDE(f, df, d2f, d3f, d4f, σ, p)
end
