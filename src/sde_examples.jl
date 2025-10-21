# Some common SDE examples and algorithms.
# Equations are described in Kloeden 1999 Section 4.4 and the code follows the same notation.

############## Linear SDEs with Additive Noise (Equation 4.2) ##############
# dx = ax(t)dt + bdt + cdW

function linearsde_analytical!(x, x0, a, b, c, t, W)
    x[1] = x0
    ito = 0.0
    for i in 2:length(W)
        at = a*t[i]
        ito = ito + exp(-at) * (W[i] - W[i-1])
        x[i] = exp(at) * (x0 + (b/a) * (1 - exp(-at)) + c * ito)
    end
    nothing
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

gbm_var(x0, a, b, t) = x0^2*exp(2*a*t)*(exp(b^2*t)-1)
