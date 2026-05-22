# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import logging
logging.disable()

import numpy as np
import dedalus.public as d3
from scipy.interpolate import BarycentricInterpolator

import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = False

# %%
def smooth_step_cinf(s):
    # 0 for s <= 0, 1 for s >= 1, smoothly increases from 0 to 1 on (0,1)
    out = np.zeros_like(s, dtype=np.float64)
    out[s >= 1] = 1.0
    m = (s > 0) & (s < 1)
    a, b = np.exp(-1/s[m]), np.exp(-1/(1-s[m]))
    out[m] = a / (a + b)
    return out

def flattenF(u, F, R1 = 0.75, R2 = 0.95):
    L = 0.5*(np.max(u) - np.min(u))
    r, R1, R2 = np.abs(u), L*R1, L*R2
    s = (r - R1) / (R2 - R1)
    θ = smooth_step_cinf(s)
    return (1-θ) * F(r) + θ * F(R2)

# %%
F = lambda u : u**2
u = np.linspace(-10,10,128,endpoint=False)
fig, ax = plt.subplots()
ax.plot(u, F(u))
ax.plot(u, flattenF(u, F))

# %%
def KBE(L, N, tmax, a, F, σ = 1.0, h = 5.e-4, dealias=2, flatten=True):
    coord = d3.Coordinate('u')
    dist = d3.Distributor(coord, dtype=np.float64)
    basis = d3.Chebyshev(coord, size=N, bounds=(-L, L), dealias=dealias)
    lift_basis = basis.derivative_basis(2)
    lift = lambda A, n : d3.Lift(A, lift_basis, n)
    du = lambda f: d3.Differentiate(f, coord)

    f = dist.Field(name='f', bases=basis)
    t1 = dist.Field(name='t1')
    t2 = dist.Field(name='t2')

    u = dist.local_grid(basis)
    au = dist.Field(name='Fu', bases=basis)
    au['g'] = a(u)

    problem = d3.IVP([f, t1, t2], namespace=locals())
    problem.add_equation("dt(f) - (σ**2/2)*du(du(f)) + lift(t1, -1) + lift(t2, -2) - au*du(f) = 0")
    problem.add_equation("du(f)(u='left') = 0")
    problem.add_equation("du(f)(u='right') = 0")

    if flatten:
        f['g'] = flattenF(u, F)
    else:
        f['g'] = F(u)

    solver = problem.build_solver(d3.SBDF4)
    solver.stop_sim_time = tmax
    while solver.proceed:
        solver.step(h)

    f.change_scales(1)

    return u, f, np.copy(f['g'])

def evaluateF(u, f, u0):
    f0 = f(u=u0).evaluate()['g'][0]
    print(f"E[F(u(T)) | u(0)={u0}] = {f0:.10f}")
    return u0, f0

# %%
# Check for OU process
L, n, tmax = 5, 1024, 1.0

a = lambda u : -u
F = lambda u : u**2

def fexact(u0, T):
    return F(u0)*np.exp(-2*T) + 0.5*(1-np.exp(-2*T))

u, f, fnum = KBE(5, 1024, 1.0, a, F)

fig, ax = plt.subplots()
ax.set(xlabel="x", ylabel = "<x^2>", yscale="log")
ax.plot(u, fnum, label="KBE")
ax.plot(u, fexact(u, 1.0), label="Analytical")
ax.legend()

# %%
u, f, fnum = KBE(5, 256, 1.0, a, F)
u0, F0 = evaluateF(u, f, 0.5)
print(fexact(u0, 1.0) - F0)

# %%
u, f, fnum = KBE(5, 256, 1.0, a, F, h=1.e-4)
u0, F0 = evaluateF(u, f, 0.5)
print(fexact(u0, 1.0) - F0)

# %%
# du = sin(u) dt + σ dW
# F(u) = u**4
L, n, tmax = 10, 1024, 1.0

a = lambda u : np.sin(u)
F = lambda u : u**4

# %%
fig, ax = plt.subplots()
ax.set(xlabel="x", ylabel = "<x^4>", yscale="log")
u, f, fnum = KBE(10, 1024, 1.0, a, F)
ax.plot(u, fnum, label="KBE")
u, f, fnum = KBE(20, 2048, 1.0, a, F)
ax.plot(u, fnum, label="KBE")
ax.legend()

# %%
for L, N in zip([5,5,5,5], [32,64,128,256]):
    print(f"L={L}, N={N}")
    u, f, fnum = KBE(L, N, 1.0, a, F, h=2.e-4)
    u0, F0 = evaluateF(u, f, 0.5)
    print(f"------------")

# %%
# With flattening
L, N, tmax = 40, 1024, 1.0
hval = [8.e-4, 4.e-4, 2.e-4, 1.e-4, 5.e-5]
f0val = []
for h in hval:
    print(f"h={h}")
    u, f, fnum = KBE(L, N, tmax, a, F, h = h)
    u0, F0 = evaluateF(u, f, 0.5)
    f0val.append(F0)
    print(f"------------")

# %%
f0val = np.asarray(f0val)
2*f0val[1:]-f0val[:-1]

# %%
# Without flattening
L, N, tmax = 40, 1024, 1.0
hval = [8.e-4, 4.e-4, 2.e-4, 1.e-4, 5.e-5]
f0val = []
for h in hval:
    print(f"h={h}")
    u, f, fnum = KBE(L, N, tmax, a, F, h = h, flatten=False)
    u0, F0 = evaluateF(u, f, 0.5)
    f0val.append(F0)
    print(f"------------")

# %%
f0val = np.asarray(f0val)
2*f0val[1:]-f0val[:-1]
