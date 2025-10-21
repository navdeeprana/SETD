# Usage :
# julia CGLE_variousD.jl DINPUT
# julia CGLE_variousD.jl 0.01

# Imports and setup
import Pkg;
Pkg.activate(".");
Pkg.instantiate();

using Revise, Printf, LinearAlgebra, FFTW, JLD2, Random, ProgressMeter
using StatsBase
includet("src/cgle.jl")
includet("src/plotting.jl")

DINPUT = parse(Float64, ARGS[1])

nx, lx = 1024, 320π
tmax, h, saveat, save_after = 2.e+5, 2.e-2, 2.e+2, 0.0
seed = rand(Random.RandomDevice(), UInt64)
pars = (b = 0.0, c = 0.1, D = DINPUT, q = 16 * (2π / lx));

fout = @sprintf "data/cgleD_nx_%04d_ln_%04d_D_%.2f.jld2" nx round(Int,lx/(2π)) pars.D

sim = cgle_simulation(nx, lx, pars, tmax, h, saveat, save_after, seed, fout);
solve(sim, tmax, saveat, save_after);