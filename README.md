# SETD
Code for the SETD paper

# Getting started

> [!IMPORTANT]
> This is not a Julia package. You cannot install it with `add SETD`.
> Tested with `julia v1.10`.

## Prerequisites

You will need a working installation of Julia, jupyterlab, jupytext, and IJulia to generate
and run the notebooks. If you can run Julia notebooks on your machine, proceed to the next step.

- Install Julia. Using [juliaup](https://github.com/JuliaLang/juliaup) is recommended.
- Install jupyterlab and jupytext using anaconda or any other way you prefer.
- Install IJulia using Julia package manager.

## Running the notebooks

- Clone/download the repository.
- Install the Julia dependencies by activating the project and then instantiating it.
- The notebooks are converted and stored under `jl/` folder as plain `.jl` files using jupytext. To recreate the notebooks from these files run `make notebooks` and then move it to the base directory of the repository. You have to manually move them to avoid overwriting any notebooks you have previously generated.
- If you do not have `make`, you can convert them directly by running `jupytext --to ipynb filename.jl`.
- Create `data` and `figs` folders to generate data and figures.
- Now you can run jupyterlab and start running the notebooks.

