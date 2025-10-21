KERNEL_NAME := julia-1.10


IPYNB := $(wildcard *.ipynb)
JULIA_TRACKED := $(wildcard jl/*.jl)

.PHONY: scripts, format, formatsrc, formatnb, notebooks, execute, clean

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

scripts: $(IPYNB) ## Convert notebooks to julia sources for tracking
	@for f in $(IPYNB); do echo "Converting $$f"; jupytext --to jl $$f; done
	@mkdir -p jl; mv ??_*.jl jl/.

format: formatsrc formatnb ## Format src/ and notebooks

formatsrc: ## Format src/
	julia -e 'using JuliaFormatter; format_file("src/")'

formatnb: $(IPYNB) ## Format all notebooks
	@for f in $(IPYNB); do echo "Formatting $$f"; \
		jupytext --pipe "julia -e 'using JuliaFormatter; format_file(ARGS[1]; margin=150)' {}" $$f; done

notebooks: $(JULIA_TRACKED) ## Convert all julia files to notebooks for working
	@for f in $(JULIA_TRACKED); do echo "Converting $$f to notebook"; jupytext --to ipynb $$f; done

execute: $(IPYNB) ## Run all notebooks
	@for f in $(IPYNB); do echo "Executing $$f"; \
		jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=$(KERNEL_NAME) $$f; done

clean:
	@find . -name ".ipynb_check*" -exec rm -r {} \;
