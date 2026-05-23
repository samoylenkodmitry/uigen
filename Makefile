# V7+ cloud-portable run targets.
#
# Common usage:
#   make local                    full Gate B run on this workstation
#   make bench-local              short benchmark (default 200 steps)
#   make local-env                print Python / torch / GPU info
#
# Cloud variants set UIGEN_RUNTIME to the matching configs/runtime/*.yaml
# and reuse the same scripts/run_experiment.py + scripts/benchmark_runtime.py.

PYTHON ?= .venv/bin/python
EXPERIMENT ?= experiments/v7_completer_gateB.yaml

LOCAL_RUNTIME  := configs/runtime/local.yaml
KAGGLE_RUNTIME := configs/runtime/kaggle.yaml
COLAB_RUNTIME  := configs/runtime/colab.yaml

.PHONY: help local kaggle colab \
        bench-local bench-kaggle bench-colab \
        local-env kaggle-env colab-env \
        local-dry kaggle-dry colab-dry

help:
	@echo "V7 portable run targets"
	@echo "  make local              run experiment with local runtime"
	@echo "  make kaggle             run experiment with kaggle runtime"
	@echo "  make colab              run experiment with colab runtime"
	@echo "  make bench-{local,kaggle,colab}   short benchmark"
	@echo "  make {local,kaggle,colab}-env     print env info"
	@echo "  make {local,kaggle,colab}-dry     show resolved command, don't run"
	@echo
	@echo "Overrides: PYTHON=<python>, EXPERIMENT=<yaml>"
	@echo "Default EXPERIMENT=$(EXPERIMENT)"

# Full experiment.
local:
	UIGEN_RUNTIME=$(LOCAL_RUNTIME) $(PYTHON) scripts/run_experiment.py --experiment $(EXPERIMENT)

kaggle:
	UIGEN_RUNTIME=$(KAGGLE_RUNTIME) $(PYTHON) scripts/run_experiment.py --experiment $(EXPERIMENT)

colab:
	UIGEN_RUNTIME=$(COLAB_RUNTIME) $(PYTHON) scripts/run_experiment.py --experiment $(EXPERIMENT)

# Dry-run prints the resolved trainer command without invoking it.
local-dry:
	UIGEN_RUNTIME=$(LOCAL_RUNTIME) $(PYTHON) scripts/run_experiment.py --experiment $(EXPERIMENT) --dry-run

kaggle-dry:
	UIGEN_RUNTIME=$(KAGGLE_RUNTIME) $(PYTHON) scripts/run_experiment.py --experiment $(EXPERIMENT) --dry-run

colab-dry:
	UIGEN_RUNTIME=$(COLAB_RUNTIME) $(PYTHON) scripts/run_experiment.py --experiment $(EXPERIMENT) --dry-run

# Short benchmark.
bench-local:
	UIGEN_RUNTIME=$(LOCAL_RUNTIME) $(PYTHON) scripts/benchmark_runtime.py --experiment $(EXPERIMENT)

bench-kaggle:
	UIGEN_RUNTIME=$(KAGGLE_RUNTIME) $(PYTHON) scripts/benchmark_runtime.py --experiment $(EXPERIMENT)

bench-colab:
	UIGEN_RUNTIME=$(COLAB_RUNTIME) $(PYTHON) scripts/benchmark_runtime.py --experiment $(EXPERIMENT)

# Environment introspection.
local-env:
	$(PYTHON) scripts/print_env.py

kaggle-env:
	$(PYTHON) scripts/print_env.py

colab-env:
	$(PYTHON) scripts/print_env.py
