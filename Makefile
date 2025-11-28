#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = bitsxlamarato-medulloblastoma
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Install development dependencies
.PHONY: requirements-dev
requirements-dev:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -e ".[dev]"

## Install documentation dependencies
.PHONY: requirements-docs
requirements-docs:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -e ".[docs]"

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/

## Format code using black and isort
.PHONY: format
format:
	$(PYTHON_INTERPRETER) -m black medulloblastoma/ tests/
	$(PYTHON_INTERPRETER) -m isort medulloblastoma/ tests/

## Lint using flake8, black, and isort
.PHONY: lint
lint:
	$(PYTHON_INTERPRETER) -m flake8 medulloblastoma/ tests/
	$(PYTHON_INTERPRETER) -m black --check medulloblastoma/ tests/
	$(PYTHON_INTERPRETER) -m isort --check-only medulloblastoma/ tests/

## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests/ -v

## Run tests with coverage
.PHONY: test-cov
test-cov:
	$(PYTHON_INTERPRETER) -m pytest tests/ -v --cov=medulloblastoma --cov-report=html --cov-report=term

## Download and prepare data
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -c "from medulloblastoma.dataset import download_data, prepare_data; download_data(); prepare_data()"

## Run preprocessing pipeline
.PHONY: preprocess
preprocess:
	$(PYTHON_INTERPRETER) -m medulloblastoma.features \
		--data_path data/raw/cavalli.csv \
		--metadata_path data/raw/cavalli_subgroups.csv \
		--save_path data/processed/

## Build documentation
.PHONY: docs
docs:
	mkdocs build

## Serve documentation locally
.PHONY: docs-serve
docs-serve:
	mkdocs serve

## Run complete analysis notebook
.PHONY: notebook
notebook:
	jupyter lab notebooks/medulloblastoma-analysis.ipynb

## Validate installation
.PHONY: validate
validate:
	$(PYTHON_INTERPRETER) -c "from medulloblastoma.dataset import download_data; from medulloblastoma.features import load_data; from medulloblastoma.plots import plot_umap_binary; print('✅ Installation validated successfully!')"

## Create conda environment
.PHONY: create_environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

## Show help
.PHONY: help
help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
