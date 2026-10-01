# MOSAIC developer entry points. Everything here is a thin wrapper around the CLI.
.PHONY: help install setup data pipeline test test-fast lint fmt typecheck ci clean report experiments

PY := .venv/Scripts/python.exe
UV := uv
PIPELINE := configs/experiments/e2e_full.yaml

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## editable install with dev extras
	$(UV) pip install --python $(PY) -e ".[dev]"

setup: install ## install + build the default synthetic dataset
	$(PY) -m mosaic data generate --profile small

data: ## regenerate the standard synthetic benchmark (deterministic)
	$(PY) -m mosaic data generate --profile research

pipeline: ## run the full end-to-end research experiment
	$(PY) -m mosaic experiment run --config $(PIPELINE)

test: ## run the whole test suite
	$(PY) -m pytest -q

test-fast: ## unit + leakage tests only
	$(PY) -m pytest -q tests/unit tests/leakage

lint: ## ruff check
	$(PY) -m ruff check src tests

fmt: ## ruff format + autofix
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

typecheck: ## mypy
	$(PY) -m mypy src/mosaic

ci: fmt lint typecheck test ## what CI runs

report: ## regenerate the research report from stored evidence
	$(PY) -m mosaic report build

clean: ## remove caches (never touches data/)
	rm -rf .pytest_cache .ruff_cache .mypy_cache .hypothesis
	find src tests -name __pycache__ -type d -exec rm -rf {} +
