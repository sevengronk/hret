.PHONY: help install test test-all sample train benchmark clean

help:
	@echo "install    install the package in editable mode with dev tools"
	@echo "test       run fast unit tests"
	@echo "test-all   run the full test suite including the end-to-end test"
	@echo "sample     regenerate data/sample"
	@echo "train      train HRET with configs/hret.yaml"
	@echo "benchmark  compare with popularity and BPR-MF"
	@echo "clean      remove caches and local runs"

install:
	python -m pip install -e ".[dev]"

test:
	pytest -m "not slow"

test-all:
	pytest

sample:
	python scripts/generate_synthetic.py --out data/sample

train:
	python scripts/train.py --config configs/hret.yaml --output runs/hret

benchmark:
	python scripts/benchmark.py --config configs/hret.yaml --output runs/benchmark

clean:
	rm -rf runs .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
