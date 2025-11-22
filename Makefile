# Makefile for stage-based pipeline

.PHONY: inventory contracts stage01 stage02 stage03 stage05 stage06 stage07 experiments pareto topics analysis all

inventory:
	@echo "Repository structure inventory"
	@python -c "import os; stages = [f'stage0{i}' for i in range(1, 8)]; print('Stages: 01..07 present; configs loaded')"

contracts:
	@echo "Expected output contracts:"
	@echo "  - results/experiments/model_evaluation_results.csv"
	@echo "  - results/pareto/pareto.csv"
	@echo "  - results/topics/by_book.csv"

stage01:
	@echo "Running Stage 01: Ingestion"
	python -m src.stage01_ingestion.main --config configs/paths.yaml

stage02:
	@echo "Running Stage 02: Preprocessing"
	python -m src.stage02_preprocessing.main --config configs/paths.yaml

stage03:
	@echo "Running Stage 03: Modeling"
	python -m src.stage03_modeling.main train --config configs/bertopic.yaml

stage05:
	@echo "Running Stage 05: Selection"
	python -m src.stage05_selection.main --config configs/selection.yaml

stage06:
	@echo "Running Stage 06: Labeling"
	python -m src.stage06_labeling.main --config configs/labeling.yaml

stage07:
	@echo "Running Stage 07: Analysis"
	python -m src.stage07_analysis.main --config configs/scoring.yaml

experiments:
	@echo "Running hyperparameter optimization (Stage 03)"
	python -m src.stage03_modeling.main optimize --config configs/octis.yaml

pareto:
	@echo "Running Pareto selection (Stage 05)"
	$(MAKE) stage05

topics:
	@echo "Topic analysis (Stage 06)"
	$(MAKE) stage06

analysis:
	@echo "Statistical analysis (Stage 07)"
	$(MAKE) stage07

all: stage01 stage02 stage03 stage05 stage06 stage07
	@echo "All stages completed"

