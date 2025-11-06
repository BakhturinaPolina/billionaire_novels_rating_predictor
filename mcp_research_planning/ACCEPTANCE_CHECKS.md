# Acceptance Checks (CI/Make Targets)

## Contracts
- `make contracts` passes and prints expected outputs
- Files exist with schemas:
  - `results/experiments/model_evaluation_results.csv` (unified)
  - `results/pareto/pareto.csv` (filtered by min_nr_topics first)
  - `data/processed/chapters_quartiled.parquet` (Q1–Q4)
  - `results/topics/topic_labels.parquet`

## Determinism
- Each stage logs the seed from config; same inputs → same outputs (subset test)

## Selection Constraint
- Test: BEST model has `nr_topics >= min_nr_topics`

## CI
- GH Actions runs `pytest -q` and `make contracts` on Python 3.10/3.11

## Make Targets (suggested)
- `check:ledger` — validate unified ledger columns and non-empty
- `check:pareto` — validate constraint applied and columns present
- `check:quartiles` — file exists; has `quartile` & `normalized_position`
- `check:labels` — `topic_labels.parquet` exists; >0 rows
- `contracts` — runs all check:* targets
