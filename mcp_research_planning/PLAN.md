# Reliability & Reproducibility — Implementation Plan

This plan operationalizes the **Pipeline Reliability & Reproducibility Implementation Plan** using stage-first structure (01–07) and two MCP servers:
- **task_researcher** → planning (milestones, tasks, dependencies, risks, acceptance checks)
- **qualitativeresearch** → provenance (code/config/data/results; derived_from/supports/answers)

---

## Milestones (4 phases)

### M1 — Foundation (Week 1)
- Seed unification across stages (01–07), add `seed` to all configs
- Hardcoded path audit & replacement via `resolve_path()`
- Pareto constraint enforcement (`nr_topics >= 200`) in Stage 05
- Unified experiment ledger schema + normalizer

**Exit criteria:** `make contracts` passes; `results/pareto/pareto.csv` exists with constraint applied; unified `model_evaluation_results.csv` present.

### M2 — Quality Gates (Week 2)
- Pandera data schemas for Goodreads & chapters (Stages 01–02)
- Pytests (subset BERTopic run; selection constraint test; config load tests)
- Minimal CI (3.10/3.11) running tests + contracts

**Exit criteria:** CI green on main; schema violations fail locally and in CI.

### M3 — Features & Mini-run (Week 3)
- Quartile binning (Q1–Q4) + normalized positions
- Topic labeling output + report
- `make mini` wired with `--subset` / `--trials` flags

**Exit criteria:** `make mini` completes; labeling report generated.

### M4 — Docs & MCP (Week 4)
- Stage READMEs, root README, contracts doc expansion
- MCP documentation (session templates, health queries, Cursor rules)
- Optional: Move GH Actions to path-filtered matrix

**Exit criteria:** All docs present; MCP health queries return expected results.

---

## Task Breakdown (by section)

### 1) Immediate Verification
- Contracts & banners: add `--dry-run` to all stage `main.py`; ensure banners show inputs/outputs/constraints
- Determinism: implement `set_seed()`; call early in each stage
- Path audit: grep for literals; replace with config-resolved paths

### 2) Fill Functional Gaps
- Stage 07: `quartile_binning.py` → `chapters_quartiled.parquet`
- Stage 05: filter by `min_nr_topics` prior to Pareto; write `pareto.csv` + `BEST_MODEL.md`

### 3) Unify Experiment Ledger (Stage 04)
- Define schema (`ledger_schema.py`)
- `scripts/normalize_ledger.py` to produce `model_evaluation_results.csv`
- Stage 04 writes in unified schema

### 4) Smoke Tests + CI
- pytest minimal set + coverage
- GH Actions workflow + `make contracts`

### 5) Data Quality Gates (Stages 01–02)
- Pandera schemas & integration; fail on violation

### 6) Topic Labeling Consistency (Stage 06)
- Config seeds/mappings; output `topic_labels.parquet` + `labeling_report.md`

### 7) Reproducible Mini-Run
- Make `mini` target with `--subset` and `--trials`

### 8) Documentation
- Update `PIPELINE_CONTRACTS.md`
- Stage READMEs + Root README

### 9) MCP Integration Docs
- Session templates; health queries; Cursor rules sketch

### 10) Backlog
- MLflow; DVC/LFS; optuna presets; saved figures; notebook linting

---

## Two-week & Four-week slices

**Week 1–2 (ship CI first):**
- M1 + half of M2 (tests + CI) → aim: first green CI with contracts

**Week 3–4:**
- M3 features (`mini`, quartiles, labeling) + M4 docs/MCP

---

## Acceptance Checks (summary)
- See `ACCEPTANCE_CHECKS.md` for the full list and Make targets.
- CI must run: `pytest -q` and `make contracts`.

---

## Dependencies & Critical Path
See `DAG.mmd` (Mermaid). Critical path optimized for earliest CI green:
1) Seeds & paths → Pareto constraint → Ledger unification → Tests → CI
2) Then quartiles & labeling → mini-run → docs & MCP
