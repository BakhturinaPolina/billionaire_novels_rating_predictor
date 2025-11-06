# Checklists (developer PR helpers)

## M1 Foundation
- [ ] Add `seed` to all configs
- [ ] Implement `set_seed()`; call in every stage at start
- [ ] Replace hardcoded paths with `resolve_path()`
- [ ] Implement Pareto pre-filter by `min_nr_topics`
- [ ] Define ledger schema; write normalizer; Stage 04 writes unified CSV

## M2 Quality
- [ ] Add Pandera schemas and integrate in Stage 01–02
- [ ] Write minimal pytests
- [ ] Author CI workflow; run tests + contracts

## M3 Features
- [ ] Implement `quartile_binning.py` and wire into Stage 07
- [ ] Output labeling parquet + report
- [ ] Add `mini` Make target; `--subset` and `--trials` flags

## M4 Docs & MCP
- [ ] Update `PIPELINE_CONTRACTS.md`
- [ ] Add Stage READMEs and root README
- [ ] Add MCP docs (session templates, health queries, rules sketch)
