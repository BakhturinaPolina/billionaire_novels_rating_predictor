# Risk Register

| ID | Risk | Likelihood | Impact | Mitigation | Early Warning Signal | Owner |
|----|------|------------|--------|------------|----------------------|-------|
| R1 | Hidden hardcoded paths break runs | Medium | High | Grep audit + config resolver; add test | CI fails `contracts` target | Tech Lead |
| R2 | Non-determinism in modeling | Medium | Medium | Global `set_seed()` and seed in configs | Varying nr_topics for same input | Modeling |
| R3 | Pareto constraint misapplied | Low | High | Unit test for `nr_topics >= 200`; assert columns | `pareto.csv` missing / wrong counts | Selection |
| R4 | Ledger schema drift | Medium | Medium | Normalizer + schema test | CI diff on columns | Experiments |
| R5 | Data schema violations | Medium | High | Pandera gates in stages 01–02 | Stage exits with schema errors | Ingestion |
