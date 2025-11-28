### BERTopic + OCTIS Pipeline Overview

It serves as living documentation for anyone who needs to rerun, extend or analyse the experiments.

---

#### 1.  Repository Layout

```
octis bertopic full project/
│  bertopic_plus_octis.py        # single-script pipeline runner ⭆ optimisation
│  optimizer.py                  # generic Bayesian optimiser wrapper (skopt)
│  restart_script.py             # crash-safe launcher with automatic restart
│  topic_npy_to_json.py          # helper to convert NumPy topic arrays → JSON
│
├─ Billionaire_OCTIS_ALL_Models_Results_CSV_Only/
│     └─ <embedding-model>/
│           result.json          # full optimisation log (one per model)
│
├─ model_evaluation_results.csv  # flat summary across **all** models (one row per trial)
└─ … (other helper files)
```

---

#### 2.  High-Level Flow

1. **Dataset loading**  
   A pre-processed corpus (prepared outside this project) is loaded via OCTIS’ `Dataset` API.

2. **Embedding generation**  
   `bertopic_plus_octis.py` computes **sentence-level embeddings** using one of several SBERT-family checkpoints (e.g. *all-MiniLM-L12-v2*).  Large embeddings are cached (`*.npy`) for reuse.

3. **Model wrapper**  
   A custom subclass `BERTopicOctisModelWithEmbeddings` (lines 261-430) adapts BERTopic to OCTIS’ model interface:
   • internally instantiates `BERTopic` with supplied hyper-parameters  
   • exposes `train_model()` expected by OCTIS  
   • returns topic-word, topic-doc, etc. matrices for downstream metrics.

4. **Bayesian Hyper-Parameter Optimisation**  
   • Search space spans UMAP, HDBSCAN, vectoriser and BERTopic knobs.  
   • `optimizer.Optimizer` wraps **scikit-optimize** (`skopt`).  
   • Objective = maximise OCTIS *Coherence(c_v)*; *Topic Diversity* tracked as secondary metric.  
   • RF surrogate + LCB acquisition (can be switched).  
   • Each point retrains BERTopic *model_runs* times and averages metric.

5. **Persistence & Monitoring**  
   • Every iteration is timed; metrics, hyper-params saved into <model>/`result.json`.  
   • Periodic CSV dumps consolidate results across embedding models.

6. **Crash-Resilience**  
   `restart_script.py` restarts the optimisation command after failure and e-mails logs (via `send_mail`).

---

#### 3.  Understanding the Result Files

Each `result.json` follows the same schema (excerpt from *all-MiniLM-L12-v2*):

* `search_space`     – canonical hyper-parameter bounds
* `current_call`     – iterations completed  (≤ `number_of_call`)
* `dict_model_runs`  – nested dict ⇒ metric ⇒ iteration ⇒ list[float]
* `f_val`            – metric value per iteration (–1 indicates failed run)
* `x_iters`          – parallel list of tested hyper-parameter settings

Combined CSV (`model_evaluation_results.csv`) flattens the same information; each row stores the *best-seen* configuration for a given embedding model.

---

#### 4.  How to Pick the Best Model

1. **Primary metric**  – high `Coherence(c_v)` (> 0.75 in observed scale)  
2. **Diversity check** – avoid mode-collapse (`TopicDiversity` < 0.3 unhealthy)  
3. Inspect *topic-word* lists qualitatively.

Example – *all-MiniLM-L12-v2* best trial  
`Coherence   0.82`   /   `TopicDiversity   0.24`   @ iteration 89.

---

#### 5.  References

* OCTIS documentation – <https://octis.readthedocs.io/en/latest/readme.html>
* BERTopic paper – Grootendorst (2022)  
* scikit-optimize API – <https://scikit-optimize.github.io/stable/>

---

*Last updated: 2025-11-05*
