# Statistical Results: Themes, Reach, and Perceived Quality (Pilot N=92)

**Date**: January 2025  
**Sample Size**: N=92 (pilot analysis)  
**Analysis Type**: Effect-size-based results with bootstrap confidence intervals

---

This section reports effect-size-based results from the pilot analysis of billionaire romance novels. We separate two outcome channels:

* **Reach / visibility**: `log_rating_count` (log number of ratings)

* **Perceived quality**: `rating_mean` and **Bayesian-adjusted Goodreads rating** `avg_rating_bayes`

All reported effects are **standardized** (beta_std) and estimated via **bootstrap** with **95% confidence intervals** (CI) and **sign stability** *P(beta>0)*.

**Download the DOCX version:** [statistical_results_section.docx](sandbox:/mnt/data/statistical_results_section.docx)

---

## Table 1. Macro-axis definitions

Each macro-axis is a weighted combination of standardized CORE predictors.

| axis                | component                  | weight | note                                |
| :------------------ | :------------------------- | :----- | :---------------------------------- |
| AX_status_dominance | D_power_wealth_luxury__pc1 | 1.0    |                                     |
| AX_status_dominance | R2_alpha_guarding          | 1.0    |                                     |
| AX_payoff_safety    | A2_emotional_safety__pc1   | 1.0    |                                     |
| AX_payoff_safety    | Q_repair                   | 0.7    |                                     |
| AX_payoff_safety    | R1_protective_caretaking   | 0.7    |                                     |
| AX_negative_affect  | F2_anger_frustration       | 1.0    |                                     |
| AX_negative_affect  | F3_anxiety_worry           | 1.0    |                                     |
| AX_negative_affect  | F1_sadness_grief           | 0.6    |                                     |
| AX_explicitness     | C_explicit_eroticism       | 1.0    | explicit sexual content only        |
| AX_attraction       | B1_attraction_chemistry    | 1.0    | chemistry / non-explicit attraction |

---

## Table 2. Macro-axis level effects

**How to read:** beta_std = standardized effect size; CI = 95% bootstrap interval; P(beta>0) = sign stability.

### Table 2.1 Macro-axis effects → reach (log_rating_count)

| Predictor           | beta_std | CI_low | CI_high | P(beta>0) | N  | Controls |
| :------------------ | :------- | :----- | :------ | :-------- | :- | :------- |
| AX_status_dominance | 0.461    | 0.284  | 0.613   | 1.000     | 92 |          |
| AX_drama_obstacle   | 0.335    | 0.074  | 0.574   | 0.993     | 92 |          |
| AX_payoff_safety    | 0.327    | 0.141  | 0.500   | 0.999     | 92 |          |
| AX_negative_affect  | -0.121   | -0.283 | 0.051   | 0.089     | 92 |          |
| AX_explicitness     | -0.297   | -0.503 | -0.057  | 0.004     | 92 |          |

### Table 2.2 Macro-axis effects → quality beyond reach (rating_mean)

| Predictor           | beta_std | CI_low | CI_high | P(beta>0) | N  | Controls         |
| :------------------ | :------- | :----- | :------ | :-------- | :- | :--------------- |
| AX_payoff_safety    | 0.207    | -0.005 | 0.413   | 0.974     | 92 | log_rating_count |
| AX_status_dominance | 0.006    | -0.184 | 0.195   | 0.525     | 92 | log_rating_count |
| AX_drama_obstacle   | -0.026   | -0.274 | 0.223   | 0.417     | 92 | log_rating_count |
| AX_negative_affect  | -0.133   | -0.312 | 0.044   | 0.051     | 92 | log_rating_count |
| AX_explicitness     | -0.175   | -0.356 | 0.019   | 0.040     | 92 | log_rating_count |

---

## Table 3. Macro-axis effects → Bayesian-adjusted quality (avg_rating_bayes)

### Table 3.1 No control (raw perceived quality)

| Predictor           | beta_std | CI_low | CI_high | P(beta>0) | N  | Controls |
| :------------------ | :------- | :----- | :------ | :-------- | :- | :------- |
| AX_payoff_safety    | 0.166    | -0.006 | 0.322   | 0.973     | 92 |          |
| AX_status_dominance | 0.125    | -0.086 | 0.288   | 0.895     | 92 |          |
| AX_drama_obstacle   | -0.023   | -0.240 | 0.194   | 0.399     | 92 |          |
| AX_negative_affect  | -0.091   | -0.249 | 0.074   | 0.201     | 92 |          |
| AX_explicitness     | -0.154   | -0.310 | 0.001   | 0.080     | 92 |          |

### Table 3.2 Controlling reach (quality beyond reach)

| Predictor           | beta_std | CI_low | CI_high | P(beta>0) | N  | Controls      |
| :------------------ | :------- | :----- | :------ | :-------- | :- | :------------ |
| AX_payoff_safety    | 0.077    | -0.091 | 0.248   | 0.823     | 92 | log_n_ratings |
| AX_status_dominance | 0.029    | -0.158 | 0.209   | 0.618     | 92 | log_n_ratings |
| AX_drama_obstacle   | -0.008   | -0.221 | 0.207   | 0.476     | 92 | log_n_ratings |
| AX_negative_affect  | -0.062   | -0.222 | 0.099   | 0.217     | 92 | log_n_ratings |
| AX_explicitness     | -0.125   | -0.281 | 0.027   | 0.074     | 92 | log_n_ratings |

---

## Table 4. CORE predictor level effects (top 10)

### Table 4.1 CORE → reach (log_rating_count)

| Predictor                  | beta_std | CI_low | CI_high | P(beta>0) | N  | Controls |
| :------------------------- | :------- | :----- | :------ | :-------- | :- | :------- |
| R2_alpha_guarding          | 0.435    | 0.226  | 0.597   | 1.000     | 92 |          |
| D_power_wealth_luxury__pc1 | 0.371    | 0.195  | 0.546   | 1.000     | 92 |          |
| A2_emotional_safety__pc1   | 0.323    | 0.126  | 0.503   | 0.998     | 92 |          |
| Q_repair                   | 0.230    | 0.021  | 0.406   | 0.985     | 92 |          |
| J_social_support_kin       | 0.191    | -0.043 | 0.426   | 0.949     | 92 |          |
| Q_miscommunication         | 0.110    | -0.214 | 0.441   | 0.720     | 92 |          |
| I_humor_lightness          | 0.037    | -0.200 | 0.230   | 0.663     | 92 |          |
| K_professional_intrusion   | 0.045    | -0.166 | 0.253   | 0.644     | 92 |          |
| M_health_recovery_growth   | 0.024    | -0.234 | 0.205   | 0.636     | 92 |          |
| R1_protective_caretaking   | 0.024    | -0.259 | 0.284   | 0.576     | 92 |          |

### Table 4.2 CORE → quality beyond reach (rating_mean; control log_rating_count)

| Predictor                  | beta_std | CI_low | CI_high | P(beta>0) | N  | Controls         |
| :------------------------- | :------- | :----- | :------ | :-------- | :- | :--------------- |
| R1_protective_caretaking   | 0.224    | 0.053  | 0.357   | 0.995     | 92 | log_rating_count |
| A2_emotional_safety__pc1   | 0.153    | -0.060 | 0.344   | 0.950     | 92 | log_rating_count |
| M_health_recovery_growth   | 0.096    | -0.069 | 0.270   | 0.878     | 92 | log_rating_count |
| Q_repair                   | 0.097    | -0.092 | 0.284   | 0.816     | 92 | log_rating_count |
| F3_anxiety_worry           | -0.018   | -0.201 | 0.179   | 0.420     | 92 | log_rating_count |
| Q_miscommunication         | -0.077   | -0.290 | 0.144   | 0.273     | 92 | log_rating_count |
| B1_attraction_chemistry    | -0.037   | -0.235 | 0.163   | 0.316     | 92 | log_rating_count |
| R2_alpha_guarding          | 0.004    | -0.174 | 0.181   | 0.509     | 92 | log_rating_count |
| D_power_wealth_luxury__pc1 | 0.018    | -0.151 | 0.185   | 0.581     | 92 | log_rating_count |
| C_explicit_eroticism       | -0.122   | -0.314 | 0.055   | 0.106     | 92 | log_rating_count |

---

## Table 5. Arc effects on rating_mean (pacing)

Arc predictors are `end_minus_begin` and `middle_minus_begin`. Positive beta means the theme increases over the story in higher-rated books (controlling reach).

### Table 5.1 Most positive arcs

| Predictor                                     | beta_std | CI_low | CI_high | P(beta>0) | N  |
| :-------------------------------------------- | :------- | :----- | :------ | :-------- | :- |
| F2_anger_frustration__end_minus_begin         | 0.238    | 0.075  | 0.412   | 0.995     | 92 |
| F3_anxiety_worry__end_minus_begin             | 0.190    | 0.015  | 0.364   | 0.981     | 92 |
| R_jealousy_possessiveness__middle_minus_begin | 0.121    | -0.000 | 0.261   | 0.973     | 92 |
| O_aesthetics_appearance__end_minus_begin      | 0.150    | -0.035 | 0.316   | 0.950     | 92 |
| K_professional_intrusion__middle_minus_begin  | 0.182    | -0.130 | 0.419   | 0.905     | 92 |
| R2_alpha_guarding__middle_minus_begin         | 0.089    | -0.043 | 0.238   | 0.893     | 92 |
| A2_emotional_safety__middle_minus_begin       | 0.134    | -0.100 | 0.347   | 0.878     | 92 |
| O_aesthetics_appearance__middle_minus_begin   | 0.136    | -0.110 | 0.351   | 0.869     | 92 |
| D_power_wealth_luxury__end_minus_begin        | 0.092    | -0.073 | 0.247   | 0.868     | 92 |
| K_professional_intrusion__end_minus_begin     | 0.152    | -0.095 | 0.388   | 0.858     | 92 |
| R_jealousy_possessiveness__end_minus_begin    | 0.073    | -0.086 | 0.229   | 0.811     | 92 |
| F1_sadness_grief__middle_minus_begin          | 0.088    | -0.087 | 0.295   | 0.800     | 92 |

### Table 5.2 Most negative arcs

| Predictor                                   | beta_std | CI_low | CI_high | P(beta>0) | N  |
| :------------------------------------------ | :------- | :----- | :------ | :-------- | :- |
| J_social_support_kin__end_minus_begin       | -0.177   | -0.350 | -0.007  | 0.019     | 92 |
| S_scene_anchors__middle_minus_begin         | -0.140   | -0.321 | 0.032   | 0.057     | 92 |
| S_scene_anchors__end_minus_begin            | -0.158   | -0.346 | 0.015   | 0.074     | 92 |
| C_explicit_eroticism__middle_minus_begin    | -0.112   | -0.279 | 0.057   | 0.089     | 92 |
| C_explicit_eroticism__end_minus_begin       | -0.120   | -0.306 | 0.066   | 0.103     | 92 |
| B1_attraction_chemistry__middle_minus_begin | -0.099   | -0.260 | 0.062   | 0.116     | 92 |
| A3_everyday_tenderness__middle_minus_begin  | -0.092   | -0.296 | 0.106   | 0.150     | 92 |
| Q_repair__middle_minus_begin                | -0.077   | -0.277 | 0.121   | 0.213     | 92 |
| B2_emotional_intimacy__end_minus_begin       | -0.069   | -0.239 | 0.105   | 0.218     | 92 |
| B2_emotional_intimacy__middle_minus_begin   | -0.061   | -0.227 | 0.111   | 0.249     | 92 |
| A3_everyday_tenderness__end_minus_begin     | -0.057   | -0.240 | 0.127   | 0.269     | 92 |
| R1_protective_caretaking__end_minus_begin   | -0.050   | -0.240 | 0.142   | 0.283     | 92 |

---

## Table 6. Predictive performance (repeated cross-validation)

| Outcome          | Model         | Folds | Repeats | CV_R2_mean | CV_R2_sd | CV_R2_min | CV_R2_max |
| :--------------- | :------------ | :---- | :------ | :--------- | :------- | :-------- | :-------- |
| rating_mean      | metadata_only | 5     | 20      | 0.108      | 0.031    | 0.060     | 0.173     |
| rating_mean      | metadata+core | 5     | 20      | 0.056      | 0.041    | -0.014    | 0.148     |
| log_rating_count | core_only     | 5     | 20      | 0.050      | 0.037    | -0.017    | 0.125     |

---

# Figures (download links)

All plots are PNG at 300 dpi.

* **Figure 1** Macro-axis effects → reach (log_rating_count): [macro_level_log_rating_count.png](sandbox:/mnt/data/report_assets/macro_level_log_rating_count.png)

* **Figure 2** Macro-axis effects → quality beyond reach (rating_mean): [macro_level_rating_mean.png](sandbox:/mnt/data/report_assets/macro_level_rating_mean.png)

* **Figure 3** Macro-axis effects → Bayesian quality (no control): [macro_bayes_no_control.png](sandbox:/mnt/data/report_assets/macro_bayes_no_control.png)

* **Figure 4** Macro-axis effects → Bayesian quality (controlling reach): [macro_bayes_control_log_n.png](sandbox:/mnt/data/report_assets/macro_bayes_control_log_n.png)

* **Figure 5** Arc effects (most positive): [arc_core_positive.png](sandbox:/mnt/data/report_assets/arc_core_positive.png)

* **Figure 6** Arc effects (most negative): [arc_core_negative.png](sandbox:/mnt/data/report_assets/arc_core_negative.png)

* **Figure 7** Predictive performance (repeated CV): [cv_r2.png](sandbox:/mnt/data/report_assets/cv_r2.png)

---

## Interpretation (simple and clear)

**Reach vs quality are different.** Reach (`log_rating_count`) mostly tracks visibility/market reach. Quality (ratings) tracks reader evaluation. The same themes do not predict both equally well.

**What predicts reach:** The strongest axis is **status/dominance** (luxury + alpha guarding), followed by **payoff/safety** (safety language + repair + caretaking). **Explicit erotics** is consistently negative for reach in this sample.

**What predicts quality beyond reach:** The clearest positive signal is **payoff/safety**. Negative affect and explicitness lean negative for quality, but are weaker than payoff/safety.

**Bayesian-adjusted quality:** Without controlling reach, payoff/safety is again the strongest positive axis; status/dominance can look positive because it overlaps with reach. When controlling reach, status/dominance collapses toward zero while payoff/safety remains mildly positive. This supports the two-channel story: status/dominance is mainly a reach driver; payoff/safety is closer to a quality driver.

**Pacing matters:** Higher-rated books tend to show a stronger **late-story crisis escalation** (anger/anxiety increases toward the end). Meanwhile, increasing end-of-book emphasis on kin/social support or setting/scene anchors is associated with lower ratings. A simple reading is: better-rated books manage pacing by concentrating toward a crisis-and-resolution arc rather than drifting into diffuse wrap-up material.

**Pilot framing:** This is a small-N pilot (N=92). The goal is not p-values; it's directionally stable effects, reach vs quality separation, and hypothesis refinement for the larger corpus.

---

**Report Generated**: January 2025  
**Analysis Type**: Bootstrap-based effect size estimation  
**Sample Size**: N=92 (pilot)

