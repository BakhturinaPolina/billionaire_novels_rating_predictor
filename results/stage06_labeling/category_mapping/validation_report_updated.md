# Category Mapping Validation Report

## Summary

- **Total topics analyzed**: 361
- **Z misclassifications found**: 4
- **Explicit/Intimacy issues**: 0
- **Regex pattern issues**: 1

## Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| A_commitment_hea | 15 | 4.2% |
| B_mutual_intimacy | 32 | 8.9% |
| C_explicit | 12 | 3.3% |
| D_luxury_wealth_status | 10 | 2.8% |
| E_threat_danger | 10 | 2.8% |
| F_negative_affect | 16 | 4.4% |
| G_rituals_gifts | 8 | 2.2% |
| H_domestic_nesting | 8 | 2.2% |
| I_humor_lightness | 12 | 3.3% |
| J_family_support | 19 | 5.3% |
| K_work_corporate | 22 | 6.1% |
| L_vices_addictions | 1 | 0.3% |
| M_health_recovery | 6 | 1.7% |
| N_separation_reunion | 2 | 0.6% |
| O_appearance_aesthetics | 22 | 6.1% |
| P_tech_media | 15 | 4.2% |
| Q_miscommunication | 11 | 3.0% |
| R1_protectiveness | 4 | 1.1% |
| R2_jealousy | 1 | 0.3% |
| S_scene_anchor | 17 | 4.7% |
| T_repair_apology | 3 | 0.8% |
| Z_noise_oog | 115 | 31.9% |

## ⚠️ Z Misclassifications (High Priority)

Topics classified as Z_noise_oog that contain romance-relevant keywords:

### Topic 26: Nighttime Stand
- **Current**: Z_noise_oog
- **Issue**: Contains 'nighttime' but classified as Z
- **Expected**: B_mutual_intimacy, H_domestic_nesting
- **Keywords**: tonight, night, evening, nights, stand, nighttime, fun, different, best, middle
- **Severity**: medium

### Topic 106: Nighttime Confession
- **Current**: Z_noise_oog
- **Issue**: Contains 'nighttime' but classified as Z
- **Expected**: B_mutual_intimacy, H_domestic_nesting
- **Keywords**: lines, easy, understandable, item, idea, night
- **Severity**: medium

### Topic 241: First Touch
- **Current**: Z_noise_oog
- **Issue**: Contains 'touch' but classified as Z
- **Expected**: B_mutual_intimacy
- **Keywords**: start
- **Severity**: medium

### Topic 274: Wild Night Out
- **Current**: Z_noise_oog
- **Issue**: Contains 'night' but classified as Z
- **Expected**: B_mutual_intimacy, H_domestic_nesting
- **Keywords**: clubs, members, exclusive, stag, wilder, restaurants, member, strip, whores, frequent
- **Severity**: medium

## Regex Pattern Matching Issues

### Topic 39: Moist Mouth Play
- **Current**: C_explicit
- **Expected**: B_mutual_intimacy
- **Issue**: Should be B_mutual_intimacy but is C_explicit
- **C_explicit patterns matched**: 1

## Recommendations

1. **Run fix Z script** to reclassify topics with romance-relevant keywords
2. **Review regex patterns** for 'intercourse', 'mouth play', 'outfit'
3. **Update patterns** if needed to catch these cases

## Next Steps

1. Review identified issues
2. Run `fix_z_topics.py` if Z misclassifications found
3. Update regex patterns in `map_topics_to_categories.py` if needed
4. Re-run category mapping after fixes