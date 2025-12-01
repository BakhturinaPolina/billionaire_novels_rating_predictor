# Category Mapping Validation Report

## Summary

- **Total topics analyzed**: 361
- **Z misclassifications found**: 4
- **Explicit/Intimacy issues**: 0
- **Regex pattern issues**: 1

## Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| A_commitment_hea | 13 | 3.6% |
| B_mutual_intimacy | 10 | 2.8% |
| C_explicit | 9 | 2.5% |
| D_luxury_wealth_status | 4 | 1.1% |
| E_threat_danger | 11 | 3.0% |
| F_negative_affect | 14 | 3.9% |
| G_rituals_gifts | 5 | 1.4% |
| H_domestic_nesting | 11 | 3.0% |
| I_humor_lightness | 7 | 1.9% |
| J_family_support | 15 | 4.2% |
| K_work_corporate | 24 | 6.6% |
| L_vices_addictions | 1 | 0.3% |
| M_health_recovery | 8 | 2.2% |
| N_separation_reunion | 1 | 0.3% |
| O_appearance_aesthetics | 30 | 8.3% |
| P_tech_media | 16 | 4.4% |
| Q_miscommunication | 12 | 3.3% |
| R1_protectiveness | 2 | 0.6% |
| R2_jealousy | 1 | 0.3% |
| S_scene_anchor | 19 | 5.3% |
| T_repair_apology | 4 | 1.1% |
| Z_noise_oog | 144 | 39.9% |

## ⚠️ Z Misclassifications (High Priority)

Topics classified as Z_noise_oog that contain romance-relevant keywords:

### Topic 26: tonight
- **Current**: Z_noise_oog
- **Issue**: Contains 'nighttime' but classified as Z
- **Expected**: B_mutual_intimacy, H_domestic_nesting
- **Keywords**: tonight, night, evening, nights, stand, nighttime, fun, different, best, middle
- **Severity**: medium

### Topic 39: lip
- **Current**: Z_noise_oog
- **Issue**: Contains 'moan' but classified as Z
- **Expected**: B_mutual_intimacy, C_explicit
- **Keywords**: lip, teeth, nose, tongue, mouth, bridge, cheek, moan, lower, saliva
- **Severity**: medium

### Topic 56: sleep
- **Current**: Z_noise_oog
- **Issue**: Contains 'night' but classified as Z
- **Expected**: B_mutual_intimacy, H_domestic_nesting
- **Keywords**: sleep, asleep, nap, hours, night, lack, tired, nights, hour, insomnia
- **Severity**: medium

### Topic 106: lines
- **Current**: Z_noise_oog
- **Issue**: Contains 'night' but classified as Z
- **Expected**: B_mutual_intimacy, H_domestic_nesting
- **Keywords**: lines, easy, understandable, item, idea, night
- **Severity**: medium

## Regex Pattern Matching Issues

### Topic 39: Moist Mouth Play
- **Current**: Z_noise_oog
- **Expected**: B_mutual_intimacy
- **Issue**: Should be B_mutual_intimacy but is Z_noise_oog
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