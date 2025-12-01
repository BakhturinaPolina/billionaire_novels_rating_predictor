# Category Mapping Validation Report

## Summary

- **Total topics analyzed**: 361
- **Z misclassifications found**: 3
- **Explicit/Intimacy issues**: 0
- **Regex pattern issues**: 0

## Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| A_commitment_hea | 5 | 1.4% |
| B_mutual_intimacy | 14 | 3.9% |
| C_explicit | 10 | 2.8% |
| D_luxury_wealth_status | 3 | 0.8% |
| E_threat_danger | 8 | 2.2% |
| F_negative_affect | 27 | 7.5% |
| G_rituals_gifts | 11 | 3.0% |
| H_domestic_nesting | 16 | 4.4% |
| I_humor_lightness | 13 | 3.6% |
| J_family_support | 18 | 5.0% |
| K_work_corporate | 40 | 11.1% |
| L_vices_addictions | 2 | 0.6% |
| M_health_recovery | 10 | 2.8% |
| N_separation_reunion | 6 | 1.7% |
| O_appearance_aesthetics | 23 | 6.4% |
| P_tech_media | 5 | 1.4% |
| Q_miscommunication | 10 | 2.8% |
| R1_protectiveness | 5 | 1.4% |
| R2_jealousy | 2 | 0.6% |
| S_scene_anchor | 20 | 5.5% |
| T_repair_apology | 3 | 0.8% |
| Z_noise_oog | 110 | 30.5% |

## ⚠️ Z Misclassifications (High Priority)

Topics classified as Z_noise_oog that contain romance-relevant keywords:

### Topic 26: Nighttime Fun
- **Current**: Z_noise_oog
- **Issue**: Contains 'nighttime' but classified as Z
- **Expected**: B_mutual_intimacy, H_domestic_nesting
- **Keywords**: tonight, night, evening, nights, stand, nighttime, fun, different, best, middle
- **Severity**: medium

### Topic 106: Simple Nighttime Routine
- **Current**: Z_noise_oog
- **Issue**: Contains 'nighttime' but classified as Z
- **Expected**: B_mutual_intimacy, H_domestic_nesting
- **Keywords**: lines, easy, understandable, item, idea, night
- **Severity**: medium

### Topic 204: Forgetful Friends and Silly Nights
- **Current**: Z_noise_oog
- **Issue**: Contains 'night' but classified as Z
- **Expected**: B_mutual_intimacy, H_domestic_nesting
- **Keywords**: forgetting, silliness, requisite, friendliest, nachos, ingredient, implications, hospice, slumber, d
- **Severity**: medium

## Recommendations

1. **Run fix Z script** to reclassify topics with romance-relevant keywords
2. **Review regex patterns** for 'intercourse', 'mouth play', 'outfit'
3. **Update patterns** if needed to catch these cases

## Next Steps

1. Review identified issues
2. Run `fix_z_topics.py` if Z misclassifications found
3. Update regex patterns in `map_topics_to_categories.py` if needed
4. Re-run category mapping after fixes