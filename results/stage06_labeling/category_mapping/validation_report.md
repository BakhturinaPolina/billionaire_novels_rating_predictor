# Category Mapping Validation Report

## Summary

- **Total topics analyzed**: 361
- **Z misclassifications found**: 7
- **Explicit/Intimacy issues**: 1
- **Regex pattern issues**: 3

## Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| A_commitment_hea | 11 | 3.0% |
| B_mutual_intimacy | 26 | 7.2% |
| C_explicit | 7 | 1.9% |
| D_luxury_wealth_status | 8 | 2.2% |
| E_threat_danger | 8 | 2.2% |
| F_negative_affect | 17 | 4.7% |
| G_rituals_gifts | 8 | 2.2% |
| H_domestic_nesting | 14 | 3.9% |
| I_humor_lightness | 13 | 3.6% |
| J_family_support | 25 | 6.9% |
| K_work_corporate | 23 | 6.4% |
| L_vices_addictions | 1 | 0.3% |
| M_health_recovery | 5 | 1.4% |
| N_separation_reunion | 2 | 0.6% |
| O_appearance_aesthetics | 24 | 6.6% |
| P_tech_media | 15 | 4.2% |
| Q_miscommunication | 12 | 3.3% |
| R1_protectiveness | 2 | 0.6% |
| R2_jealousy | 1 | 0.3% |
| S_scene_anchor | 17 | 4.7% |
| T_repair_apology | 4 | 1.1% |
| Z_noise_oog | 118 | 32.7% |

## ⚠️ Z Misclassifications (High Priority)

Topics classified as Z_noise_oog that contain romance-relevant keywords:

### Topic 26: Nighttime Stand
- **Current**: Z_noise_oog
- **Issue**: Contains 'nighttime' but classified as Z
- **Expected**: B_mutual_intimacy, H_domestic_nesting
- **Keywords**: tonight, night, evening, nights, stand, nighttime, fun, different, best, middle
- **Severity**: medium

### Topic 31: Tomorrow's Outfit
- **Current**: Z_noise_oog
- **Issue**: Contains 'outfit' but classified as Z
- **Expected**: O_appearance_aesthetics
- **Keywords**: clothes, outfit, wardrobe, clothing, outfits, tomorrow, selling, different, fashion, compatible
- **Severity**: medium

### Topic 39: Moist Mouth Play
- **Current**: Z_noise_oog
- **Issue**: Contains 'mouth play' but classified as Z
- **Expected**: B_mutual_intimacy, C_explicit
- **Keywords**: lip, teeth, nose, tongue, mouth, bridge, cheek, moan, lower, saliva
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

### Topic 32: Hesitant Intercourse
- **Current**: Z_noise_oog
- **Issue**: Contains 'intercourse' but classified as Z
- **Expected**: C_explicit, B_mutual_intimacy
- **Keywords**: offense, say, answer, head, worries, intercourse, plans, word, question, hesitation
- **Severity**: high

## Explicit vs Intimacy Classification Issues

### Topic 15: Intimate Touch
- **Current**: B_mutual_intimacy
- **Issue**: Contains explicit keywords but classified as B_mutual_intimacy only
- **Expected**: C_explicit, B_mutual_intimacy

## Regex Pattern Matching Issues

### Topic 32: Hesitant Intercourse
- **Current**: Z_noise_oog
- **Expected**: C_explicit
- **Issue**: Should be C_explicit but is Z_noise_oog
- **C_explicit patterns matched**: 1

### Topic 39: Moist Mouth Play
- **Current**: Z_noise_oog
- **Expected**: B_mutual_intimacy
- **Issue**: Should be B_mutual_intimacy but is Z_noise_oog
- **C_explicit patterns matched**: 1

### Topic 31: Tomorrow's Outfit
- **Current**: Z_noise_oog
- **Expected**: O_appearance_aesthetics
- **Issue**: Should be O_appearance_aesthetics but is Z_noise_oog

## Recommendations

1. **Run fix Z script** to reclassify topics with romance-relevant keywords
2. **Review regex patterns** for 'intercourse', 'mouth play', 'outfit'
3. **Update patterns** if needed to catch these cases
4. **Review explicit vs intimacy boundaries** - some topics may need split assignments

## Next Steps

1. Review identified issues
2. Run `fix_z_topics.py` if Z misclassifications found
3. Update regex patterns in `map_topics_to_categories.py` if needed
4. Re-run category mapping after fixes