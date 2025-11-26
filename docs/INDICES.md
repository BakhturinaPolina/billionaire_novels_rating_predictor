# Derived Indices

## Overview

This document defines all derived indices computed in Stage 07 analysis. Indices combine multiple thematic composites to quantify narrative qualities that may relate to reader appreciation.

## Index Definitions

### Love-over-Sex Index

**Formula:**
```
Love_over_Sex = (commitment_hea + tenderness_emotion) − explicit
```

**Interpretation:**
Measures the balance between emotional connection/commitment themes and explicit sexual content. Higher values indicate more emphasis on emotional intimacy relative to explicit content.

**Hypothesis (H1):** Higher in Top vs Trash novels.

**Components:**
- `commitment_hea`: Commitment and Happily Ever After themes
- `tenderness_emotion`: Emotional tenderness and intimacy
- `explicit`: Explicit sexual content (from `C_Explicit_Eroticism`)

---

### HEA Index

**Formula:**
```
HEA_Index = commitment_hea + symbolic_gifts_jewelry + festive_rituals
```

**Interpretation:**
Quantifies Happily Ever After (HEA) indicators. Combines commitment themes with symbolic gestures (gifts, jewelry) and festive rituals (proposals, weddings).

**Hypothesis (H2):** Higher in Top novels.

**Components:**
- `commitment_hea`: From `A_Reassurance_Commitment`
- `symbolic_gifts_jewelry`: From `G_Courtship_Rituals_Gifts` (gift-related)
- `festive_rituals`: From `G_Courtship_Rituals_Gifts` (ritual-related)

---

### Explicitness Ratio

**Formula:**
```
Explicitness_Ratio = explicit / (explicit + commitment_hea + tenderness_emotion + 1e-9)
```

**Interpretation:**
Proportion of explicit content relative to emotional themes. Values range from 0 (no explicit content) to 1 (only explicit content).

**Hypothesis (H2):** Higher in Trash novels.

**Components:**
- `explicit`: From `C_Explicit_Eroticism`
- `commitment_hea`: From `A_Reassurance_Commitment`
- `tenderness_emotion`: From `B_Mutual_Intimacy`
- `1e-9`: Small epsilon to prevent division by zero

---

### Luxury Saturation

**Formula:**
```
Luxury_Saturation = luxury_wealth + luxury_mobility_settings + 
                    luxury_consumption_style + nightlife_party_glamour
```

**Interpretation:**
Measures the presence of luxury and wealth themes. Combines wealth displays, luxury settings (jets, yachts), consumption patterns, and glamorous nightlife.

**Hypothesis (H3):** Predicts Top only when combined with high commitment/tenderness (interaction effect).

**Components:**
- `luxury_wealth`: Direct wealth themes from `D_Power_Wealth_Luxury`
- `luxury_mobility_settings`: Travel and mobility from `D_Power_Wealth_Luxury`
- `luxury_consumption_style`: Consumption patterns from `D_Power_Wealth_Luxury`
- `nightlife_party_glamour`: Glamorous settings from `D_Power_Wealth_Luxury` and `L_Vices_Addictions`

---

### Corporate Frame Share

**Formula:**
```
Corporate_Frame_Share = corporate_power + office_space + meetings_board
```

**Interpretation:**
Proportion of corporate/professional themes. Measures the presence of business settings, office environments, and corporate power dynamics.

**Components:**
- `corporate_power`: From `D_Power_Wealth_Luxury` (power aspect)
- `office_space`: From `K_Professional_Intrusion`
- `meetings_board`: From `K_Professional_Intrusion`

---

### Family/Fertility Index

**Formula:**
```
Family_Fertility_Index = family + fertility_pregnancy_baby + domestic_staff_childcare
```

**Interpretation:**
Family and fertility-related themes. Combines family relationships, pregnancy/baby themes, and domestic childcare settings.

**Components:**
- `family`: From `J_Social_Support_Kin`
- `fertility_pregnancy_baby`: From `M_Health_Recovery_Growth` (fertility aspect)
- `domestic_staff_childcare`: From `H_Domestic_Nesting`

---

### Comms Density

**Formula:**
```
Comms_Density = comms + public_image_scandal
```

**Interpretation:**
Communication and public image themes. Measures the presence of communication dynamics and public image/scandal themes.

**Components:**
- `comms`: Communication themes (may span multiple composites)
- `public_image_scandal`: From `P_Tech_Media_Presence`

---

### Dark-vs-Tender Index

**Formula:**
```
Dark_vs_Tender = (neg_affect + threat_violence_dark) − tenderness_emotion
```

**Interpretation:**
Balance between dark themes and tenderness. Higher values indicate more dark/negative themes relative to tenderness.

**Hypothesis (H5):** Lower in Top novels (i.e., Top novels are more tender).

**Components:**
- `neg_affect`: From `F_Angst_Negative_Affect`
- `threat_violence_dark`: From `E_Coercion_Brutality_Danger`
- `tenderness_emotion`: From `B_Mutual_Intimacy`

---

### Miscommunication Balance

**Formula:**
```
Miscommunication_Balance = (commitment_hea + tenderness_emotion + apology_repair) − miscommunication
```

**Interpretation:**
Resolution vs. conflict themes. Higher values indicate more resolution/repair themes relative to miscommunication.

**Hypothesis (H6):** Increases from begin → end of book.

**Components:**
- `commitment_hea`: From `A_Reassurance_Commitment`
- `tenderness_emotion`: From `B_Mutual_Intimacy`
- `apology_repair`: From `A_Reassurance_Commitment` (apology aspect)
- `miscommunication`: Communication conflicts (may span multiple composites)

---

### Protective–Jealousy Delta

**Formula:**
```
Protective_Jealousy_Delta = protectiveness_care − jealousy_possessiveness
```

**Interpretation:**
Caring protectiveness vs. jealous possessiveness. Higher values indicate more caring protectiveness relative to jealous possessiveness.

**Hypothesis (H4):** Higher in Top novels.

**Components:**
- `protectiveness_care`: Caring protectiveness (may combine `D_Power_Wealth_Luxury` protective aspects with `B_Mutual_Intimacy`)
- `jealousy_possessiveness`: From `F_Angst_Negative_Affect` (jealousy aspect)

---

## Composite Mapping

### Composite A: Reassurance/Commitment
- Apologies, agreements, commitments, promises, vows
- Marriage, engagement, rings
- Reassurance acts, consent & boundaries, rituals & milestones

### Composite B: Mutual Intimacy
- Kisses, tender affection, gaze, cuddling, aftercare
- Cozy intimacy, mutual emotional connection

### Composite C: Explicit Eroticism
- Explicit sexual content, arousal, orgasm, pleasure
- Performative intimacy framing

### Composite D: Power/Wealth/Luxury
- Business, money, luxury, wealth
- Jets, yachts, luxury settings
- Corporate power, travel & mobility
- Power displays

### Composite E: Coercion/Brutality/Danger
- Brutal, danger, coercion, weapons
- Security, revenge, jail, torture
- Violence, anger, threat

### Composite F: Angst/Negative Affect
- Conflict, doubt, anxiety, jealousy
- Guilt, betrayal, crying, tears
- Negative emotional states

### Composite G: Courtship Rituals/Gifts
- Courtship, dates, gifts, proposals
- Weddings, dances, symbolic gestures

### Composite H: Domestic Nesting
- Domestic, home, bedroom, kitchen
- House, cozy domestic settings

### Composite I: Humor/Lightness
- Laughter, joy, humor, banter
- Playful interactions

### Composite J: Social Support/Kin
- Family, siblings, friends
- Social circle & community

### Composite K: Professional Intrusion
- Meetings, office, appointments
- Schedule, time, work, job, boss
- Work & professional life

### Composite L: Vices/Addictions
- Alcohol, addiction, drugs, vice
- Nightlife, party scenes

### Composite M: Health/Recovery/Growth
- Health, medical, recovery, therapy
- Growth, development, fertility/pregnancy

### Composite N: Separation/Reunion
- Separation, goodbye, reunion, return
- Separation/reunion cues

### Composite O: Aesthetics/Appearance
- Clothes, appearance, makeup, jewelry
- Underwear, fashion, style

### Composite P: Tech/Media Presence
- Tech, phone, text, media
- Paparazzi, news, reports
- Public image, scandal

## Index Computation

### Per-Book Indices

All indices are computed at the book level using composite scores from Stage 06.

```python
# Example: Love-over-Sex Index
love_over_sex = (
    composites['A_Reassurance_Commitment'] + 
    composites['B_Mutual_Intimacy']
) - composites['C_Explicit_Eroticism']
```

### Per-Segment Indices (Time-Course)

For H6 (time-course hypothesis), indices are computed separately for:
- **Begin**: First third of book
- **Middle**: Middle third of book
- **End**: Final third of book

Segment-level topic probabilities are derived from chapter-level data or by splitting book into tertiles.

## Statistical Analysis

### Group Comparisons

Indices are compared across Top/Medium/Trash groups using:
- **ANOVA** (if normally distributed)
- **Kruskal-Wallis** (if non-normal)
- **Post-hoc tests** with Holm correction
- **Effect sizes**: Cohen's d

### Regression Models

Indices are used as predictors in:
- **Logistic regression**: Top (1) vs Trash (0)
- **OLS regression**: Average rating

### Interaction Effects

Key interactions tested:
- **Luxury × (Commitment+Tenderness)**: H3
- **Contractual × Tenderness**
- **PublicImage × Commitment**
- **Protective–Jealousy**: H4

---

For research hypotheses, see [SCIENTIFIC_README.md](../SCIENTIFIC_README.md).  
For data contracts, see [DATA_CONTRACTS.md](DATA_CONTRACTS.md).

