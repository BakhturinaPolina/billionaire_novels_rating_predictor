"""LLM prompts for topic labeling and category fixing."""

BASE_LABELING_PROMPT = """You are an expert in popular romance fiction analysis.

You will see one TOPIC extracted from a large corpus of romance novels.

For each topic you get:
- A short provisional topic label
- A list of top keywords
- Optionally: a short example snippet

Your tasks:

1. Decide whether the topic is mainly about ROMANTIC/THEMATIC content
   in popular romance, or mainly about non-romantic noise (sports,
   animals, random objects, etc.).

2. If it is romance-relevant, assign it to 1–3 of these categories,
   using the CODES exactly as written:

   A_commitment_hea        – commitment, HEA, promises, proposals, weddings
   B_mutual_intimacy       – non-explicit intimacy: kissing, cuddling, tenderness
   C_explicit              – explicit sexual acts, strong erotic content
   D_luxury_wealth_status  – wealth, luxury goods, high status consumption
   E_threat_danger         – coercion, danger, police, weapons, physical threat
   F_negative_affect       – fear, anxiety, guilt, anger, emotional pain
   G_rituals_gifts         – dates, gifts, flowers, holidays, courtship rituals
   H_domestic_nesting      – home, kitchen, bathroom, cozy interior, nesting
   I_humor_lightness       – jokes, banter, playful tone, comic relief
   J_family_support        – family, kinship, parenting, pregnancy, babies
   K_work_corporate        – office, workplace, corporate settings
   L_vices_addictions      – drinking, hangovers, benders, addictive behavior
   M_health_recovery       – illness, injury, hospitals, healing, recovery
   N_separation_reunion    – separation, distance, reunion, coming back
   O_appearance_aesthetics – looks, clothing, body, hair, perfume, aesthetics
   P_tech_media            – phones, texting, screens, media, tabloids
   Q_miscommunication      – misunderstandings, secrets, lies, wrong assumptions
   T_repair_apology        – apology, forgiveness, relationship repair
   R1_protectiveness       – caring protectiveness, guarding, shielding from harm
   R2_jealousy             – jealousy, possessiveness, envy of rivals
   S_scene_anchor          – generic scene locations (elevator, hallway, car, bar)
   Z_noise_oog             – out-of-domain noise (sports, animals, random stuff)

3. If the topic mixes romance and some noise, but at least about 30–40%
   of the meaning is romance-relevant, DO NOT use Z_noise_oog.
   Use Z_noise_oog ONLY when the topic is predominantly non-romantic.

4. For your natural-language label, prefer short phrases that include
   clear lexical cues for the categories, e.g. "Jealous Girlfriend",
   "Apology and Forgiveness", "Protective Bodyguard", "Office Romance".

Return your answer as strict JSON:

{
  "label": "<short topic label with clear cues>",
  "keywords": ["k1", "k2", "..."],
  "primary_categories": ["<code1>", "<code2>"],
  "secondary_categories": ["<optional_code3>"],
  "is_noise": true/false,
  "rationale": "<1–2 sentence explanation>"
}
"""

FIX_Z_EXAMPLES = """EXAMPLE 1 (true noise)
Input:
{"topic_id": 101, "label": "Ice Hockey Players",
 "keywords": ["hockey", "puck", "ice", "team"],
 "current_category": "Z_noise_oog"}

Output:
{"topic_id": 101,
 "label": "Ice Hockey Players",
 "primary_categories": ["Z_noise_oog"],
 "secondary_categories": [],
 "is_noise": true,
 "rationale": "This topic is about sports and athletic play, not about romantic relationships."}

EXAMPLE 2 (food but still noise)
Input:
{"topic_id": 102, "label": "Guacamole Dilemma",
 "keywords": ["avocado", "chips", "guacamole", "party"],
 "current_category": "Z_noise_oog"}

Output:
{"topic_id": 102,
 "label": "Guacamole Dilemma",
 "primary_categories": ["Z_noise_oog"],
 "secondary_categories": [],
 "is_noise": true,
 "rationale": "The content is about food and snacks rather than romance or emotional themes."}

EXAMPLE 3 (mis-labeled explicit romance)
Input:
{"topic_id": 37, "label": "Moist Mouth Play",
 "keywords": ["tongue", "mouth", "wet", "kissing"],
 "current_category": "Z_noise_oog"}

Output:
{"topic_id": 37,
 "label": "Wet Oral Foreplay",
 "primary_categories": ["C_explicit"],
 "secondary_categories": ["B_mutual_intimacy"],
 "is_noise": false,
 "rationale": "The focus is on erotic oral play and intimate kissing, a clear explicit sexual topic."}

EXAMPLE 4 (repair/apology)
Input:
{"topic_id": 78, "label": "Shameless Apologies",
 "keywords": ["sorry", "apologize", "regret"],
 "current_category": "Z_noise_oog"}

Output:
{"topic_id": 78,
 "label": "Apology and Forgiveness",
 "primary_categories": ["T_repair_apology"],
 "secondary_categories": ["F_negative_affect"],
 "is_noise": false,
 "rationale": "The topic is about apologizing and emotional repair within a relationship."}

EXAMPLE 5 (miscommunication)
Input:
{"topic_id": 8, "label": "Wrong Thoughts",
 "keywords": ["doubt", "misunderstand", "assumption"],
 "current_category": "Z_noise_oog"}

Output:
{"topic_id": 8,
 "label": "Misunderstood Feelings",
 "primary_categories": ["Q_miscommunication"],
 "secondary_categories": ["F_negative_affect"],
 "is_noise": false,
 "rationale": "The focus is on misunderstandings and self-doubt within a romantic context."}
"""

FIX_Z_PROMPT = """You are cleaning up topics from a romance topic model.

Each topic has already been classified by a rule-based system into
one of several romance-related categories (A–P, Q, T, R1, R2, S)
or into a noise category Z_noise_oog.

You will only see topics that were classified as Z_noise_oog
(noise / out-of-domain).

For each topic, do the following:

1. Decide if the topic is genuinely NON-ROMANTIC noise
   (e.g., sports, animals, food, random objects), or if it actually
   has clear romance-relevant content (relationships, emotions,
   sexual or emotional intimacy, weddings, family, etc.).

2. If it is genuinely non-romantic, keep it as Z_noise_oog.

3. If it *is* romance-relevant, reassign it to 1–3 of these categories
   (use codes exactly):

   A_commitment_hea
   B_mutual_intimacy
   C_explicit
   D_luxury_wealth_status
   E_threat_danger
   F_negative_affect
   G_rituals_gifts
   H_domestic_nesting
   I_humor_lightness
   J_family_support
   K_work_corporate
   L_vices_addictions
   M_health_recovery
   N_separation_reunion
   O_appearance_aesthetics
   P_tech_media
   Q_miscommunication
   T_repair_apology
   R1_protectiveness
   R2_jealousy
   S_scene_anchor
   Z_noise_oog  (only if mostly non-romantic)

Return strict JSON:

{
  "topic_id": <int>,
  "label": "<possibly improved label>",
  "primary_categories": ["<code1>", "<code2>"],
  "secondary_categories": ["<optional_code3>"],
  "is_noise": true/false,
  "rationale": "<brief explanation>"
}

Now here are some examples:

""" + FIX_Z_EXAMPLES + """

NOW CLASSIFY THIS TOPIC:

<TOPIC_JSON_HERE>
"""

