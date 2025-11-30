#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Map topic labels -> theory-aligned categories (A–P + auxiliaries),
emit topic_to_category_probs.json / .csv and (optionally) book-level aggregates.

Inputs:
  --labels  labels_pos_openrouter_romance_aware_paraphrase-MiniLM-L6-v2.json
  --book-topic-probs  book_topic_probs.csv  (optional)
  --outdir  ./out

Outputs:
  out/topic_to_category_probs.json
  out/topic_to_category_final.csv
  [optional]
  out/book_category_props.csv
  out/indices_book.csv
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# ---- Category dictionary (keys = canonical category codes used downstream) ----
# These match your composites A–P and add cross-cut layers (Q, T, R) + scenes (S) + noise (Z).
CATS = [
    "A_commitment_hea",        # A
    "B_mutual_intimacy",       # B
    "C_explicit",              # C
    "D_luxury_wealth_status",  # D
    "E_threat_danger",         # E
    "F_negative_affect",       # F
    "G_rituals_gifts",         # G
    "H_domestic_nesting",      # H
    "I_humor_lightness",       # I
    "J_family_support",        # J
    "K_work_corporate",        # K
    "L_vices_addictions",      # L
    "M_health_recovery",       # M
    "N_separation_reunion",    # N
    "O_appearance_aesthetics", # O
    "P_tech_media",            # P
    "Q_miscommunication",     # cross-cut: miscommunication/deception
    "T_repair_apology",        # cross-cut: repair/apology/forgiveness
    "R1_protectiveness",      # cross-cut: protectiveness/care
    "R2_jealousy",            # cross-cut: jealousy/possessiveness
    "S_scene_anchor",         # auxiliary: scene locations
    "Z_noise_oog"             # out-of-domain: sports/animals/oddities
]

# ---- Rule lists: if label contains any trigger (case-insensitive), assign cat ----
# Triggers are derived from your generated labels. You can extend them safely.
TRIG = {
  "A_commitment_hea": [
    r"wedding", r"engagement", r"\bhea\b", r"vow", r"reunion", r"homecoming",
    r"eternal", r"proposal", r"commitment", r"bouquet", r"married", r"marriage",
    r"\bpromise\b"  # for "True Promise"
  ],
  "B_mutual_intimacy": [
    r"\bforeplay\b", r"\bbed(time|room) intimacy\b", r"\barmchair foreplay\b", r"\bfirst intimacy\b",
    r"\bintimate touch\b", r"\bcaress(es|ed|ing)?\b", r"\bkiss(es|ed|ing)?\b", r"\btender(ness)?\b",
    r"\bhug(s|ged|ging)?\b", r"\bwhisper(s|ed|ing)?\b"  # whispers as tender intimacy
  ],
  "C_explicit": [
    r"\bnipple(s)?\b", r"\bclit\b", r"\borgasm(s)?\b", r"\bdominatrix\b", r"\bcondom\b",
    r"\bwet nude\b", r"\bbreast play\b", r"\btongue & thumb\b",
    r"\bpussy\b", r"\berection\b", r"\bcock\b", r"\bpenetrat(e|ion)\b"
  ],
  "D_luxury_wealth_status": [
    r"lacy lingerie", r"wine service", r"millionaire|hollywood|luxury car|stainless|margarita",
    r"wealthy|richest|wealthiest|wealth|businessman|dollar|expensive|tailored"
  ],
  "E_threat_danger": [
    r"police|law enforcement|dangerous liaison|stalker|coercion|weapons|threat",
    r"danger|violence|dark|murder|gun|weapon"
  ],
  "F_negative_affect": [
    r"panic|outburst|angry|rage|guilty|shame|nightmare|sleep deprivation|wrong thoughts|ashamed",
    r"fear|anxiety|sad|tears|tearful|hurt|pain|guilt|anger|furious|frustration"
  ],
  "G_rituals_gifts": [
    r"bouquet|birthday|holiday|celebration|gift|dating app",
    r"flowers|roses|presents|ritual|ceremony|festive"
  ],
  "H_domestic_nesting": [
    r"\bbed(room|time)?\b", r"\bsheets?\b", r"\bkitchen\b", r"\bdinner\b",
    r"\bbath(room)?\b", r"\bshower\b", r"\bcozy\b"
  ],
  "I_humor_lightness": [
    r"playful|gleeful|goofy|witty|banter|jokes|spectacle",
    r"laugh|laughter|smile|grin|funny|humor|amusing"
  ],
  "J_family_support": [
    r"family|parents|children|pregnancy|baby|new parent|confidante",
    r"mother|father|sister|brother|kids|parenting|infant"
  ],
  "K_work_corporate": [
    r"office|board|manager|negotiation|taboo office kiss|presentation|workstation|ipo",
    r"business|company|corporate|meeting|job|work|ceo|employee"
  ],
  "L_vices_addictions": [
    r"\bdrunken\b", r"\bbender\b", r"\bhangover\b", r"\bdecadence\b", r"\bintoxicat"
  ],
  "M_health_recovery": [
    r"clinic|physician|fever|injur|tremulous reveal|medical",
    r"hospital|doctor|surgery|trauma|health|recovery|healing"
  ],
  "N_separation_reunion": [
    r"separation|apart|distance|reunion|homecoming",
    r"away|leaving|return|together|apart"
  ],
  "O_appearance_aesthetics": [
    r"gaze|brows|eye contact|hair play|lingerie|perfume|scent|mirror",
    r"appearance|looks|attractive|beautiful|handsome|stunning|aesthetic"
  ],
  "P_tech_media": [
    r"phone|phones|vibrating|ringing|webcam|tabloid|screen|message",
    r"tech|media|communication|device|cellphone|text|call"
  ],
  "Q_miscommunication": [
    r"\bmiscommunication\b", r"\blie\b", r"\bdeception\b", r"\bdoubts?\b",
    r"\bwrong\b", r"\bmistake\b", r"\bdeceive\b"
  ],
  "T_repair_apology": [
    r"\bapolog(y|ies|ize|ised|ised)\b", r"\bforgiveness\b", r"\brepair\b",
    r"\bsorry\b", r"\bapologetic\b"
  ],
  "R1_protectiveness": [
    r"\bprotect(?:ive|ion)?\b", r"\boverprotective\b", r"\bguardian\b", r"\bguard\b",
    r"\bcare\b"  # care as protectiveness (not jealousy)
  ],
  "R2_jealousy": [
    r"\bjealous(?:y)?\b", r"\bpossessive(?:ness)?\b", r"\benvious\b", r"\benvy\b"
  ],
  "S_scene_anchor": [
    r"elevator|doorways|driveway|airport|boat|barroom|sunset|beach|silent ride",
    r"door|car|vehicle|flight|plane|stairs|hallway|lobby"
  ],
  "Z_noise_oog": [
    r"\bhockey\b", r"\barchery\b", r"\banimals?\b", r"\bpigeons?\b",
    r"\bgame\b.*\b(hockey|baseball|football|soccer)\b"  # sports games
  ],
}


def compile_rules(TRIG):
    """Compile regex patterns for category matching."""
    comp = {k: [re.compile(pat, re.I) for pat in v] for k, v in TRIG.items()}
    return comp


RULES = compile_rules(TRIG)


def infer_categories(label: str, keywords: List[str] = None) -> Dict[str, float]:
    """
    Return a dict {cat: weight} with soft one-hot assignments.
    
    Uses both label and keywords for more accurate category inference.
    All regex patterns match against the combined text (label + keywords),
    ensuring that both the human-readable label and the underlying topic keywords
    contribute to category assignment.
    
    Args:
        label: Topic label string (e.g., "Tender Tongue Play")
        keywords: Optional list of keywords (e.g., ["tongue", "kiss", "tender", "soft"])
                 These are combined with the label for pattern matching.
    
    Returns:
        Dictionary mapping category codes to weights (sums to 1.0)
    """
    # Combine label and keywords into searchable text
    # This ensures both the label and keywords are considered for category inference
    if keywords:
        keywords_text = " " + " ".join(keywords) + " "
    else:
        keywords_text = ""
    
    # Create combined search text (label + keywords)
    # All regex patterns will match against this combined text
    search_text = label + keywords_text
    
    base_hits = []
    for cat, pats in RULES.items():
        # Check both label and keywords
        if any(p.search(search_text) for p in pats):
            base_hits.append(cat)
    
    hits = set(base_hits)
    
    # PATCH 1: Make Noise truly "last resort" - drop if other hits exist
    if "Z_noise_oog" in hits and len(hits) > 1:
        hits.discard("Z_noise_oog")
    
    # Priority logic: Handle conflicts and special cases
    
    # 1. Vices priority: If vices present, prefer L over H/D unless explicit home cues
    if "L_vices_addictions" in hits:
        homey = bool(re.search(r"\b(bed(room)?|sheets?|kitchen|dinner|bath(room)?|shower|cozy)\b", search_text, re.I))
        if homey:
            # Split L + H when home cues present
            return {"L_vices_addictions": 0.67, "H_domestic_nesting": 0.33}
        else:
            # L dominant when no home cues
            return {"L_vices_addictions": 1.0}
    
    # 2. Separation/Reunion: Nudge toward N when separation terms present
    if {"A_commitment_hea", "N_separation_reunion"} <= hits:
        if re.search(r"\b(separat|apart|distance|estrange)\w*\b", search_text, re.I):
            return {"N_separation_reunion": 0.7, "A_commitment_hea": 0.3}
    
    # 3. Miscommunication vs Repair: Split evenly when both present
    if "Q_miscommunication" in hits and "T_repair_apology" in hits:
        return {"Q_miscommunication": 0.5, "T_repair_apology": 0.5}
    
    # PATCH 2: Demote Scene anchors when core category is present
    if "S_scene_anchor" in hits and len(hits) > 1:
        hits_no_s = [h for h in hits if h != "S_scene_anchor"]
        w_other = 0.75 / len(hits_no_s)
        weights = {h: w_other for h in hits_no_s}
        weights["S_scene_anchor"] = 0.25
        return weights
    
    # PATCH 5: Preference in Luxury vs Appearance co-hits
    if {"D_luxury_wealth_status", "O_appearance_aesthetics"} <= hits:
        return {"D_luxury_wealth_status": 0.7, "O_appearance_aesthetics": 0.3}
    
    # 4. Remove H from hits if only triggered by "wine" (keep D or L instead)
    if "H_domestic_nesting" in hits and "D_luxury_wealth_status" in hits:
        if not re.search(r"\b(bed(room)?|sheets?|kitchen|dinner|bath(room)?|shower|cozy)\b", search_text, re.I):
            # Only wine triggered H, remove it
            hits.discard("H_domestic_nesting")
    
    # Default equal weights for remaining hits
    if hits:
        w = 1.0 / len(hits)
        return {h: w for h in hits}
    
    # Improved backoff order (reduces empty assignments)
    # Use search_text (label + keywords) for backoff matching too
    if re.search(r"\b(elevator|door|drive|airport|boat|bar(room)?|sunset|beach|ride)\b", search_text, re.I):
        return {"S_scene_anchor": 1.0}
    if re.search(r"\b(gaze|glance|smile|laugh|brows|eye|hair|perfume|scent|lingerie)\b", search_text, re.I):
        return {"O_appearance_aesthetics": 1.0}
    if re.search(r"\b(bed(room)?|sheets?|kitchen|dinner|bath(room)?|shower|cozy)\b", search_text, re.I):
        return {"H_domestic_nesting": 1.0}
    if re.search(r"\b(playful|witty|banter|goofy|gleeful|joke)\b", search_text, re.I):
        return {"I_humor_lightness": 1.0}
    if re.search(r"\b(promise|commitment|vow|proposal)\b", search_text, re.I):
        return {"A_commitment_hea": 1.0}
    # Romance-relevant backoffs using keywords
    if re.search(r"\b(whisper(s|ed|ing)?|breathy|husky)\b", search_text, re.I):
        return {"B_mutual_intimacy": 1.0}  # whispers as tender intimacy
    if re.search(r"\b(breath(s|ing|ed)?|breathy)\b", search_text, re.I):
        return {"B_mutual_intimacy": 1.0}  # breathing as intimacy cue
    if re.search(r"\b(emotional intimacy|intimacy|tender|sweet)\b", search_text, re.I):
        return {"B_mutual_intimacy": 1.0}
    # PATCH 6: Removed decision fallback - don't overuse Q_miscommunication for neutral decisions
    
    # Final fallback: Noise bucket
    return {"Z_noise_oog": 1.0}


def load_labels(fp: Path) -> Dict:
    """Load topic labels JSON file."""
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict, fp: Path):
    """Save object as JSON file."""
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(rows: List[Dict], fp: Path, fieldnames: List[str]):
    """Save rows as CSV file."""
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----- Optional aggregation and indices -----
def load_book_topic_probs(fp: Path) -> List[Dict]:
    """Load book-topic probability CSV."""
    rows = []
    with open(fp, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # Expect columns: book_id, topic_id, prob
            rows.append(row)
    return rows


def aggregate_book_props(book_topic_rows: List[Dict], topic2cat: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate topic probabilities to book-level category proportions.
    
    Args:
        book_topic_rows: List of dicts with book_id, topic_id, prob
        topic2cat: Dict mapping topic_id -> {cat: weight}
    
    Returns:
        Dict mapping book_id -> {cat: prob}
    """
    agg = defaultdict(lambda: defaultdict(float))
    for row in book_topic_rows:
        book = row["book_id"]
        tid = row["topic_id"]
        prob = float(row["prob"])
        if tid in topic2cat:
            for cat, w in topic2cat[tid].items():
                agg[book][cat] += prob * w
    return agg


def compute_indices(catvec: Dict[str, float]) -> Dict[str, float]:
    """
    Compute derived indices from category vector.
    
    Implements indices from SCIENTIFIC_README.md:
    - Love-over-Sex
    - HEA Index
    - Explicitness Ratio
    - Luxury Saturation
    - Corporate Frame Share
    - Family/Fertility Index
    - Comms Density
    - Dark-vs-Tender
    - Miscommunication Balance
    - Protective–Jealousy Delta
    """
    g = lambda k: catvec.get(k, 0.0)
    
    # Map cats to composite building blocks:
    commitment_hea = g("A_commitment_hea")
    tenderness_emotion = g("B_mutual_intimacy")
    explicit = g("C_explicit")
    luxury = g("D_luxury_wealth_status")
    neg_affect = g("F_negative_affect")
    threat_dark = g("E_threat_danger")
    comms = g("P_tech_media")
    family = g("J_family_support")
    
    # Proxies for subcomponents (can refine if you split subcodes later):
    symbolic_gifts = g("G_rituals_gifts")
    festive_rituals = g("G_rituals_gifts")
    corporate_share = g("K_work_corporate")
    
    # Q and T are now separate categories (no longer split)
    miscommunication = g("Q_miscommunication")
    repair_apology = g("T_repair_apology")
    
    # PATCH 3: R is now split into R1 and R2 for true delta computation
    protectiveness = g("R1_protectiveness")
    jealousy = g("R2_jealousy")

    out = {}
    out["Love_over_Sex"] = (commitment_hea + tenderness_emotion) - explicit
    out["HEA_Index"] = commitment_hea + symbolic_gifts + festive_rituals
    denom = explicit + commitment_hea + tenderness_emotion + 1e-9
    out["Explicitness_Ratio"] = explicit / denom
    out["Luxury_Saturation"] = luxury
    out["Corporate_Frame_Share"] = corporate_share
    out["Family_Fertility_Index"] = family
    out["Comms_Density"] = comms  # add public image if split later
    out["Dark_vs_Tender"] = (neg_affect + threat_dark) - tenderness_emotion
    out["Miscommunication_Balance"] = (commitment_hea + tenderness_emotion + repair_apology) - miscommunication
    out["Protective_minus_Jealousy"] = protectiveness - jealousy
    
    return out


def main():
    """Main entry point for category mapping."""
    ap = argparse.ArgumentParser(
        description="Map topic labels to theory-aligned categories (A-P, Q, R, S)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--labels", required=True, type=Path, help="Path to labels JSON file")
    ap.add_argument("--book-topic-probs", type=Path, help="Optional: book-topic probability CSV")
    ap.add_argument("--outdir", type=Path, default=Path("out"), help="Output directory")
    args = ap.parse_args()

    topics = load_labels(args.labels)  # {topic_id: {"label": str, "keywords": [...]}}
    topic_to_cat = {}
    rows = []

    for tid, rec in topics.items():
        label = rec.get("label", "")
        keywords = rec.get("keywords", [])
        cats = infer_categories(label, keywords=keywords)
        topic_to_cat[str(tid)] = cats
        
        # Find primary category (highest weight)
        primary_cat = None
        if cats:
            primary_cat = max(cats.items(), key=lambda x: x[1])[0]
        
        # Flatten for CSV
        keywords_str = ", ".join(keywords) if keywords else ""
        if not cats:
            rows.append({
                "topic_id": tid, 
                "label": label, 
                "keywords": keywords_str,
                "category": "", 
                "weight": "0.000000",
                "is_primary": "False"
            })
        else:
            for c, w in cats.items():
                rows.append({
                    "topic_id": tid, 
                    "label": label, 
                    "keywords": keywords_str,
                    "category": c, 
                    "weight": f"{w:.6f}",
                    "is_primary": "True" if c == primary_cat else "False"
                })

    outdir = args.outdir
    save_json(topic_to_cat, outdir / "topic_to_category_probs.json")
    save_csv(rows, outdir / "topic_to_category_final.csv",
             fieldnames=["topic_id", "label", "keywords", "category", "weight", "is_primary"])
    
    # Also create a summary CSV with one row per topic (primary category only)
    summary_rows = []
    for tid, rec in topics.items():
        label = rec.get("label", "")
        keywords = rec.get("keywords", [])
        keywords_str = ", ".join(keywords) if keywords else ""
        cats = topic_to_cat.get(str(tid), {})
        
        if cats:
            # Get primary category
            primary_cat = max(cats.items(), key=lambda x: x[1])[0]
            primary_weight = cats[primary_cat]
            # Get all categories as comma-separated string
            all_cats = ", ".join([f"{c}({w:.3f})" for c, w in sorted(cats.items(), key=lambda x: x[1], reverse=True)])
        else:
            primary_cat = ""
            primary_weight = 0.0
            all_cats = ""
        
        summary_rows.append({
            "topic_id": tid,
            "label": label,
            "keywords": keywords_str,
            "primary_category": primary_cat,
            "primary_weight": f"{primary_weight:.6f}",
            "all_categories": all_cats
        })
    
    save_csv(summary_rows, outdir / "topic_to_category_summary.csv",
            fieldnames=["topic_id", "label", "keywords", "primary_category", "primary_weight", "all_categories"])

    # Optional: aggregate to book-level and compute indices
    if args.book_topic_probs:
        book_rows = load_book_topic_probs(args.book_topic_probs)
        agg = aggregate_book_props(book_rows, topic_to_cat)
        # Save book category props
        b_rows = []
        for book, vec in agg.items():
            for c in CATS:
                b_rows.append({"book_id": book, "category": c, "value": f"{vec.get(c,0.0):.6f}"})
        save_csv(b_rows, outdir / "book_category_props.csv",
                 fieldnames=["book_id", "category", "value"])
        # Indices
        i_rows = []
        for book, vec in agg.items():
            idx = compute_indices(vec)
            i_rows.append({"book_id": book, **{k: f"{v:.6f}" for k, v in idx.items()}})
        # fieldnames stable order
        fns = ["book_id", "Love_over_Sex", "HEA_Index", "Explicitness_Ratio", "Luxury_Saturation",
               "Corporate_Frame_Share", "Family_Fertility_Index", "Comms_Density",
               "Dark_vs_Tender", "Miscommunication_Balance", "Protective_minus_Jealousy"]
        save_csv(i_rows, outdir / "indices_book.csv", fieldnames=fns)


if __name__ == "__main__":
    main()

