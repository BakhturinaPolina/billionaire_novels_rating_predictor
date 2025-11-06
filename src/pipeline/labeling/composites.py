#!/usr/bin/env python3
import pandas as pd, numpy as np, re
from pathlib import Path

PREP = "prepared_books.parquet"
MANUAL_MD = "Billionaire_Manual_Mapping_Topics_by_Thematic_Clusters.md"
CODEBOOK_CSV = "focused_topic_codebook.csv"  # optional
OUT_PARQUET = "ap_composites.parquet"

COMP = ['A_Reassurance_Commitment','B_Mutual_Intimacy','C_Explicit_Eroticism','D_Power_Wealth_Luxury',
        'E_Coercion_Brutality_Danger','F_Angst_Negative_Affect','G_Courtship_Rituals_Gifts','H_Domestic_Nesting',
        'I_Humor_Lightness','J_Social_Support_Kin','K_Professional_Intrusion','L_Vices_Addictions',
        'M_Health_Recovery_Growth','N_Separation_Reunion_Cues','O_Aesthetics_Appearance','P_Tech_Media_Presence']

def build_mapping(topic_cols, manual_text, codebook):
    topic_set = set(topic_cols)
    cur=None; t2cluster={}; clusters=set()
    for line in manual_text.splitlines():
        if re.match(r'^\\s{0,3}#{2,3}\\s', line) or re.match(r'^\\s*\\*\\*.*\\*\\*\\s*$', line):
            head = re.sub(r'^[#\\s]+','',line).strip().strip('*').strip()
            if head:
                cur=head; clusters.add(head); 
            continue
        if cur:
            low = line.lower()
            for t in topic_set:
                if re.search(r'\\b'+re.escape(t.lower())+r'\\b', low):
                    t2cluster.setdefault(t,set()).add(cur)
    c2c = {}
    for cl in clusters:
        cll = cl.lower(); dest=[]
        if any(k in cll for k in ["affirm","agreement","apolog","commit","promise","vow","marriage","engagement","ring"]): dest.append('A_Reassurance_Commitment')
        if any(k in cll for k in ["kiss","tender","affection","gaze","cuddle","aftercare","intimacy","cozy"]): dest.append('B_Mutual_Intimacy')
        if any(k in cll for k in ["sex","arousal","explicit","orgasm","pleasure"]): dest.append('C_Explicit_Eroticism')
        if any(k in cll for k in ["business","money","luxury","wealth","jet","yacht","paparazzi","press","news","camera","air travel"]): dest.append('D_Power_Wealth_Luxury')
        if any(k in cll for k in ["brutal","danger","coerc","weapon","security","revenge","jail","torture","violence","anger"]): dest.append('E_Coercion_Brutality_Danger')
        if any(k in cll for k in ["conflict","doubt","anxiety","jealous","guilt","betray","cry","tears"]): dest.append('F_Angst_Negative_Affect')
        if any(k in cll for k in ["courtship","date","gift","proposal","wedding","dance"]): dest.append('G_Courtship_Rituals_Gifts')
        if any(k in cll for k in ["domestic","home","bedroom","kitchen","house","cozy"]): dest.append('H_Domestic_Nesting')
        if any(k in cll for k in ["laughter","joy","humor","banter","playful"]): dest.append('I_Humor_Lightness')
        if any(k in cll for k in ["family","sibling","friend","social"]): dest.append('J_Social_Support_Kin')
        if any(k in cll for k in ["meeting","office","appointment","schedule","time","work","job","boss"]): dest.append('K_Professional_Intrusion')
        if any(k in cll for k in ["alcohol","addiction","drug","vice","nightlife"]): dest.append('L_Vices_Addictions')
        if any(k in cll for k in ["health","medical","recovery","therapy","growth","development"]): dest.append('M_Health_Recovery_Growth')
        if any(k in cll for k in ["separation","goodbye","reunion","return"]): dest.append('N_Separation_Reunion_Cues')
        if any(k in cll for k in ["clothes","appearance","makeup","jewelry","underwear","fashion","style"]): dest.append('O_Aesthetics_Appearance')
        if any(k in cll for k in ["tech","phone","text","media","paparazzi","news","report"]): dest.append('P_Tech_Media_Presence')
        if dest: c2c[cl]=dest
    t2comps = {t:set() for t in topic_cols}
    for t, cls in t2cluster.items():
        for cl in cls:
            for c in c2c.get(cl, []):
                t2comps[t].add(c)
    if codebook is not None and not codebook.empty:
        for _, r in codebook.iterrows():
            t = r['Topic']; cat = str(r['PrimaryCategory']); sub = str(r.get('IntimacyFramingSubtype',""))
            if not t2comps.get(t):
                if cat in ["Reassurance Acts","Consent & Boundaries","Rituals & Milestones"]: t2comps.setdefault(t,set()).add('A_Reassurance_Commitment')
                if cat=="Courtship Rituals & Dates": t2comps.setdefault(t,set()).add('G_Courtship_Rituals_Gifts')
                if cat=="Domesticity & Bonding": t2comps.setdefault(t,set()).add('H_Domestic_Nesting')
                if cat=="Humor & Lightness": t2comps.setdefault(t,set()).add('I_Humor_Lightness')
                if cat=="Work & Professional Life": t2comps.setdefault(t,set()).add('K_Professional_Intrusion')
                if cat=="Travel & Mobility": t2comps.setdefault(t,set()).add('D_Power_Wealth_Luxury')
                if cat=="Social Circle & Community": t2comps.setdefault(t,set()).add('J_Social_Support_Kin')
                if cat=="Conflict & Angst": t2comps.setdefault(t,set()).add('F_Angst_Negative_Affect')
                if cat=="Power Displays": t2comps.setdefault(t,set()).add('D_Power_Wealth_Luxury')
                if cat=="Intimacy Framing":
                    if "Mutual" in sub: t2comps.setdefault(t,set()).add('B_Mutual_Intimacy')
                    elif "Performative" in sub: t2comps.setdefault(t,set()).add('C_Explicit_Eroticism')
                    else:
                        t2comps.setdefault(t,set()).update(['B_Mutual_Intimacy','C_Explicit_Eroticism'])
                if cat=="Vices/Addictions": t2comps.setdefault(t,set()).add('L_Vices_Addictions')
                if cat=="Health/Recovery & Growth": t2comps.setdefault(t,set()).add('M_Health_Recovery_Growth')
                if cat=="Aesthetics & Appearance": t2comps.setdefault(t,set()).add('O_Aesthetics_Appearance')
                if cat=="P Tech/Media Presence": t2comps.setdefault(t,set()).add('P_Tech_Media_Presence')
    return t2comps

def main():
    df = pd.read_parquet(PREP)
    non_topic = {'Book_Title','Title_key','Title','Author','RatingsCount','Score','Popularity_ReadingNow','Popularity_Wishlisted','Pages','PublishedDate','popularity_index','Group','Year'} | {c for c in df.columns if c.startswith('z_')}
    topic_cols = [c for c in df.columns if c not in non_topic]
    manual_text = Path(MANUAL_MD).read_text(encoding='utf-8', errors='ignore') if Path(MANUAL_MD).exists() else ""
    codebook = pd.read_csv(CODEBOOK_CSV) if Path(CODEBOOK_CSV).exists() else None
    mapping = build_mapping(topic_cols, manual_text, codebook)
    comps = pd.DataFrame({'Title': df['Title'], 'Author': df['Author'], 'Group': df['Group'], 'Pages': df['Pages'], 'Year': df['Year']})
    for c in COMP: comps[c]=0.0
    weighted = {"Woman Pleasure and Arousal": {'C_Explicit_Eroticism':0.7, 'B_Mutual_Intimacy':0.3}}
    for t in topic_cols:
        if t in weighted: dests = weighted[t]
        else: dests = {d:1.0/len(mapping[t]) for d in mapping.get(t, [])} if mapping.get(t) else {}
        for dst, w in dests.items():
            comps[dst] += df[t].fillna(0)*w
    comps.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {OUT_PARQUET}")

if __name__ == "__main__":
    main()
