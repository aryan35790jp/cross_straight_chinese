# -*- coding: utf-8 -*-
"""
Full enrichment + information-theoretic measure for the 1955 variant-abolition study.
Adds: Kangxi radical (gloss validation), Taiwan/PRC encoding (cross-strait retention),
fixed restoration matching, and bits-of-erased-distinction (Paper-3-style).
"""
import zipfile, json, re, math
from collections import Counter, defaultdict

groups = json.load(open('data/variant_groups.json', encoding='utf-8'))
z = zipfile.ZipFile('Unihan.zip')

def load(fname):
    out = z.read(fname).decode('utf-8')
    d = defaultdict(dict)
    for line in out.split('\n'):
        if not line or line.startswith('#'):
            continue
        p = line.split('\t')
        if len(p) < 3:
            continue
        ch = chr(int(p[0][2:], 16))
        d[ch][p[1]] = p[2]
    return d

R = load('Unihan_Readings.txt')
S = load('Unihan_IRGSources.txt')
V = load('Unihan_Variants.txt')

def mand(ch):    return R.get(ch, {}).get('kMandarin', '').split()[0] if R.get(ch, {}).get('kMandarin') else ''
def gloss(ch):   return R.get(ch, {}).get('kDefinition', '')
def radical(ch): return S.get(ch, {}).get('kRSUnicode', '').split('.')[0]
def in_tw(ch):   return 'kIRG_TSource' in S.get(ch, {})   # encoded in Taiwan CNS 11643
def in_prc(ch):  return 'kIRG_GSource' in S.get(ch, {})   # encoded in PRC GB

# ---- restoration set: expand simplified restored chars to their traditional variants ----
restored_simp = set('阪挫䜣䜩晔詟诃䲡䌷刬鲙诓雠翦邱於澹骼彷菰溷徼薰黏桉愣晖凋鎔镕')
restored_all = set(restored_simp)
for ch in list(restored_simp):
    for fld in ('kTraditionalVariant', 'kSemanticVariant', 'kZVariant'):
        for tok in V.get(ch, {}).get(fld, '').split():
            m = re.match(r'U\+([0-9A-Fa-f]+)', tok)
            if m:
                restored_all.add(chr(int(m.group(1), 16)))

STOP = set('the a an of to and or in on for with that this is are be as by from at '
           'used name kind variant same see also to make made into not no one two'.split())
def gtok(ch):
    g = re.sub(r'[^a-z\s]', ' ', gloss(ch).lower())
    return set(t for t in g.split() if t not in STOP and len(t) > 2)

def tone_strip(p):
    return p.translate(str.maketrans('āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ','aaaaeeeeiiiioooouuuuuuuu'))

rows = []
H_upper = H_recov = H_irrecov = 0.0
H_valid = 0.0   # distinction-validated bits
for g in groups:
    orth = g['orthodox']; variants = g['variants']
    members = [orth] + variants
    n = len(members)
    reads = {m: mand(m) for m in members}

    # phonological class
    o = reads[orth]
    hetero = any(reads[v] and o and tone_strip(reads[v]) != tone_strip(o) for v in variants)
    tonal  = (not hetero) and any(reads[v] and o and reads[v] != o for v in variants)
    phon = 'heterophonic' if hetero else ('tonal-variant' if tonal else 'homophonic')

    # semantic distinction + radical cross-check
    ot = gtok(orth)
    distinct_gloss = any(gtok(v) and ot and not (gtok(v) & ot) for v in variants)
    o_rad = radical(orth)
    radical_differs = any(radical(v) and o_rad and radical(v) != o_rad for v in variants)

    restored = any(m in restored_all for m in members)

    # cross-strait retention: variant dropped by PRC but still encoded by Taiwan
    tw_keeps = [v for v in variants if in_tw(v)]
    tw_retains_dropped = len(tw_keeps) > 0

    # ---- information-theoretic (Paper-3 formula) ----
    # written-form uncertainty assuming each member a distinct sense (UPPER BOUND)
    h_up = math.log2(n)
    # pronunciation classes
    pclass = Counter(tone_strip(reads[m]) if reads[m] else f'?{m}' for m in members)
    h_irr = sum((c / n) * math.log2(c) for c in pclass.values())  # within-class residual
    H_upper += h_up; H_irrecov += h_irr; H_recov += (h_up - h_irr)
    # distinction-validated: count distinct senses = distinct (reading, gloss-cluster)
    distinct_senses = len(set((tone_strip(reads[m]) if reads[m] else m,) for m in members))
    h_val = math.log2(distinct_senses) if distinct_senses > 1 else 0.0
    H_valid += h_val

    rows.append({
        'orthodox': orth, 'variants': variants, 'n_members': n,
        'phon_class': phon, 'distinct_gloss': distinct_gloss,
        'radical_differs': radical_differs, 'restored': restored,
        'tw_retains_dropped_variant': tw_retains_dropped,
        'o_reading': o, 'var_readings': [reads[v] for v in variants],
        'bits_upper': round(h_up, 4), 'bits_irrecoverable': round(h_irr, 4),
        'bits_validated': round(h_val, 4),
    })

json.dump(rows, open('data/dataset_full.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=0)

N = len(rows)
bearing = [r for r in rows if r['phon_class'] != 'homophonic' or r['distinct_gloss'] or r['restored']]
het_tonal = sum(r['phon_class'] != 'homophonic' for r in rows)
tw_retain = sum(r['tw_retains_dropped_variant'] for r in rows)
restored_n = sum(r['restored'] for r in rows)
# gloss validation against radical
dg = [r for r in rows if r['distinct_gloss']]
dg_radical_agree = sum(r['radical_differs'] for r in dg)

print('='*64)
print(f'POPULATION: {N} groups, {sum(r["n_members"] for r in rows)} characters')
print('-'*64)
print('INFORMATION-THEORETIC (summed over all groups):')
print(f'  Upper-bound written uncertainty : {H_upper:8.1f} bits')
print(f'  Recoverable from pronunciation  : {H_recov:8.1f} bits ({100*H_recov/H_upper:.1f}%)')
print(f'  Irrecoverable (context-only)    : {H_irrecov:8.1f} bits ({100*H_irrecov/H_upper:.1f}%)')
print(f'  Distinction-validated bits      : {H_valid:8.1f} bits')
print('-'*64)
print('DISTINCTION-BEARING GROUPS:')
print(f'  reading-bearing (hetero+tonal)  : {het_tonal}')
print(f'  any signal (union)              : {len(bearing)} ({100*len(bearing)/N:.1f}%)')
print(f'  officially restored by PRC      : {restored_n}')
print('-'*64)
print('GLOSS-SIGNAL VALIDATION (vs Kangxi radical):')
print(f'  distinct-gloss groups           : {len(dg)}')
print(f'  of which radicals also differ   : {dg_radical_agree} ({100*dg_radical_agree/max(1,len(dg)):.1f}%)')
print('-'*64)
print('CROSS-STRAIT RETENTION:')
print(f'  groups where Taiwan (CNS 11643) still encodes a PRC-dropped variant: {tw_retain} ({100*tw_retain/N:.1f}%)')
