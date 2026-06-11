# -*- coding: utf-8 -*-
"""ONE consistent pipeline: enriched typology (reading incl. tone | gloss | onomastic),
recompute population distinction-bearing AND the restoration experiment. Sanity-checked."""
import re, json, zipfile
from collections import defaultdict
from math import comb

z = zipfile.ZipFile('Unihan.zip')
R = defaultdict(dict)
for line in z.read('Unihan_Readings.txt').decode('utf-8').split('\n'):
    if line and not line.startswith('#'):
        p = line.split('\t')
        if len(p) >= 3:
            R[chr(int(p[0][2:], 16))][p[1]] = p[2]
mand = lambda c: (R.get(c, {}).get('kMandarin', '').split() or [''])[0]
gloss = lambda c: R.get(c, {}).get('kDefinition', '')
def tstrip(p): return p.translate(str.maketrans('āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ', 'aaaaeeeeiiiioooouuuuuuuu'))
STOP = set('the a an of to and or in on for with that this is are be as by from at used name kind variant same see also to make made into not no one two'.split())
def gtok(c):
    g = re.sub(r'[^a-z\s]', ' ', gloss(c).lower()); return set(t for t in g.split() if t not in STOP and len(t) > 2)
def onom(c): return bool(re.search(r'surname|\bplace\b', gloss(c), re.I))

def classify(orth, variants):
    oR, ot, o_onom = mand(orth), gtok(orth), onom(orth)
    reading = any(mand(v) and oR and mand(v) != oR for v in variants)          # segment OR tone
    hetero  = any(mand(v) and oR and tstrip(mand(v)) != tstrip(oR) for v in variants)
    gloss_d = any(gtok(v) and ot and not (gtok(v) & ot) for v in variants)
    onom_d  = any(onom(v) and not o_onom for v in variants)
    return (reading or gloss_d or onom_d), reading, hetero, gloss_d, onom_d

# ---------- adjusted population (consistent source) ----------
adj = json.load(open('data/variant_groups.json', encoding='utf-8'))
N = len(adj)
cb = sum(classify(g['orthodox'], g['variants'])[0] for g in adj)
cr = sum(classify(g['orthodox'], g['variants'])[1] for g in adj)
cg = sum(classify(g['orthodox'], g['variants'])[3] for g in adj)
co = sum(classify(g['orthodox'], g['variants'])[4] for g in adj)
print(f"POPULATION ({N} groups): distinction-bearing = {cb} ({100*cb/N:.1f}%)")
print(f"   reading(incl tone)={cr}  gloss-distinct={cg}  onomastic={co}")

# ---------- original 810 table (same clean as restoration_experiment) ----------
CJK = r'[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002ffff]'
def clean(seg):
    seg = re.sub(r'<ref[^>]*>.*?</ref>', '', seg, flags=re.S)
    seg = re.sub(r'\{\{!\|([^|}]+)\|[^}]*\}\}', r'\1', seg)
    seg = re.sub(r'&#x([0-9A-Fa-f]+);', lambda m: chr(int(m.group(1),16)), seg)
    seg = re.sub(r'-\{|\}-', '', seg); seg = re.sub(r'\[\[[^\]]*\]\]', '', seg)
    return seg
orig = []
for line in open('data/variant_table_1955_original.txt', encoding='utf-8').read().split('\n'):
    if not line.lstrip().startswith(':'): continue
    m = re.search(r'〔(.+?)〕', line)
    if not m: continue
    o = re.findall(CJK, clean(line[:m.start()])); v = re.findall(CJK, clean(m.group(1)))
    if o and v: orig.append({'orthodox': o[0], 'variants': v})
adj_chars = set()
for g in adj: adj_chars.update([g['orthodox']] + g['variants'])
restored = [(g['orthodox'], v) for g in orig for v in g['variants'] if v not in adj_chars]
print(f"\nSANITY: original groups parsed={len(orig)} (official 810); restored variants={len(restored)} (expected ~26-36)")

db = sum(classify(o, [v])[0] for o, v in restored)
n = len(restored); p0 = cb / N
pval = sum(comb(n,k)*p0**k*(1-p0)**(n-k) for k in range(db, n+1))
print(f"RESTORATION: {db}/{n} = {100*db/n:.1f}% distinction-bearing  (base {100*p0:.1f}%)  binomial p={pval:.2e}")
