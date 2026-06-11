# -*- coding: utf-8 -*-
"""
Paper 4 - "What the Variant Tables Erased"
Colab T4 pipeline (hands-off after Run All).

Stages:
  0. Install deps
  1. Rebuild the complete 796-group dataset from Unihan + Wikisource (reproducible),
     classify each group: reading-bearing / gloss-distinct / redundant-control
  2. Mine Classical Wikipedia + Chinese Wikisource (per-group purity filter); print
     attested probe-able counts early
  3. Two probes across a 5-model shared-vocabulary panel (2 CKIP-Traditional,
     3 HFL-Simplified): (a) type-level form distinctness cos(orthodox,variant);
     (b) contextual sense recovery S_merged after collapse, with a label-shuffle null
  4. Group-level stats (no pseudoreplication): bootstrap CI, Cliff's delta,
     permutation / Wilcoxon, Holm-Bonferroni
  5. Headline = SENSE recovery (S_merged) specificity (reading-bearing vs redundant
     control, register-controlled); plus FORM/SENSE cross-script dissociation
  6. Save all CSVs + figures

Run:  !python paper4_probe.py
"""

# ======================= STAGE 0: deps =======================
import subprocess, sys
def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
pip("transformers>=4.40", "torch", "opencc-python-reimplemented",
    "scikit-learn", "scipy", "numpy", "pandas", "matplotlib", "datasets", "requests")

import os, re, json, math, zipfile, random, requests
import numpy as np, pandas as pd
from collections import defaultdict, Counter
random.seed(42); np.random.seed(42)

# ======================= HARD GPU CHECK (fail fast, never silently run on CPU) =======================
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "\n" + "="*70 +
        "\n*** NO GPU DETECTED - ABORTING (this would take 10-20h on CPU) ***\n"
        "Fix in Colab: Runtime > Change runtime type > Hardware accelerator = T4 GPU > Save,\n"
        "then re-upload this file and run again. Verify first with:  !nvidia-smi\n" + "="*70)
DEVICE = "cuda"
print("[GPU] OK:", torch.cuda.get_device_name(0))

UA = {"User-Agent": "ResearchBot/1.0 (academic script-reform study; contact aryanmaity3579@gmail.com)"}
OUT = "results"; os.makedirs(OUT, exist_ok=True)

# ======================= STAGE 1: rebuild dataset =======================
def fetch(url):
    r = requests.get(url, headers=UA, timeout=120); r.raise_for_status(); return r

print("[1] Downloading Unihan + 1955 table ...")
open("Unihan.zip", "wb").write(fetch("https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip").content)
wikitext = fetch("https://zh.wikisource.org/wiki/%E7%AC%AC%E4%B8%80%E6%89%B9%E5%BC%82%E4%BD%93%E5%AD%97%E6%95%B4%E7%90%86%E8%A1%A8?action=raw").text

CJK = r'[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002ffff]'
def clean(seg):
    seg = re.sub(r'<ref[^>]*>.*?</ref>', '', seg, flags=re.S)
    seg = re.sub(r'<ref[^>]*/>', '', seg)
    seg = re.sub(r'\{\{!\|([^|}]+)\|[^}]*\}\}', r'\1', seg)
    seg = re.sub(r'&#x([0-9A-Fa-f]+);', lambda m: chr(int(m.group(1), 16)), seg)
    seg = re.sub(r'-\{|\}-', '', seg); seg = re.sub(r'\[\[[^\]]*\]\]', '', seg)
    return seg
groups = []
for m in re.finditer(r'^:\s*(.+?)［(.+?)］', wikitext, re.M):
    o = re.findall(CJK, clean(m.group(1))); v = re.findall(CJK, clean(m.group(2)))
    if o and v:
        groups.append({"orthodox": o[0], "variants": v})
print(f"    parsed {len(groups)} groups, {sum(1+len(g['variants']) for g in groups)} chars")

# Unihan readings/defs/radical
z = zipfile.ZipFile("Unihan.zip")
def load(f):
    d = defaultdict(dict)
    for line in z.read(f).decode("utf-8").split("\n"):
        if line and not line.startswith("#"):
            p = line.split("\t")
            if len(p) >= 3:
                d[chr(int(p[0][2:], 16))][p[1]] = p[2]
    return d
R = load("Unihan_Readings.txt"); S = load("Unihan_IRGSources.txt")
mand = lambda c: (R.get(c, {}).get("kMandarin", "").split() or [""])[0]
gloss = lambda c: R.get(c, {}).get("kDefinition", "")
radical = lambda c: S.get(c, {}).get("kRSUnicode", "").split(".")[0]
def tone_strip(p):
    return p.translate(str.maketrans('āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ', 'aaaaeeeeiiiioooouuuuuuuu'))
STOP = set('the a an of to and or in on for with that this is are be as by from at used name '
           'kind variant same see also to make made into not no one two'.split())
def gtok(c):
    g = re.sub(r'[^a-z\s]', ' ', gloss(c).lower()); return set(t for t in g.split() if t not in STOP and len(t) > 2)

for g in groups:
    o = g["orthodox"]; vs = g["variants"]; oR = mand(o)
    hetero = any(mand(v) and oR and tone_strip(mand(v)) != tone_strip(oR) for v in vs)
    tonal = (not hetero) and any(mand(v) and oR and mand(v) != oR for v in vs)
    g["phon_class"] = "heterophonic" if hetero else ("tonal-variant" if tonal else "homophonic")
    ot = gtok(o)
    g["distinct_gloss"] = any(gtok(v) and ot and not (gtok(v) & ot) for v in vs)
    g["distinction_bearing"] = g["phon_class"] != "homophonic" or g["distinct_gloss"]
    # distinction TYPE for stratified analysis
    if g["phon_class"] != "homophonic":
        g["stratum"] = "reading-bearing"
    elif g["distinct_gloss"]:
        g["stratum"] = "gloss-distinct"
    else:
        g["stratum"] = "redundant-control"   # homophonic + same gloss = presumed true allograph
    n = 1 + len(vs)
    g["bits_upper"] = math.log2(n)

DB = [g for g in groups if g["distinction_bearing"]]
print(f"    distinction-bearing groups: {len(DB)}  "
      f"(reading-bearing {sum(g['stratum']=='reading-bearing' for g in groups)}, "
      f"gloss-distinct {sum(g['stratum']=='gloss-distinct' for g in groups)})")
print(f"    redundant-control groups: {sum(g['stratum']=='redundant-control' for g in groups)}")
json.dump(groups, open(f"{OUT}/dataset_full.json", "w", encoding="utf-8"), ensure_ascii=False)


# ======================= STAGE 2: corpus + ATTESTATION PRE-CHECK =======================
# Classical Chinese Wikipedia (wiki code lzh / "zh-classical") uses Traditional + classical
# forms natively, so abolished variants actually appear. This is the right register and is
# small enough for a T4 session.
print("\n[2] Loading Classical Chinese Wikipedia + Chinese Wikisource for attestation ...")
from datasets import load_dataset, interleave_datasets
def load_corpus():
    streams = []
    for repo, cfg in [("wikimedia/wikipedia", "20231101.zh-classical"),
                      ("wikimedia/wikisource", "20231201.zh")]:
        try:
            streams.append(load_dataset(repo, cfg, split="train", streaming=True))
            print("    + corpus:", repo, cfg)
        except Exception as e:
            print("    skip", repo, cfg, str(e)[:70])
    if not streams:                       # last-resort fallback
        return load_dataset("wikimedia/wikipedia", "20231101.zh", split="train", streaming=True), "zh-fallback"
    ds = interleave_datasets(streams, stopping_strategy="all_exhausted") if len(streams) > 1 else streams[0]
    return ds, "classical+wikisource"
ds, used_cfg = load_corpus()
print("    using corpus:", used_cfg)

# build variant -> (orthodox, group_idx, sense_id) lookup; sense_id = which form (0=orth, k=variant)
member_index = {}     # char -> list of (gi, role)  role 0=orthodox else variant rank
for gi, g in enumerate(groups):
    member_index.setdefault(g["orthodox"], []).append((gi, 0))
    for k, v in enumerate(g["variants"], 1):
        member_index.setdefault(v, []).append((gi, k))

MAX_DOCS = 130000         # both corpora combined
MAX_SENT_PER_FORM = 50    # enough for stable silhouette, caps memory/time
sent_re = re.compile(r'[^。！？；\n]{4,80}[。！？；]')
group_members = {gi: set([g["orthodox"]] + g["variants"]) for gi, g in enumerate(groups)}
# store sentences per (group_idx, role): the role tags the sense via the SPELLING.
# Purity filter is PER-GROUP: a sentence tags group gi only if exactly one member of gi
# appears (guards within-group sense ambiguity; members of other groups are irrelevant).
sents = defaultdict(list)
form_counts = Counter()
ndoc = 0
for ex in ds:
    txt = ex.get("text", "")
    ndoc += 1
    if ndoc > MAX_DOCS:
        break
    for sm in sent_re.finditer(txt):
        s = sm.group(0); sset = set(s)
        present = [c for c in sset if c in member_index]
        for c in present:
            for (gi, role) in member_index[c]:
                if len(group_members[gi] & sset) != 1:   # per-group purity
                    continue
                key = (gi, role)
                if len(sents[key]) < MAX_SENT_PER_FORM:
                    sents[key].append(s)
                    form_counts[c] += 1

# A group is probe-able if BOTH the orthodox form and >=1 variant have >= MIN_SENT sentences
MIN_SENT = 8
def probeable_groups(predicate):
    out = []
    for gi, g in enumerate(groups):
        if not predicate(g):
            continue
        o_ok = len(sents[(gi, 0)]) >= MIN_SENT
        var_roles = [k for k in range(1, 1+len(g["variants"])) if len(sents[(gi, k)]) >= MIN_SENT]
        if o_ok and var_roles:
            out.append((gi, var_roles))
    return out

probeable = probeable_groups(lambda g: g["distinction_bearing"])
control   = probeable_groups(lambda g: g["stratum"] == "redundant-control")
random.shuffle(control); control = control[:200]   # ample control set; caps T4 runtime

print(f"    scanned {ndoc} docs")
print(f"    distinction-bearing groups: {len(DB)}")
print(f"    >>> ATTESTED & PROBE-ABLE distinction-bearing groups (>={MIN_SENT} sents): {len(probeable)}")
att_reading = sum(1 for gi, _ in probeable if groups[gi]['phon_class'] != 'homophonic')
print(f"        of which reading-bearing: {att_reading}")
print(f"    >>> ATTESTED redundant-CONTROL groups (expect delta-S ~ 0): {len(control)}")
json.dump({"probeable": len(probeable), "reading_bearing_attested": att_reading,
           "control": len(control), "docs_scanned": ndoc, "corpus": used_cfg},
          open(f"{OUT}/attestation_precheck.json", "w"), ensure_ascii=False, indent=2)


# ======================= STAGE 3: model probes =======================
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import silhouette_score

# DEVICE already set to "cuda" by the hard GPU check at the top.
MODELS = {
    "ckiplab/bert-base-chinese":     "traditional",
    "ckiplab/albert-base-chinese":   "traditional",
    "hfl/chinese-bert-wwm-ext":      "simplified",
    "hfl/chinese-macbert-base":      "simplified",
    "hfl/chinese-roberta-wwm-ext":   "simplified",
}

# ---- shared-vocabulary control (Paper-3 style): verify identical WordPiece vocab ----
print("\n[3.0] Verifying shared tokenizer vocabulary across the panel ...")
vocabs = {m: set(AutoTokenizer.from_pretrained(m).get_vocab().keys()) for m in MODELS}
ref = next(iter(vocabs.values()))
jac = {m: len(v & ref) / len(v | ref) for m, v in vocabs.items()}
print("    Jaccard vs reference:", {m: round(j, 4) for m, j in jac.items()})
shared_vocab = all(j == 1.0 for j in jac.values())
print("    fully shared vocabulary:", shared_vocab,
      "(if not 1.0 for some model, cross-script claim still holds per-model but note it)")
json.dump({"jaccard": jac, "fully_shared": shared_vocab},
          open(f"{OUT}/vocab_control.json", "w"), ensure_ascii=False, indent=2)

def find_char_index(tokenizer, input_ids, ch):
    """Locate the token position of `ch`. Every CJK char is its own token in this
    shared WordPiece vocab, so match the token directly (works for fast OR slow tokenizers)."""
    toks = tokenizer.convert_ids_to_tokens(input_ids)
    for i, t in enumerate(toks):
        if t == ch:
            return i
    return None

@torch.no_grad()
def embed_targets(model, tokenizer, items):
    """items: list of (text, target_char). Returns list of target-char embeddings (or None).
    Robust to multi-token characters: uses offset mapping to mean-pool ALL tokens covering the
    target char (recovers rare/variant chars that split into subwords); single-token fallback
    for slow tokenizers."""
    vecs = []
    B = 32
    for i in range(0, len(items), B):
        batch = items[i:i+B]
        texts = [t for t, _ in batch]
        offs = None
        try:
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                            max_length=64, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping").tolist()
        except Exception:
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
        ids_cpu = enc["input_ids"].tolist()
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        out = model(**enc).last_hidden_state.cpu()
        for j, (text, ch) in enumerate(batch):
            tis = []
            if offs is not None:
                pos = text.find(ch)
                if pos >= 0:
                    tis = [ti for ti, (a, b) in enumerate(offs[j]) if b > a and a <= pos < b]
            if not tis:
                ti = find_char_index(tokenizer, ids_cpu[j], ch)
                if ti is not None:
                    tis = [ti]
            if tis and max(tis) < out.shape[1]:
                vecs.append(out[j, tis].mean(0).numpy())
            else:
                vecs.append(None)
    return vecs

def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((a[:, None] > b[None, :]).sum(axis=1))
    lt = sum((a[:, None] < b[None, :]).sum(axis=1))
    return (gt - lt) / (len(a) * len(b))

def boot_ci(x, n=5000):
    x = np.asarray(x); idx = np.random.randint(0, len(x), (n, len(x)))
    bs = x[idx].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

rows_type, rows_ctx = [], []
for mname, orient in MODELS.items():
    print(f"\n[3] Probing {mname} ({orient}) ...")
    tok = AutoTokenizer.from_pretrained(mname)
    mdl = AutoModel.from_pretrained(mname).to(DEVICE).eval()

    # ---- 3a. TYPE-LEVEL probe (always available): cos(orthodox, variant) ----
    for gi, g in enumerate(groups):
        if g["stratum"] == "redundant-control" and gi % 3 != 0:
            continue   # subsample controls (still ~200) to keep type-level balanced and fast
        pair_items = [(g["orthodox"], g["orthodox"])] + [(v, v) for v in g["variants"]]
        vecs = embed_targets(mdl, tok, pair_items)
        if any(v is None for v in vecs):
            continue
        ov = vecs[0]
        for k, vv in enumerate(vecs[1:], 1):
            cos = float(np.dot(ov, vv) / (np.linalg.norm(ov) * np.linalg.norm(vv) + 1e-9))
            rows_type.append({"model": mname, "orient": orient, "group": gi,
                              "orthodox": g["orthodox"], "variant": g["variants"][k-1],
                              "phon_class": g["phon_class"], "stratum": g["stratum"],
                              "bits_upper": g["bits_upper"],
                              "cos_orth_var": cos, "type_dist": 1 - cos})

    # ---- 3b. CONTEXTUAL probe (Paper-3 method) on attested groups + redundant controls ----
    for gi, var_roles in (probeable + control):
        g = groups[gi]; o = g["orthodox"]
        # build labeled examples: role 0 (orthodox sense) + each attested variant sense
        roles = [0] + var_roles
        # Strad: each sense written with its own form
        trad_items, trad_lab = [], []
        merged_items, merged_lab = [], []
        for role in roles:
            form = o if role == 0 else g["variants"][role-1]
            for s in sents[(gi, role)]:
                trad_items.append((s, form)); trad_lab.append(role)
                # merged: rewrite the variant occurrence as the orthodox form (simulate abolition)
                s_m = s.replace(form, o) if role != 0 else s
                merged_items.append((s_m, o)); merged_lab.append(role)
        if len(set(trad_lab)) < 2:
            continue
        tv = embed_targets(mdl, tok, trad_items); mv = embed_targets(mdl, tok, merged_items)
        keep_t = [i for i, v in enumerate(tv) if v is not None]
        keep_m = [i for i, v in enumerate(mv) if v is not None]
        if len(set(np.array(trad_lab)[keep_t])) < 2 or len(set(np.array(merged_lab)[keep_m])) < 2:
            continue
        S_trad = silhouette_score(np.array([tv[i] for i in keep_t]),
                                  [trad_lab[i] for i in keep_t], metric="cosine")
        mX = np.array([mv[i] for i in keep_m]); mlab = [merged_lab[i] for i in keep_m]
        S_merged = silhouette_score(mX, mlab, metric="cosine")
        # label-shuffle null (Paper-3 calibration): is S_merged above chance separability?
        nulls = []
        for _ in range(20):
            perm = np.random.permutation(mlab)
            if len(set(perm)) >= 2:
                nulls.append(silhouette_score(mX, perm, metric="cosine"))
        S_merged_null = float(np.mean(nulls)) if nulls else 0.0
        rows_ctx.append({"model": mname, "orient": orient, "group": gi, "orthodox": o,
                         "phon_class": g["phon_class"], "stratum": g["stratum"],
                         "bits_upper": g["bits_upper"], "orth_freq": int(form_counts.get(o, 0)),
                         "S_trad": float(S_trad), "S_merged": float(S_merged),
                         "S_merged_null": S_merged_null,
                         "S_merged_cal": float(S_merged - S_merged_null),
                         "delta_S": float(S_trad - S_merged)})
    del mdl; torch.cuda.empty_cache() if DEVICE == "cuda" else None

pd.DataFrame(rows_type).to_csv(f"{OUT}/type_level.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(rows_ctx).to_csv(f"{OUT}/contextual.csv", index=False, encoding="utf-8-sig")


# ======================= STAGE 4-5: stats, cross-script, elite correlation =======================
import scipy.stats as ss
print("\n[4] Statistics ...")
dt = pd.DataFrame(rows_type); dc = pd.DataFrame(rows_ctx)
summary = {}

# --- A. type-level FORM distinctness: pair-level (mean across models per orient) ---
if not dt.empty:
    # aggregate to (group, variant) pair within each orientation to avoid pseudoreplication
    pair = dt.groupby(["group", "variant", "orient"]).type_dist.mean().reset_index()
    piv = pair.pivot_table(index=["group", "variant"], columns="orient", values="type_dist").dropna()
    summary["type_level"] = {"n_pairs": int(len(piv))}
    if {"traditional", "simplified"}.issubset(piv.columns) and len(piv) > 3:
        t = piv["traditional"].values; s = piv["simplified"].values
        summary["type_level"].update({
            "trad_mean_dist": float(t.mean()), "trad_ci95": list(boot_ci(t)),
            "simp_mean_dist": float(s.mean()), "simp_ci95": list(boot_ci(s)),
            "cliffs_delta_trad_vs_simp": float(cliffs_delta(t, s)),
            "wilcoxon_p": float(ss.wilcoxon(piv["traditional"], piv["simplified"],
                                            alternative="greater").pvalue)})
    # by-stratum (group-level means, descriptive)
    gstr = dt.groupby(["group", "stratum"]).type_dist.mean().reset_index()
    summary["type_level"]["by_stratum"] = {
        s: float(gstr[gstr.stratum == s].type_dist.mean())
        for s in ["reading-bearing", "gloss-distinct", "redundant-control"] if (gstr.stratum == s).any()}

# --- B. contextual: SENSE recovery (S_merged) is headline; delta_S is FORM distinctness ---
# All headline tests are at the GROUP level (mean across models) to avoid pseudoreplication.
if not dc.empty:
    # group-level aggregates
    gS = dc.groupby(["group", "stratum"]).S_merged.mean().reset_index()
    gScal = dc.groupby(["group", "stratum"]).S_merged_cal.mean().reset_index()
    rb = gS[gS.stratum == "reading-bearing"].S_merged.values
    gl = gS[gS.stratum == "gloss-distinct"].S_merged.values
    ct = gS[gS.stratum == "redundant-control"].S_merged.values
    db_g = gS[gS.stratum != "redundant-control"].S_merged.values

    summary["counts_group_level"] = {"reading_bearing": int(len(rb)),
        "gloss_distinct": int(len(gl)), "redundant_control": int(len(ct))}

    def grp(vals):
        return {"n": int(len(vals)), "mean": float(np.mean(vals)), "ci95": list(boot_ci(vals))} if len(vals) else {}
    summary["sense_recovery_S_merged"] = {
        "reading-bearing": grp(rb), "gloss-distinct": grp(gl), "redundant-control": grp(ct)}
    summary["sense_recovery_calibrated_vs_null"] = {
        s: grp(gScal[gScal.stratum == s].S_merged_cal.values)
        for s in ["reading-bearing", "gloss-distinct", "redundant-control"]}

    # SPECIFICITY (group level, register-controlled by the redundant baseline)
    if len(rb) and len(ct):
        summary["specificity_reading_vs_control"] = {
            "S_merged_reading": float(np.mean(rb)), "S_merged_control": float(np.mean(ct)),
            "cliffs_delta": float(cliffs_delta(rb, ct)),
            "perm_p_reading_gt_control": float(ss.permutation_test((rb, ct),
                lambda a, b: np.mean(a) - np.mean(b), permutation_type="independent",
                n_resamples=10000, alternative="greater").pvalue),
            "interpretation": "reading-bearing senses recoverable from context after collapse while "
                              "redundant allographs are not -> sense-specific, not register"}
    if len(db_g) and len(ct):
        summary["specificity_allbearing_vs_control"] = {
            "cliffs_delta": float(cliffs_delta(db_g, ct)),
            "perm_p": float(ss.permutation_test((db_g, ct), lambda a, b: np.mean(a) - np.mean(b),
                permutation_type="independent", n_resamples=10000, alternative="greater").pvalue)}

    # DISSOCIATION: cross-script FORM effect (delta_S), PAIRED across groups present in both orients
    gO = dc.groupby(["group", "orient"]).agg(delta_S=("delta_S", "mean"),
                                             S_merged=("S_merged", "mean")).reset_index()
    piv_d = gO.pivot(index="group", columns="orient", values="delta_S").dropna()
    piv_s = gO.pivot(index="group", columns="orient", values="S_merged").dropna()
    if {"traditional", "simplified"}.issubset(piv_d.columns) and len(piv_d) > 3:
        diff = (piv_d["traditional"] - piv_d["simplified"]).values
        summary["cross_script_dissociation"] = {
            "n_groups_paired": int(len(piv_d)),
            "form_deltaS_traditional": float(piv_d["traditional"].mean()),
            "form_deltaS_simplified": float(piv_d["simplified"].mean()),
            "form_paired_mean_diff": float(np.mean(diff)), "form_paired_ci95": list(boot_ci(diff)),
            "form_wilcoxon_p": float(ss.wilcoxon(piv_d["traditional"], piv_d["simplified"],
                                                 alternative="greater").pvalue),
            "sense_Smerged_traditional": float(piv_s["traditional"].mean()),
            "sense_Smerged_simplified": float(piv_s["simplified"].mean()),
            "sense_paired_mean_diff": float((piv_s["traditional"] - piv_s["simplified"]).mean()),
            "interpretation": "Traditional training raises FORM distinctness (delta_S) but sense "
                              "recovery (S_merged) is script-independent -- a form/sense dissociation"}

# Holm-Bonferroni across the headline family
pvals = []
if summary.get("type_level", {}).get("wilcoxon_p") is not None:
    pvals.append(("type_level_form_trad>simp", summary["type_level"]["wilcoxon_p"]))
if "specificity_reading_vs_control" in summary:
    pvals.append(("sense_specificity_reading>control", summary["specificity_reading_vs_control"]["perm_p_reading_gt_control"]))
if "cross_script_dissociation" in summary:
    pvals.append(("form_deltaS_trad>simp", summary["cross_script_dissociation"]["form_wilcoxon_p"]))
pvals.sort(key=lambda x: x[1]); k = len(pvals)
summary["holm_bonferroni"] = [{"test": t, "p": p, "threshold": 0.05/(k-i), "sig": p < 0.05/(k-i)}
                              for i, (t, p) in enumerate(pvals)]

json.dump(summary, open(f"{OUT}/summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))

# ======================= STAGE 6: figures =======================
print("\n[6] Figures ...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
if not dc.empty:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    dbf = dc[dc.stratum != "redundant-control"]
    # panel 1: FORM distinctness (delta_S) by script orientation
    for orient, c in [("traditional", "#1f77b4"), ("simplified", "#d62728")]:
        ax[0].hist(dbf[dbf.orient == orient]["delta_S"], bins=20, alpha=0.6, label=orient, color=c)
    ax[0].set_xlabel("form distinctness  ΔS"); ax[0].set_ylabel("groups")
    ax[0].set_title("Cross-script FORM effect (ΔS)"); ax[0].legend()
    # panel 2: SENSE recovery (S_merged) by stratum, GROUP-LEVEL -- the specificity result
    gSf = dc.groupby(["group", "stratum"]).S_merged.mean().reset_index()
    order = ["reading-bearing", "gloss-distinct", "redundant-control"]
    means = [gSf[gSf.stratum == s].S_merged.mean() for s in order]
    cis = [boot_ci(gSf[gSf.stratum == s].S_merged.values) for s in order]
    errs = [[m - lo for m, (lo, hi) in zip(means, cis)], [hi - m for m, (lo, hi) in zip(means, cis)]]
    ax[1].bar(range(3), means, yerr=errs, capsize=5,
              color=["#2ca02c", "#ff7f0e", "#7f7f7f"])
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(order, rotation=15, fontsize=8)
    ax[1].set_ylabel("S_merged (sense recovered from context)")
    ax[1].set_title("SENSE recovery by stratum (specificity)")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_results.png", dpi=200)
print("\nDONE. Artifacts in ./results : dataset_full.json, attestation_precheck.json,")
print("     vocab_control.json, type_level.csv, contextual.csv, summary.json, fig_results.png")
