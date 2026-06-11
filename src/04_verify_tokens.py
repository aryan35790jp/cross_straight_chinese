# -*- coding: utf-8 -*-
"""Verify every character in the 796-group dataset gets an IDENTICAL token id across all
five models -> the weight-level claim cannot be a tokenization artifact."""
import json
from transformers import AutoTokenizer

groups = json.load(open('data/dataset_full.json', encoding='utf-8'))
chars = set()
for g in groups:
    chars.add(g['orthodox']); chars.update(g['variants'])
chars = sorted(chars)
print('characters in dataset:', len(chars))

MODELS = ["ckiplab/bert-base-chinese", "ckiplab/albert-base-chinese",
          "hfl/chinese-bert-wwm-ext", "hfl/chinese-macbert-base", "hfl/chinese-roberta-wwm-ext"]
toks = {m: AutoTokenizer.from_pretrained(m) for m in MODELS}
ref = MODELS[0]

unk = {m: toks[m].unk_token_id for m in MODELS}
mismatch = []      # chars with differing ids across models
unk_chars = {m: 0 for m in MODELS}
single_tok = {m: 0 for m in MODELS}
for ch in chars:
    ids = {}
    for m in MODELS:
        enc = toks[m].encode(ch, add_special_tokens=False)
        ids[m] = enc[0] if len(enc) == 1 else tuple(enc)
        if len(enc) == 1:
            single_tok[m] += 1
        if len(enc) == 1 and enc[0] == unk[m]:
            unk_chars[m] += 1
    if len(set(str(v) for v in ids.values())) > 1:
        mismatch.append((ch, ids))

print('single-token coverage per model:', single_tok)
print('UNK chars per model:', unk_chars)
print('characters with DIFFERING token id across the 5 models:', len(mismatch))
for ch, ids in mismatch[:15]:
    print('   ', ch, ids)
print()
print('VERDICT:', 'IDENTICAL token ids for all dataset chars across all 5 models'
      if not mismatch else f'{len(mismatch)} chars differ -> report these explicitly')
