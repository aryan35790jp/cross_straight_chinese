# -*- coding: utf-8 -*-
"""Parse the 1955 第一批异体字整理表 from Wikisource raw wikitext into a structured dataset."""
import re, json

raw = open('data/variant_table_1955_adjusted.txt', encoding='utf-8').read()

# Entries look like:  :庵［菴］   or  :暗［闇晻］
# orthodox char(s) before ［ ; abolished variants inside ［ ］ (fullwidth brackets)
# Some entries embed templates {{!|CHAR|desc}} for rare/IDS chars -> extract the CHAR (first arg)

def clean_segment(seg):
    """Replace {{!|X|...}} templates with X, strip other wiki markup, keep CJK + compat ideographs."""
    seg = re.sub(r'<ref[^>]*>.*?</ref>', '', seg, flags=re.S)  # drop editorial footnotes
    seg = re.sub(r'<ref[^>]*/>', '', seg)
    seg = re.sub(r'\{\{!\|([^|}]+)\|[^}]*\}\}', r'\1', seg)   # {{!|X|desc}} -> X
    seg = re.sub(r'&#x([0-9A-Fa-f]+);', lambda m: chr(int(m.group(1),16)), seg)  # numeric refs
    seg = re.sub(r'-\{|\}-', '', seg)  # conversion markers
    seg = re.sub(r'\[\[[^\]]*\]\]', '', seg)  # links
    return seg

# capture lines starting with ':' that contain fullwidth brackets ［ ］
line_re = re.compile(r'^:\s*(.+?)［(.+?)］', re.M)

groups = []
for m in line_re.finditer(raw):
    orth_seg = clean_segment(m.group(1))
    var_seg  = clean_segment(m.group(2))
    # orthodox = the CJK/compat chars in orth_seg (usually 1)
    orth = re.findall(r'[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002ffff]', orth_seg)
    variants = re.findall(r'[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002ffff]', var_seg)
    if orth and variants:
        groups.append({'orthodox': orth[0], 'variants': variants})

total_groups = len(groups)
total_variants = sum(len(g['variants']) for g in groups)
total_chars = total_groups + total_variants
print('groups parsed:', total_groups)
print('abolished variants:', total_variants)
print('total chars (orthodox+variants):', total_chars)

# group size distribution (members = 1 orthodox + variants)
from collections import Counter
sizes = Counter(1+len(g['variants']) for g in groups)
print('member-count distribution:', dict(sorted(sizes.items())))

# show last few to check we reached 'z' section
print('last 5 groups:', [(g['orthodox'], ''.join(g['variants'])) for g in groups[-5:]])
print('first 5 groups:', [(g['orthodox'], ''.join(g['variants'])) for g in groups[:5]])

json.dump(groups, open('data/variant_groups.json','w',encoding='utf-8'), ensure_ascii=False, indent=0)
