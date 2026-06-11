# What the Variant Tables Erased

Code, data, and results for the study of the 1955 *First List of Processed
Variant Characters* (第一批异体字整理表): what distinctions the variant abolition
removed, and whether language models carry a trace of them.

The reform grouped characters judged to be variants of one another, chose one
orthodox form per group, and abolished the rest (810 groups, 1,055 characters
removed; 796 groups after later adjustments). Taiwan never adopted the list and
keeps the abolished forms as standard. This repository reconstructs the complete
population from primary sources, quantifies what the abolition removed, and
probes five language models that share an identical token vocabulary.

## Repository layout

```
paper/      LaTeX source (main.tex) and the results figure
src/        data construction and analysis code
data/       the dataset and the source tables
results/    outputs of the verified model-probe run
```

- `src/01_parse_table.py` parses the adjusted 1955 table into groups.
- `src/02_build_dataset.py` enriches each group with Unihan readings, glosses,
  and radicals, and writes the dataset.
- `src/03_typology_and_restoration.py` computes the typology, the
  information-theoretic measure, and the restoration experiment.
- `src/04_verify_tokens.py` checks that all dataset characters receive an
  identical token id across the five models (the shared-vocabulary control).
- `src/05_probe_models_colab.py` is the self-contained model-probe pipeline. It
  downloads its own inputs and runs end to end on a single Colab T4 GPU in under
  two hours. This is the canonical reproduction entry point.

## Data sources

- The 1955 table (original and adjusted), from the public Wikisource text.
- The Unicode Han Database (Unihan), for readings, glosses, and radicals.
- Classical Chinese Wikipedia and Chinese Wikisource, for the probe corpus.
- Five public encoders: two Traditional-trained (CKIP `bert-base-chinese`,
  `albert-base-chinese`) and three Simplified-trained (HFL `chinese-bert-wwm-ext`,
  `chinese-macbert-base`, `chinese-roberta-wwm-ext`).

`Unihan.zip` is not committed (it is large and downloadable). `src/02` and
`src/03` expect it in the working directory; `src/05` downloads it automatically.

## Reproducing

```
pip install -r requirements.txt
# full model probe (downloads everything, needs a GPU):
python src/05_probe_models_colab.py
# dataset + typology + restoration (needs Unihan.zip in the working dir):
python src/01_parse_table.py
python src/02_build_dataset.py
python src/03_typology_and_restoration.py
```

## Key numbers

- Population: 796 groups, 1,821 characters; 204 (25.6%) distinction-bearing.
- 922.2 bits of upper-bound written-form uncertainty; 90.9% unrecoverable from
  pronunciation.
- Form: Traditional models keep abolished forms distinct from their orthodox
  counterpart far more than Simplified models (Cliff's delta = 0.78).
- Sense: recovery from context is specific to genuine distinctions
  (Cliff's delta = 0.26, p = 0.02) and does not differ by script.
- Restoration: variants the state later restored are distinction-bearing at more
  than twice the base rate (p < 0.001).

## License

Code under MIT. The 1955 table text is in the public domain; Unihan is
distributed under the Unicode license.
