# same-fim: Similarity-Aware Monotonic Entropy for FWER-controlled association-rule mining without a minimum-support threshold

[![DOI](https://zenodo.org/badge/1215897959.svg)](https://doi.org/10.5281/zenodo.19663187)
[![PyPI](https://img.shields.io/pypi/v/same-fim.svg)](https://pypi.org/project/same-fim/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Python reference implementation of **SAME**, a frequent-itemset miner that
(1) derives its support thresholds from the information content of the data,
and (2) attaches a Tarone--Bonferroni FWER guarantee to every returned rule.

Full method, theorems, and evaluation: Benarab, Necir & Aoures, "SAME: Similarity-Adaptive
Monotonic Entropy Based Method for Frequent Itemsets Extraction" (2026, under
review at *Big Data Mining and Analytics*).

## Install

```bash
pip install same-fim
```

Optional baselines used in the paper's benchmark:

```bash
pip install "same-fim[baselines]"
```

## Minimal example

```python
import pandas as pd
from same_fim import SAME

df = pd.read_csv("my_binary_data.csv").astype("int8")

# parameter-free mode: alpha and persistence are derived from the data
est = SAME(auto_hyperparams=True, search_mode="dfs", max_k=5)
est.fit(df.values, feature_names=list(df.columns))

for r in est.result_.rules:
    if r.passes_fwer:
        print(r)
```

## Command-line

```bash
same-mine --input data.csv --out rules.csv --mode dfs --auto --fwer-only
```

## Reproducing the paper

From a checkout of the repository:

```bash
pip install -e ".[baselines,dev]"
python experiments/reproduce.py
```

This runs the domain benchmark on the five datasets (ABIDE, EEG Eye State,
synth_neuro, ClinVar, Pfam-UniProt), the scaling study (`n` up to `10^6`),
the downstream classification probes on ABIDE and EEG, the
auto-hyperparameter ablation, and regenerates every figure in the paper's
`fig_v2/` directory.

### Seeded variance run (Table `tab:variance`)

The single-seed numbers in the main tables are supplemented by a 5-seed
variance run whose output populates `\TBD{...}` placeholders in the LaTeX:

```bash
python experiments/bench_seeded.py \
    --seeds 5 --timeout 1800 \
    --datasets abide eeg synth_neuro clinvar pfam \
    --methods same_dfs same_opus apriori apriori_bonferroni fpgrowth \
    --out results/variance.csv

# From ../EINF-PAPER/, inject the CSV values into paper.tex in-place.
python wire_variance.py --csv ../SAME_v4/experiments/results/variance.csv \
                       --tex paper.tex
```

`wire_variance.py` writes a `.bak` on first run and is idempotent: re-running
with a refreshed CSV overwrites any previously substituted cells.

### Apriori + post-hoc Bonferroni baseline

```bash
python experiments/apriori_bonferroni.py \
    --csv datasets/abide.csv --sigma 0.10 --alpha 0.05 \
    --method bonferroni --out apriori_bonf_abide.csv
```

This is the reviewer-requested isolation of "adaptive threshold" from "FWER
correction": same Fisher test and same `alpha` as SAME, differing only in
the support threshold. See [paper.tex `tab:baselines_ext`] for the
side-by-side at `sigma = 0.10`.

### Docker baselines

Dockerfiles for LAMP, SPuManTE, WYlight, OPUS Miner, SPMF, and Kingfisher
are in `experiments/baselines_ext/docker/`. See that directory's README for
per-image build instructions and the running order.

## Core guarantees

SAME returns association rules with:

- **A data-derived support threshold** combining a LAMP-style base floor
  `s_0`, a Webb (2007) layered per-level decay, a Hoeffding margin, and a
  Matthews-rescaled cohesion penalty.
- **Tarone--Bonferroni FWER control at a user-selected `alpha`** (default
  `0.05`) on the exact testable count.
- **Polynomial time in `n`** for fixed maximum itemset cardinality, with
  Roaring-bitmap TID lists.

`auto_hyperparams=True` removes the Hoeffding-margin fraction and the
persistence threshold from the user-facing interface, leaving only the
standard statistical confidence level `alpha`.

## Citation

```bibtex
@article{benarab2026same,
  author  = {Massyl Benarab and Hamid Necir and Younes Aoures},
  title   = {{SAME}: Similarity-Aware Monotonic Entropy for FWER-controlled association-rule mining without a minimum-support threshold},
  journal = {Springer KAIS},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT. See `LICENSE`.
