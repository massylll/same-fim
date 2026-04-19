# Significance-aware baseline containers

Dockerfiles for every significance-aware baseline used in Table
`tab:baselines_ext` of the paper, plus the runner scripts that invoke them
on each dataset and emit a CSV row. Every image builds from a pinned base
(Debian slim + Python 3.12 or OpenJDK 21) so results are reproducible
independent of the host OS.

## Images

| Image                        | Baseline                  | Source language | Port of paper table column                           |
|------------------------------|---------------------------|-----------------|------------------------------------------------------|
| `Dockerfile.lamp`            | LAMP (Terada 2013)        | Python 2 + C    | FWER-controlled, supervised                          |
| `Dockerfile.spumante`        | SPuManTE (Pellegrina 2019)| Python 3 + C    | FWER-controlled, supervised                          |
| `Dockerfile.wylight`         | WYlight (Llinares 2015)   | C++             | FWER-controlled, supervised (Westfall-Young)         |
| `Dockerfile.opusminer`       | OPUS Miner (Webb 2014)    | R / C++         | Significance-corrected, unsupervised                 |
| `Dockerfile.kingfisher`      | Kingfisher (Hamalainen)   | C++             | Chi-square pruning, best-first (not run in paper)    |
| `Dockerfile.spmf`            | SPMF (FPClose, TopKRules) | Java 21         | Unsupervised, no significance filter                 |
| `Dockerfile.apriori_bonf`    | Apriori + Bonferroni      | Python 3.12     | NEW — FWER-controlled, unsupervised (this revision)  |

## Build

```bash
# from the repository root
cd SAME_v4/experiments/baselines_ext/docker

# build one image
docker build -f Dockerfile.lamp -t same-bench/lamp:latest .

# build all of them
for f in Dockerfile.*; do
    name=${f#Dockerfile.}
    docker build -f "$f" -t "same-bench/$name:latest" .
done
```

## Run the full battery

```bash
# Windows: run from WSL or Git Bash
bash run_baselines_all.sh
```

This executes every image against every dataset with a 30-minute wall-clock
timeout per cell and merges the per-image CSVs into
`../../results/baselines_ext.csv`, which is what `wire_csvs.py` consumes
when it rebuilds Table `tab:baselines_ext`.

## Status (as of 2026-04-18)

| Baseline          | Builds on Linux | Builds on Windows/WSL | Runs end-to-end         | Notes                                               |
|-------------------|-----------------|-----------------------|-------------------------|-----------------------------------------------------|
| LAMP              | yes             | yes (WSL)             | yes on 3 of 5           | times out on EEG and synth_neuro at max_comb=5      |
| SPuManTE          | yes             | yes (WSL)             | yes on 5 of 5           | 0 rules on synth_neuro is the expected result       |
| WYlight           | yes             | yes (WSL)             | yes on 5 of 5           | 100 permutations                                    |
| OPUS Miner        | yes             | yes (WSL)             | yes on 5 of 5           | top-K = 200                                         |
| SPMF              | yes             | yes                   | yes on 5 of 5           | Java-based, OS-independent                          |
| Kingfisher        | yes             | untested              | not run for submission  | C++ build via upstream Makefile (in image)          |
| Apriori+Bonf.     | yes             | yes                   | yes on 5 of 5           | Python-only; no container strictly needed           |

## Reproducing a single cell by hand

```bash
# LAMP on ABIDE at max_comb=5, FWER alpha=0.05
docker run --rm -v "$PWD/../../..":/workspace same-bench/lamp:latest \
    python /workspace/SAME_v4/experiments/baselines_ext/docker/run_lamp.py \
        --csv /workspace/SAME_v4/datasets/abide.csv \
        --alpha 0.05 --max-comb 5 --timeout 1800

# Apriori + Bonferroni on the same dataset (no container needed)
python ../../apriori_bonferroni.py \
    --csv ../../../datasets/abide.csv \
    --sigma 0.10 --alpha 0.05 --method bonferroni
```

## Gaps / TODO

- `run_opusminer.py` currently forwards to `run_opusminer.R`; the Python
  wrapper that parses the R CSV is in place, but the timing figures include
  the R interpreter startup (~0.8s on the test machine). Subtract that if
  you need a fair wall-time comparison.
- Kingfisher has a Dockerfile but no automated runner. If you need it,
  adapt `run_lamp.py` — the interface is similar.
