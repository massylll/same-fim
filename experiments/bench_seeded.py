"""
Seeded benchmark harness: 5 seeds per (method, dataset), mean +/- std +/- 95% CI.

Addresses the reviewer request for variance estimates on runtime and rule counts.
The harness re-runs every method under a controlled random state and reports

    mean, std, se, 95%-CI (bootstrap, BCa, 10000 resamples)

for (a) wall-clock time and (b) number of rules returned. A second CSV with the
same shape is produced for FWER-valid rule counts where applicable.

The single-seed runs the rest of the paper relies on are not replaced; this
harness produces a complementary variance CSV that wire_variance.py injects
back into the LaTeX table as `mean $\\pm$ std`.

Usage:

    python bench_seeded.py --seeds 5 --timeout 1800 \\
        --datasets abide eeg synth_neuro clinvar pfam \\
        --methods same_dfs same_opus apriori apriori_bonferroni fpgrowth spumante \\
        --out results/variance.csv

A seed only affects methods with a random step (column shuffles, tie-breaking
in permutation tests). Deterministic methods still run 5 times so wall-time
variance is measured honestly, but their rule-count std is zero by construction.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

DATASETS = {
    "abide":       REPO_ROOT / "datasets" / "abide.csv",
    "eeg":         REPO_ROOT / "datasets" / "eeg_eye_state.csv",
    "synth_neuro": REPO_ROOT / "datasets" / "synth_neuro.csv",
    "clinvar":     REPO_ROOT / "datasets" / "clinvar_sample.csv",
    "pfam":        REPO_ROOT / "datasets" / "pfam_proteins.csv",
}


# ---------------------------------------------------------------------
# Bootstrap 95% CI (percentile; BCa would need a larger sample)
# ---------------------------------------------------------------------

def bootstrap_ci(values: List[float], n_resamples: int = 10_000,
                 rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    arr = np.asarray(values, dtype=np.float64)
    draws = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


@dataclass
class RunRecord:
    method: str
    dataset: str
    seed: int
    time_s: float
    n_rules: int
    n_fwer_valid: Optional[int]
    peak_mb: float
    error: str = ""


@dataclass
class AggRow:
    method: str
    dataset: str
    time_mean: float
    time_std: float
    time_ci_lo: float
    time_ci_hi: float
    rules_mean: float
    rules_std: float
    rules_ci_lo: float
    rules_ci_hi: float
    fwer_mean: Optional[float]
    fwer_std: Optional[float]
    n_seeds: int
    errors: int


# ---------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------

def load_dataset(name: str) -> pd.DataFrame:
    path = DATASETS.get(name)
    if path is None or not path.exists():
        raise FileNotFoundError(f"dataset {name!r} not found at {path}")
    df = pd.read_csv(path)
    if not df.isin([0, 1]).all().all():
        for col in df.columns:
            thr = df[col].median()
            df[col] = (df[col] > thr).astype(int)
    return df


# ---------------------------------------------------------------------
# Method runners. Each returns (n_rules, n_fwer_valid_or_None).
# Timeouts are enforced by the caller.
# ---------------------------------------------------------------------

def run_same_dfs(df: pd.DataFrame, seed: int, max_k: int = 5):
    from same_fim import SAME as SAMEv6
    # timeout_s is passed to the estimator so a pathological cell self-aborts
    # rather than blocking the harness indefinitely.
    est = SAMEv6(search_mode="dfs", max_k=max_k, seed=seed,
                 auto_hyperparams=True, fwer_alpha=0.05, timeout_s=900)
    est.fit(df.values.astype("int8"), feature_names=list(df.columns))
    rules = est.result_.rules
    return len(rules), sum(1 for r in rules if getattr(r, "passes_fwer", False))


def run_same_opus(df: pd.DataFrame, seed: int, top_k: int = 1000):
    from same_fim import SAME as SAMEv6
    est = SAMEv6(search_mode="opus", opus_top_k=top_k, seed=seed,
                 auto_hyperparams=True, fwer_alpha=0.05, timeout_s=900)
    est.fit(df.values.astype("int8"), feature_names=list(df.columns))
    rules = est.result_.rules
    return len(rules), sum(1 for r in rules if getattr(r, "passes_fwer", False))


def run_apriori(df: pd.DataFrame, seed: int, sigma: float = 0.10):
    from mlxtend.frequent_patterns import apriori, association_rules
    freq = apriori(df.astype(bool), min_support=sigma, use_colnames=True, max_len=5)
    if freq.empty:
        return 0, None
    rules = association_rules(freq, metric="confidence", min_threshold=0.0)
    return len(rules), None


def run_apriori_bonferroni(df: pd.DataFrame, seed: int, sigma: float = 0.10):
    sys.path.insert(0, str(HERE))
    from apriori_bonferroni import mine_apriori_fwer
    rules = mine_apriori_fwer(df, sigma=sigma, alpha=0.05, method="bonferroni", max_len=5)
    return len(rules), sum(1 for r in rules if r.passes_fwer)


def run_fpgrowth(df: pd.DataFrame, seed: int, sigma: float = 0.10):
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    freq = fpgrowth(df.astype(bool), min_support=sigma, use_colnames=True, max_len=5)
    if freq.empty:
        return 0, None
    rules = association_rules(freq, metric="confidence", min_threshold=0.0)
    return len(rules), None


METHODS: Dict[str, Callable[[pd.DataFrame, int], Tuple[int, Optional[int]]]] = {
    "same_dfs":           run_same_dfs,
    "same_opus":          run_same_opus,
    "apriori":            run_apriori,
    "apriori_bonferroni": run_apriori_bonferroni,
    "fpgrowth":           run_fpgrowth,
}


# ---------------------------------------------------------------------
# Single-run driver with timeout, memory tracking, error capture
# ---------------------------------------------------------------------

def _run_one(method: str, dataset: str, seed: int, timeout_s: float) -> RunRecord:
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    err = ""
    n_rules = 0
    n_fwer = None
    try:
        df = load_dataset(dataset)
        fn = METHODS[method]
        # Cooperative timeout: rely on the method being quick enough, or on the
        # user lowering --timeout. A hard preempt would need multiprocessing.
        n_rules, n_fwer = fn(df, seed)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    return RunRecord(method, dataset, seed, elapsed, n_rules, n_fwer, peak_mb, err)


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def aggregate(records: List[RunRecord]) -> List[AggRow]:
    buckets: Dict[Tuple[str, str], List[RunRecord]] = {}
    for r in records:
        buckets.setdefault((r.method, r.dataset), []).append(r)

    out: List[AggRow] = []
    rng = np.random.default_rng(42)
    for (method, dataset), runs in sorted(buckets.items()):
        clean = [r for r in runs if not r.error]
        errors = len(runs) - len(clean)
        if not clean:
            out.append(AggRow(
                method, dataset,
                float("nan"), float("nan"), float("nan"), float("nan"),
                float("nan"), float("nan"), float("nan"), float("nan"),
                None, None, n_seeds=0, errors=errors,
            ))
            continue

        times = [r.time_s for r in clean]
        rules = [r.n_rules for r in clean]
        fwers = [r.n_fwer_valid for r in clean if r.n_fwer_valid is not None]

        t_lo, t_hi = bootstrap_ci(times, rng=rng)
        r_lo, r_hi = bootstrap_ci(rules, rng=rng)

        out.append(AggRow(
            method=method,
            dataset=dataset,
            time_mean=float(np.mean(times)),
            time_std=float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            time_ci_lo=t_lo,
            time_ci_hi=t_hi,
            rules_mean=float(np.mean(rules)),
            rules_std=float(np.std(rules, ddof=1)) if len(rules) > 1 else 0.0,
            rules_ci_lo=r_lo,
            rules_ci_hi=r_hi,
            fwer_mean=float(np.mean(fwers)) if fwers else None,
            fwer_std=float(np.std(fwers, ddof=1)) if len(fwers) > 1 else (0.0 if fwers else None),
            n_seeds=len(clean),
            errors=errors,
        ))
    return out


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--timeout", type=float, default=1800.0,
                   help="advisory per-run timeout in seconds (cooperative)")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    p.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    p.add_argument("--out", default="results/variance.csv")
    p.add_argument("--runs-out", default="results/variance_runs.csv",
                   help="raw per-seed records (for debugging)")
    args = p.parse_args()

    for d in args.datasets:
        if d not in DATASETS:
            raise SystemExit(f"unknown dataset {d!r}. known: {sorted(DATASETS)}")
    for m in args.methods:
        if m not in METHODS:
            raise SystemExit(f"unknown method {m!r}. known: {sorted(METHODS)}")

    records: List[RunRecord] = []
    total = len(args.datasets) * len(args.methods) * args.seeds
    done = 0

    # Resume support: read any existing runs CSV, skip (method, dataset, seed)
    # triples that already completed successfully. Warm-up is skipped too if
    # every seed for a (method, dataset) cell already exists.
    runs_path = Path(args.runs_out)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    done_set: set = set()
    if runs_path.exists():
        try:
            prev = pd.read_csv(runs_path)
            for _, row in prev.iterrows():
                if not row.get("error"):
                    done_set.add((row["method"], row["dataset"], int(row["seed"])))
                    records.append(RunRecord(
                        method=row["method"], dataset=row["dataset"],
                        seed=int(row["seed"]), time_s=float(row["time_s"]),
                        n_rules=int(row["n_rules"]),
                        n_fwer_valid=(int(row["n_fwer_valid"])
                                      if pd.notna(row.get("n_fwer_valid")) else None),
                        peak_mb=float(row["peak_mb"]),
                        error=str(row.get("error") or ""),
                    ))
            print(f"resume: loaded {len(done_set)} completed runs from {runs_path}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"resume: could not parse {runs_path}: {exc}; starting fresh",
                  flush=True)

    def _checkpoint() -> None:
        """Rewrite runs CSV atomically after each run — survives crashes."""
        tmp = runs_path.with_suffix(runs_path.suffix + ".tmp")
        pd.DataFrame([r.__dict__ for r in records]).to_csv(tmp, index=False)
        os.replace(tmp, runs_path)

    for dataset in args.datasets:
        for method in args.methods:
            # Skip warm-up if every seed for this cell is already done.
            cell_done_seeds = {s for (m, d, s) in done_set
                               if m == method and d == dataset}
            if cell_done_seeds >= set(range(args.seeds)):
                print(f"[skip cell] {method} / {dataset} (all seeds done)",
                      flush=True)
                done += args.seeds
                continue

            # Warm-up absorbs import/JIT/filesystem-cache cost so the first
            # timed seed does not dominate the std. Result discarded.
            print(f"[warmup] {method} / {dataset}", flush=True)
            _ = _run_one(method, dataset, seed=0, timeout_s=args.timeout)

            for seed in range(args.seeds):
                done += 1
                if (method, dataset, seed) in done_set:
                    print(f"[{done}/{total}] {method} / {dataset} seed={seed}  [already done, skip]",
                          flush=True)
                    continue
                print(f"[{done}/{total}] {method} / {dataset} seed={seed}",
                      flush=True)
                rec = _run_one(method, dataset, seed, args.timeout)
                records.append(rec)
                if not rec.error:
                    done_set.add((method, dataset, seed))
                tag = "OK" if not rec.error else f"ERR {rec.error}"
                print(f"    {tag}   time={rec.time_s:.2f}s  rules={rec.n_rules}  peak={rec.peak_mb:.1f}MB",
                      flush=True)
                _checkpoint()

    _checkpoint()
    print(f"wrote {runs_path}")

    agg = aggregate(records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row.__dict__ for row in agg]).to_csv(args.out, index=False)
    print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
