"""
Ablation: auto-hyperparams (data-derived alpha, persistence_threshold)
          vs. manual defaults (alpha=0.1, persistence_threshold=0.1)
on the five in-domain datasets, for both DFS and OPUS.

Output: experiments/out/auto_ablation.csv
"""
from __future__ import annotations
import sys, time, tracemalloc
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
from same_fim import SAME as SAMEv6

OUT = ROOT / "experiments" / "out" / "auto_ablation.csv"
DATASETS = [
    ("abide",          10),
    ("eeg_eye_state",   7),
    ("synth_neuro",     5),
    ("clinvar_sample",  6),
    ("pfam_proteins",   4),
]

rows = []
for ds, max_k in DATASETS:
    df = pd.read_csv(ROOT / "datasets" / f"{ds}.csv").astype(np.int8)
    X = df.values; fn = list(df.columns)
    for mode in ("dfs", "opus"):
        for auto in (False, True):
            tracemalloc.start()
            t0 = time.perf_counter()
            est = SAMEv6(max_k=max_k, seed=42, mode="all", test_method="gtest",
                          top_k_rules=10_000, fwer_alpha=0.05,
                          search_mode=mode, fdr_method="tarone",
                          opus_top_k=1000, auto_hyperparams=auto)
            est.fit(X, feature_names=fn)
            t = round(time.perf_counter() - t0, 3)
            peak = round(tracemalloc.get_traced_memory()[1] / 2**20, 2)
            tracemalloc.stop()
            r = est.result_
            n_it = sum(len(v) for v in r.frequent_itemsets.values())
            lifts = [rr.lift for rr in r.rules] or [0.0]
            row = {"dataset": ds, "mode": mode, "auto": auto,
                   "alpha_used": round(est.alpha, 4),
                   "persistence_used": round(est.persistence_threshold, 4),
                   "time_s": t, "peak_mem_MB": peak,
                   "n_itemsets": n_it, "n_rules": len(r.rules),
                   "n_rules_fwer": sum(1 for rr in r.rules if rr.passes_fwer),
                   "max_lift": round(max(lifts), 4)}
            rows.append(row)
            print(f"{ds:15s} {mode:4s} auto={auto}  alpha={row['alpha_used']:.4f} "
                  f"pi={row['persistence_used']:.4f}  "
                  f"rules={row['n_rules']:>6} fwer={row['n_rules_fwer']:>6} "
                  f"lift={row['max_lift']:.2f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)

print(f"\nWrote {OUT}")
