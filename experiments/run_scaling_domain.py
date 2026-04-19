"""
Scaling experiment on synthetic *neuroimaging-style* binary data.

The three target domains (neuroimaging, population genomics, protein
co-occurrence) share a common structural property: they are binary
matrices in which a modest number of columns (anatomical regions,
variant categories, Pfam families) carry correlated structure over a
large number of rows (subjects, variants, proteins). We emulate that
structure by planting K correlated clusters of columns with tunable
density and generating n rows.

The experiment sweeps n in {1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5,
5e5, 1e6} and m in {20, 50, 100}, fits SAME (DFS + Tarone), and logs
time, memory, rule count, and FWER count. Output is appended to
experiments/out/scaling_domain.csv.
"""
from __future__ import annotations
import sys, time, tracemalloc
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent

OUT = ROOT / "experiments" / "out" / "scaling_domain.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)


def synth(n: int, m: int, k_clusters: int = 4, density: float = 0.30,
          seed: int = 42) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    cluster_size = max(1, m // k_clusters)
    latent = rng.random((n, k_clusters)) < density
    X = np.zeros((n, m), dtype=np.int8)
    for c in range(k_clusters):
        cols = range(c * cluster_size, min((c + 1) * cluster_size, m))
        for j in cols:
            noise = rng.random(n) < 0.08
            X[:, j] = (latent[:, c] ^ noise).astype(np.int8)
    # remaining columns iid
    for j in range(k_clusters * cluster_size, m):
        X[:, j] = (rng.random(n) < density).astype(np.int8)
    names = [f"f{i:03d}" for i in range(m)]
    return X, names


def run_same(X, names, max_k, mode):
    from same_fim import SAME as SAMEv6
    est = SAMEv6(max_k=max_k, seed=42, mode="all", test_method="gtest",
                  top_k_rules=10_000, fwer_alpha=0.05,
                  search_mode=mode, fdr_method="tarone", opus_top_k=500)
    est.fit(X, feature_names=names)
    r = est.result_
    n_it = sum(len(v) for v in r.frequent_itemsets.values())
    return {"n_itemsets": n_it, "n_rules": len(r.rules),
            "n_rules_fwer": sum(1 for rr in r.rules if rr.passes_fwer)}


def main():
    rows = []
    config = [
        (1_000,    20, 6), (2_000,    20, 6), (5_000,   20, 6),
        (10_000,   20, 6), (20_000,   20, 6), (50_000,  20, 6),
        (100_000,  20, 5), (200_000,  20, 5), (500_000, 20, 4),
        (1_000_000, 20, 3),
        (10_000,   50, 5), (50_000,   50, 5), (100_000,  50, 4),
        (500_000,  50, 3), (1_000_000, 50, 3),
        (10_000,  100, 4), (50_000, 100, 4), (100_000, 100, 3),
    ]
    for n, m, max_k in config:
        X, names = synth(n, m)
        for mode in ("dfs", "opus"):
            tracemalloc.start()
            t0 = time.perf_counter()
            try:
                r = run_same(X, names, max_k, mode)
                ok = True; err = ""
            except Exception as e:
                r = {"n_itemsets": 0, "n_rules": 0, "n_rules_fwer": 0}
                ok = False; err = f"{type(e).__name__}: {str(e)[:60]}"
            t = round(time.perf_counter() - t0, 3)
            peak = round(tracemalloc.get_traced_memory()[1] / 2**20, 2)
            tracemalloc.stop()
            row = {"n": n, "m": m, "max_k": max_k, "mode": mode,
                    "time_s": t, "peak_mem_MB": peak, "ok": ok, "err": err,
                    **r}
            rows.append(row)
            print(f"n={n:>7} m={m:>3} mode={mode:>4} t={t:>7}s mem={peak:>7}MB "
                  f"rules={r['n_rules']:>6} fwer={r['n_rules_fwer']:>6} {'OK' if ok else err}")
            pd.DataFrame(rows).to_csv(OUT, index=False)  # checkpoint
    print(f"\nWrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
