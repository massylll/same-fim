"""
Generate synthetic binary datasets at various sizes and run SAME
(OPUS+Tarone mode) on each to measure empirical scaling.

Output: experiments/out/polynomial_scaling.csv

Also fits log(time) = a * log(n) + b for each m value to estimate the
empirical complexity exponent.
"""
from __future__ import annotations

import sys
import time
import tracemalloc
import threading
from pathlib import Path

import numpy as np
import pandas as pd

# ---
# Paths
# ---
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "experiments" / "out"
OUT.mkdir(parents=True, exist_ok=True)


# ---
# Configuration
# ---
sizes_n = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
m_values = [10, 15, 20]
density = 0.3
max_k = 5

TIMEOUT_DEFAULT = 120
TIMEOUT_LARGE = 300  # for n >= 500_000


# ---
# Timeout helper (threading-based, Windows-safe)
# ---
class _TimeoutError(Exception):
    pass


def _run_with_timeout_and_measure(fn, timeout_s, *args, **kwargs):
    """Run *fn* in a daemon thread with tracemalloc; return (time_s, peak_MB, result)."""
    result_box = [None]
    exc_box = [None]
    mem_box = [0.0]

    def _target():
        try:
            tracemalloc.start()
            result_box[0] = fn(*args, **kwargs)
            mem_box[0] = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
            tracemalloc.stop()
        except Exception as e:
            try:
                tracemalloc.stop()
            except Exception:
                pass
            exc_box[0] = e

    t0 = time.perf_counter()
    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    elapsed = time.perf_counter() - t0
    if t.is_alive():
        raise _TimeoutError(f"timeout after {timeout_s}s")
    if exc_box[0] is not None:
        raise exc_box[0]
    return round(elapsed, 4), round(mem_box[0], 2), result_box[0]


# ---
# Synthetic data generator
# ---
def gen_synth(n: int, m: int, density: float = 0.3, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n, m)) < density).astype(np.int8)


# ---
# Single run
# ---
def run_once(n: int, m: int):
    from same_fim import SAME as SAMEv6

    X = gen_synth(n, m, density=density)
    feature_names = [f"x{i}" for i in range(m)]

    est = SAMEv6(
        search_mode="opus",
        max_k=max_k,
        opus_top_k=500,
        fdr_method="tarone",
        top_k_rules=5000,
        seed=42,
    )
    est.fit(X, feature_names=feature_names)
    r = est.result_

    n_itemsets = sum(len(v) for v in r.frequent_itemsets.values())
    n_rules = len(r.rules)

    return {
        "n_itemsets": n_itemsets,
        "n_rules": n_rules,
    }


# ---
# Main
# ---
def main():
    rows = []
    total_t0 = time.perf_counter()

    print(f"Polynomial scaling experiment")
    print(f"  sizes_n = {sizes_n}")
    print(f"  m_values = {m_values}")
    print(f"  density = {density}")
    print(f"  max_k = {max_k}")
    print(f"{'-'*70}")

    for m in m_values:
        print(f"\n--- m = {m} ---")
        for n in sizes_n:
            timeout = TIMEOUT_LARGE if n >= 500_000 else TIMEOUT_DEFAULT
            tag = f"  n={n:>10,}  m={m}"
            print(tag, end=" ... ", flush=True)

            row = {
                "n": n,
                "m": m,
                "time_s": "",
                "peak_mem_MB": "",
                "n_itemsets": "",
                "n_rules": "",
                "error": "",
            }

            try:
                elapsed, mem, res = _run_with_timeout_and_measure(
                    run_once, timeout, n, m
                )
                row["time_s"] = elapsed
                row["peak_mem_MB"] = mem
                row["n_itemsets"] = res["n_itemsets"]
                row["n_rules"] = res["n_rules"]
                print(
                    f"t={elapsed}s  mem={mem}MB  "
                    f"itemsets={res['n_itemsets']}  rules={res['n_rules']}"
                )
            except _TimeoutError:
                row["error"] = "timeout"
                print(f"TIMEOUT ({timeout}s)")
            except Exception as e:
                row["error"] = str(e)[:200]
                print(f"ERROR: {str(e)[:80]}")

            rows.append(row)

    # -------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------
    df = pd.DataFrame(rows)
    out_path = OUT / "polynomial_scaling.csv"
    df.to_csv(out_path, index=False)

    total_elapsed = time.perf_counter() - total_t0
    print(f"\n{'-'*70}")
    print(f"  Wrote {len(rows)} rows to {out_path}")
    print(f"  Total wall time: {total_elapsed:.1f}s")
    print(f"{'-'*70}")

    # -------------------------------------------------------------------
    # Fit log-log regression to estimate empirical scaling exponent
    # -------------------------------------------------------------------
    print("\n\n--- empirical scaling exponent ---")
    print("  Fitting log(time) = a * log(n) + b for each m\n")

    df_ok = df[df["error"] == ""].copy()
    if df_ok.empty:
        df_ok = df[df["error"].isna()].copy()

    if not df_ok.empty:
        df_ok["time_s"] = pd.to_numeric(df_ok["time_s"], errors="coerce")
        df_ok["n"] = pd.to_numeric(df_ok["n"], errors="coerce")

        for m_val in m_values:
            subset = df_ok[df_ok["m"] == m_val].dropna(subset=["time_s", "n"])
            if len(subset) >= 2:
                log_n = np.log(subset["n"].values)
                log_t = np.log(subset["time_s"].values)
                # Linear regression: log_t = a * log_n + b
                A = np.vstack([log_n, np.ones(len(log_n))]).T
                coeffs, residuals, rank, sv = np.linalg.lstsq(A, log_t, rcond=None)
                a, b = coeffs
                r2 = 1 - (np.sum((log_t - A @ coeffs)**2) /
                           np.sum((log_t - log_t.mean())**2))
                print(f"  m={m_val:2d}:  exponent a = {a:.3f}  "
                      f"(intercept b = {b:.3f},  R^2 = {r2:.4f})")
                if abs(a - 1.0) < 0.3:
                    print(f"         --> near-linear scaling (a ~ 1.0)")
                elif a < 1.0:
                    print(f"         --> sub-linear scaling")
                else:
                    print(f"         --> super-linear scaling (a = {a:.2f})")
            else:
                print(f"  m={m_val}: Not enough data points for regression")
    else:
        print("  (no successful runs to analyze)")

    print("\nDone.")


if __name__ == "__main__":
    main()
