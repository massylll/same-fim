"""SPMF wrapper — runs inside the Java-21 container.

Fournier-Viger's SPMF library (v2.65+) contains 250+ algorithms.  We
benchmark two representative ones:

  * FPClose (closed frequent itemsets, sigma-parameterised) — a
    redundancy-aware FP-Growth variant.
  * TopKRules (top-K association rules, K-parameterised) — the closest
    SPMF algorithm to SAME in that it avoids a user-chosen sigma.

Neither exposes a per-rule FWER filter, so both are reported alongside
mlxtend / PAMI in the unvalidated-rule family.  Output rows are
compatible with domain_benchmark_final.csv.
"""
from __future__ import annotations
import argparse, os, resource, subprocess, sys, tempfile, time
from pathlib import Path

SPMF_JAR = "/work/spmf.jar"


def csv_to_spmf(csv_path: Path, out_path: Path) -> None:
    """SPMF transaction format: one line per tx, space-separated 1-based ids."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    with out_path.open("w") as f:
        for _, row in df.iterrows():
            items = [str(i + 1) for i, v in enumerate(row) if v == 1]
            f.write((" ".join(items) if items else "") + "\n")


def _spmf(algo: str, args: list[str], txn_file: Path, out_file: Path,
           timeout_s: int = 600) -> tuple[float, float, str, int]:
    cmd = ["java", "-jar", SPMF_JAR, "run", algo,
            str(txn_file), str(out_file)] + [str(a) for a in args]
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=timeout_s)
        err = "" if r.returncode == 0 else f"rc={r.returncode}"
        stderr_tail = r.stderr[-200:] if r.stderr else ""
    except subprocess.TimeoutExpired:
        err = f"timeout_{timeout_s}s"; stderr_tail = ""
    elapsed = round(time.perf_counter() - t0, 3)
    peak_mb = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 2)
    n_pat = 0
    if out_file.exists():
        with out_file.open() as f:
            n_pat = sum(1 for ln in f if ln.strip())
    if err and n_pat == 0:
        err = f"{err}: {stderr_tail.strip()[:80]}"
    return elapsed, peak_mb, err, n_pat


def run(csv_path: Path, dataset: str) -> list[dict]:
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        txn = tdir / "input.txt"
        csv_to_spmf(csv_path, txn)
        rows: list[dict] = []

        # FPClose at sigma = 0.10 (closed frequent itemsets)
        out = tdir / "fpclose.txt"
        t, mem, err, n = _spmf("FPClose", ["0.10"], txn, out)
        rows.append({"dataset": dataset, "method": "spmf_fpclose",
                      "param_name": "sigma", "param_value": 0.10,
                      "time_s": t, "peak_mem_MB": mem,
                      "n_itemsets": n, "n_rules": n, "max_lift": 0.0,
                      "error": err, "n_rules_fwer": 0})

        # TopKRules at K=200, min_conf=0.5 (top-K association rules)
        out = tdir / "topk.txt"
        t, mem, err, n = _spmf("TopKRules", [200, 0.5], txn, out)
        rows.append({"dataset": dataset, "method": "spmf_topkrules",
                      "param_name": "K", "param_value": 200,
                      "time_s": t, "peak_mem_MB": mem,
                      "n_itemsets": n, "n_rules": n, "max_lift": 0.0,
                      "error": err, "n_rules_fwer": 0})

        return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    rows = run(args.input, args.dataset)
    import csv
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{args.dataset} / {r['method']}: t={r['time_s']} "
               f"rules={r['n_rules']} err={r['error']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
