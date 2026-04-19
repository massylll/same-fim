"""OPUS Miner wrapper — runs inside rocker/r-ver:4.4.1 container.

OPUS Miner (Webb & Vreeken, IEEE TKDE 2014) implements branch-and-bound
top-K productive, self-sufficient itemset mining with internal
significance-based pruning.  The R package `opusminer` is the
reference implementation.

The wrapper converts the input CSV to OPUS Miner's transactional
format, invokes `run_opusminer.R`, and writes a row compatible with
domain_benchmark_final.csv.  We report raw OPUS Miner rule counts; the
method tests each candidate for productivity + self-sufficiency at
alpha=0.05 with multiple-testing correction, so the output is an
\"approximately FWER-controlled\" inventory.
"""
from __future__ import annotations
import argparse, resource, subprocess, sys, tempfile, time
from pathlib import Path

R_SCRIPT = "/work/run_opusminer.R"
K = 200


def csv_to_opus(csv_path: Path, out_path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(csv_path)
    with out_path.open("w") as f:
        for _, row in df.iterrows():
            items = [str(i + 1) for i, v in enumerate(row) if v == 1]
            f.write((" ".join(items) if items else "") + "\n")


def run(csv_path: Path, dataset: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        txn = tdir / "input.txt"
        out = tdir / "output.txt"
        csv_to_opus(csv_path, txn)
        cmd = ["Rscript", R_SCRIPT, str(txn), str(out), str(K)]
        t0 = time.perf_counter()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=1800)
            status = "ok" if r.returncode == 0 else f"rc={r.returncode}"
            stderr_tail = r.stderr[-200:] if r.stderr else ""
        except subprocess.TimeoutExpired:
            status = "timeout_1800s"; stderr_tail = ""
        elapsed = round(time.perf_counter() - t0, 3)
        peak_mb = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                          / 1024, 2)
        n_pat = 0
        if out.exists():
            with out.open() as f:
                n_pat = sum(1 for ln in f if ln.strip())
        err = "" if status == "ok" else f"{status}: {stderr_tail.strip()[:100]}"
        return {"dataset": dataset, "method": "opusminer",
                 "param_name": "K", "param_value": K,
                 "time_s": elapsed, "peak_mem_MB": peak_mb,
                 "n_itemsets": n_pat, "n_rules": n_pat, "max_lift": 0.0,
                 "error": err, "n_rules_fwer": n_pat}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    row = run(args.input, args.dataset)
    import csv
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader(); w.writerow(row)
    print(f"{args.dataset}: {row}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
