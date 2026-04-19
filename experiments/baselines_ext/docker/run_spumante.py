"""SPuManTE I/O wrapper — runs inside the Ubuntu container.

Pipeline:
  1. split input CSV's last column as the class label; the remaining
     columns become the transaction matrix.
  2. run `correct/fim_closed`  to compute the SPuManTE-corrected
     significance threshold + minimum-testable-support under the
     unconditional LAMP-style testing framework (Pellegrina et al.).
  3. run `enumerate/fim_closed` with that threshold to emit the
     significant closed itemsets.

Both binaries are supervised (require class labels); this is the
intended use case of SPuManTE.  We emit a row compatible with
domain_benchmark_final.csv.

Usage (inside docker):
  python3 run_spumante.py --input /data/abide.csv --dataset abide \
      --alpha 0.05 --out /out/spumante_abide.csv
"""
from __future__ import annotations
import argparse, os, re, resource, subprocess, sys, tempfile, time
from pathlib import Path

CORRECT  = "/work/SPuManTE/unconditional/correct/fim_closed"
ENUMERATE = "/work/SPuManTE/unconditional/enumerate/fim_closed"
EPSILON = "0.1"   # SPuManTE's internal eps parameter; 0.1 is the paper default


def split_to_spumante(csv_path: Path, tdir: Path) -> tuple[Path, Path]:
    """Write SPuManTE's two-file format:
      class_labels.txt    one 0/1 label per line (the CSV's last column)
      transactions.txt    one line per tx, space-separated 1-based item ids
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    label_col = df.columns[-1]
    y = df[label_col].astype(int).tolist()
    X = df.drop(columns=[label_col])

    lbl_f = tdir / "class_labels.txt"
    txn_f = tdir / "transactions.txt"
    lbl_f.write_text("\n".join(str(v) for v in y) + "\n")
    with txn_f.open("w") as f:
        for _, row in X.iterrows():
            items = [str(i + 1) for i, v in enumerate(row) if v == 1]
            f.write(" ".join(items) + "\n")
    return lbl_f, txn_f


def run(csv_path: Path, alpha: float, dataset: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        lbl, txn = split_to_spumante(csv_path, tdir)
        base = str(tdir / "out")

        # step 1: corrected threshold + testable support
        cmd1 = [CORRECT, base, str(alpha), str(lbl), str(txn), EPSILON]
        t0 = time.perf_counter()
        try:
            r1 = subprocess.run(cmd1, capture_output=True, text=True,
                                 timeout=600)
        except subprocess.TimeoutExpired:
            return _fail(dataset, alpha, "timeout_correct_600s", t0)
        if r1.returncode != 0:
            return _fail(dataset, alpha, f"correct_rc={r1.returncode}", t0,
                         err=r1.stderr[:300])

        # Parse corrected threshold + LCM support from out_results.txt
        # (SPuManTE writes a structured summary there).
        results_f = Path(base + "_results.txt")
        text = results_f.read_text() if results_f.exists() else r1.stdout
        thr_m = re.search(r"orrected significance threshold:\s*([0-9eE.+-]+)",
                           text)
        ts_m  = re.search(r"Final LCM support:\s*([0-9]+)", text)
        if not (thr_m and ts_m):
            return _fail(dataset, alpha, "could_not_parse_threshold", t0,
                         err=text[:500])
        corrected_thr = thr_m.group(1)
        min_testable  = ts_m.group(1)

        # step 2: enumerate significant closed itemsets
        cmd2 = [ENUMERATE, base + "_enum", corrected_thr, min_testable,
                 str(lbl), str(txn), EPSILON]
        try:
            r2 = subprocess.run(cmd2, capture_output=True, text=True,
                                 timeout=600)
        except subprocess.TimeoutExpired:
            return _fail(dataset, alpha, "timeout_enumerate_600s", t0)
        elapsed = round(time.perf_counter() - t0, 3)
        peak_mb = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                          / 1024, 2)
        # Count significant itemsets. The enumerate binary writes them to
        # <base>_enum_sig_itemsets.txt (one itemset per line).
        n_sig = 0
        enum_out = Path(base + "_enum_sig_itemsets.txt")
        if enum_out.exists():
            with enum_out.open() as f:
                n_sig = sum(1 for ln in f if ln.strip())
        if n_sig == 0:
            # Fallback: parse "Number of significant patterns found: N" from log
            stdout_m = re.search(r"[Nn]umber of significant patterns[^:]*:\s*([0-9]+)",
                                   r2.stdout)
            if stdout_m:
                n_sig = int(stdout_m.group(1))

        status = "ok" if r2.returncode == 0 else f"enumerate_rc={r2.returncode}"
        return {"dataset": dataset, "method": "spumante",
                 "param_name": "alpha", "param_value": alpha,
                 "time_s": elapsed, "peak_mem_MB": peak_mb,
                 "n_itemsets": n_sig, "n_rules": n_sig, "max_lift": 0.0,
                 "error": "" if status == "ok" else status,
                 "n_rules_fwer": n_sig}


def _fail(ds, alpha, msg, t0, err=""):
    elapsed = round(time.perf_counter() - t0, 3)
    peak_mb = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                      / 1024, 2)
    return {"dataset": ds, "method": "spumante",
             "param_name": "alpha", "param_value": alpha,
             "time_s": elapsed, "peak_mem_MB": peak_mb,
             "n_itemsets": 0, "n_rules": 0, "max_lift": 0.0,
             "error": f"{msg}: {err}" if err else msg,
             "n_rules_fwer": 0}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--dataset", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    row = run(args.input, args.alpha, args.dataset)
    import csv
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader(); w.writerow(row)
    print(f"{args.dataset}: {row}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
