"""WYlight I/O wrapper — runs inside the Ubuntu container.

WYlight (Llinares-Lopez, Sugiyama, Papaxanthos, Borgwardt, KDD 2015) is a
supervised Westfall-Young permutation significance miner. Two-stage:

  1. lcm_wy_fisher/fim_closed <out_base> <n_perm> <target_fwer>
       <class_labels_file> <transactions_file> <rand_seed>
     writes <out_base>_results.txt with the corrected significance
     threshold and the final LCM support.

  2. lcm_comp_pvalues_fisher/fim_closed <out_base_sig> <sig_th>
       <lcm_th> <class_labels_file> <transactions_file>
     emits the significant itemsets.

Emits a row compatible with domain_benchmark_final.csv.
"""
from __future__ import annotations
import argparse, re, resource, subprocess, sys, tempfile, time
from pathlib import Path

WY_BIN   = "/work/WYlight/lcm_wy_fisher/fim_closed"
ENUM_BIN = "/work/WYlight/lcm_comp_pvalues_fisher/fim_closed"
N_PERM    = "100"   # Westfall-Young permutation budget
RAND_SEED = "42"


def split_to_wylight(csv_path: Path, tdir: Path) -> tuple[Path, Path]:
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
            f.write((" ".join(items) if items else "") + "\n")
    return lbl_f, txn_f


def run(csv_path: Path, alpha: float, dataset: str) -> dict:
    for b in (WY_BIN, ENUM_BIN):
        if not Path(b).exists():
            return _fail(dataset, alpha, f"binary_missing:{b}", time.perf_counter())
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        lbl, txn = split_to_wylight(csv_path, tdir)
        base = str(tdir / "wy")
        t0 = time.perf_counter()

        # Stage 1: permutation-based corrected threshold
        cmd1 = [WY_BIN, base, N_PERM, str(alpha),
                 str(lbl), str(txn), RAND_SEED]
        try:
            r1 = subprocess.run(cmd1, capture_output=True, text=True,
                                 timeout=1800)
        except subprocess.TimeoutExpired:
            return _fail(dataset, alpha, "timeout_wy_stage1_1800s", t0)
        if r1.returncode != 0:
            return _fail(dataset, alpha, f"wy_stage1_rc={r1.returncode}", t0,
                         err=(r1.stderr or r1.stdout)[-300:])

        rp = Path(base + "_results.txt")
        if not rp.exists():
            return _fail(dataset, alpha, "no_wy_results_file", t0,
                         err=r1.stdout[:300])
        text = rp.read_text()
        thr_m = re.search(r"[Cc]orrected.*threshold[^:]*:\s*([0-9eE.+-]+)", text)
        ts_m  = re.search(r"[Ff]inal\s+LCM\s+support[^:]*:\s*([0-9]+)", text)
        if not (thr_m and ts_m):
            return _fail(dataset, alpha, "cannot_parse_wy_results", t0,
                         err=text[:300])

        # Stage 2: enumerate significant itemsets
        base_sig = base + "_sig"
        cmd2 = [ENUM_BIN, base_sig, thr_m.group(1), ts_m.group(1),
                 str(lbl), str(txn)]
        try:
            r2 = subprocess.run(cmd2, capture_output=True, text=True,
                                 timeout=600)
        except subprocess.TimeoutExpired:
            return _fail(dataset, alpha, "timeout_wy_stage2_600s", t0)
        elapsed = round(time.perf_counter() - t0, 3)
        peak_mb = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                          / 1024, 2)

        n_sig = 0
        for name in ("sig_itemsets.txt", "significant_itemsets.txt",
                      "enum_sig_itemsets.txt"):
            sig_f = Path(base_sig + "_" + name)
            if sig_f.exists():
                with sig_f.open() as f:
                    n_sig = sum(1 for ln in f if ln.strip())
                break
        if n_sig == 0:
            m = re.search(r"[Nn]umber of significant patterns[^:]*:\s*([0-9]+)",
                           r2.stdout)
            if m:
                n_sig = int(m.group(1))

        status = "ok" if r2.returncode == 0 else f"enum_rc={r2.returncode}"
        return {"dataset": dataset, "method": "wylight",
                 "param_name": "alpha", "param_value": alpha,
                 "time_s": elapsed, "peak_mem_MB": peak_mb,
                 "n_itemsets": n_sig, "n_rules": n_sig, "max_lift": 0.0,
                 "error": "" if status == "ok" else status,
                 "n_rules_fwer": n_sig}


def _fail(ds, alpha, msg, t0, err=""):
    elapsed = round(time.perf_counter() - t0, 3)
    peak_mb = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                      / 1024, 2)
    return {"dataset": ds, "method": "wylight",
             "param_name": "alpha", "param_value": alpha,
             "time_s": elapsed, "peak_mem_MB": peak_mb,
             "n_itemsets": 0, "n_rules": 0, "max_lift": 0.0,
             "error": f"{msg}: {err}" if err else msg, "n_rules_fwer": 0}


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
