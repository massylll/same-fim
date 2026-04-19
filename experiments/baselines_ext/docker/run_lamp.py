"""LAMP I/O wrapper — runs inside the Ubuntu container.

LAMP (Terada et al., PNAS 2013) expects two CSVs:
  item_file   header: #gene,feat1,feat2,...    rows: id,0/1/0/...
  value_file  header: #gene,expression         rows: id,0_or_1

We split the last column of the input binary CSV as the "expression"
(label), the rest as items.  Emits a row compatible with
domain_benchmark_final.csv.

Usage:
  python3 run_lamp.py --input /data/abide.csv --dataset abide \
      --alpha 0.05 --out /out/lamp_abide.csv
"""
from __future__ import annotations
import argparse, re, resource, subprocess, sys, tempfile, time
from pathlib import Path

LAMP_DIR = Path("/work/lamp")


def split_item_value(csv_path: Path, tdir: Path) -> tuple[Path, Path]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    target = df.columns[-1]
    item_df = df.drop(columns=[target])
    val_df  = df[[target]]

    ids = [f"t{i}" for i in range(len(df))]

    item_file = tdir / "items.csv"
    val_file  = tdir / "values.csv"
    item_df.insert(0, "#gene", ids)
    # Ensure header row has a # prefix
    item_df.rename(columns={"#gene": "#gene"}, inplace=True)
    item_df.to_csv(item_file, index=False)

    val_df.insert(0, "#gene", ids)
    val_df.to_csv(val_file, index=False)
    return item_file, val_file


def run(csv_path: Path, alpha: float, dataset: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        item_file, val_file = split_item_value(csv_path, tdir)

        cmd = ["python3", str(LAMP_DIR / "lamp.py"),
                "-p", "fisher",
                "--max_comb", "5",   # match SAME's k_max; without this LAMP
                                      #  explores every lattice level and
                                      #  times out on m>=15 datasets.
                str(item_file), str(val_file), str(alpha)]
        t0 = time.perf_counter()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=1800, cwd=str(LAMP_DIR))
            status = "ok" if r.returncode == 0 else f"rc={r.returncode}"
            stdout = r.stdout
            stderr = r.stderr
        except subprocess.TimeoutExpired:
            status = "timeout_1800s"; stdout = ""; stderr = ""
        elapsed = round(time.perf_counter() - t0, 3)
        peak_mb = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                          / 1024, 2)

        # LAMP prints "# of significant combinations: N"
        n_sig_m = re.search(
            r"#\s*of significant combinations:\s*([0-9]+)", stdout)
        n_sig = int(n_sig_m.group(1)) if n_sig_m else 0

        err_msg = ""
        if status != "ok":
            err_msg = f"{status}: {stderr[:200]}"

        return {"dataset": dataset, "method": "lamp_fisher",
                 "param_name": "alpha", "param_value": alpha,
                 "time_s": elapsed, "peak_mem_MB": peak_mb,
                 "n_itemsets": n_sig, "n_rules": n_sig, "max_lift": 0.0,
                 "error": err_msg,
                 "n_rules_fwer": n_sig}


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
