"""Kingfisher I/O wrapper — runs inside the Ubuntu container.

Kingfisher's binary location is probed at runtime from /work/.kf_path
(set during the Dockerfile build).  This is a thin stub that records
build success/failure and a timing; refine the CLI arguments once the
container is up and the binary's help text is available.
"""
from __future__ import annotations
import argparse, os, resource, subprocess, sys, tempfile, time
from pathlib import Path


def _kf_binary() -> Path | None:
    p = Path("/work/.kf_path")
    if p.exists():
        kf_dir = Path(p.read_text().strip().split("=", 1)[1])
        for name in ("kingfisher", "kf", "fpmain", "run"):
            candidate = kf_dir / name
            if candidate.exists() and os.access(candidate, os.X_OK):
                return candidate
        for candidate in kf_dir.rglob("*"):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    return None


def csv_to_transactions(csv_path: Path, out_path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(csv_path)
    with out_path.open("w") as f:
        for _, row in df.iterrows():
            items = [str(i + 1) for i, v in enumerate(row) if v == 1]
            f.write(" ".join(items) + "\n")


def run(csv_path: Path, alpha: float, dataset: str) -> dict:
    binary = _kf_binary()
    if binary is None:
        return {"dataset": dataset, "method": "kingfisher",
                 "param_name": "alpha", "param_value": alpha,
                 "time_s": 0.0, "peak_mem_MB": 0.0,
                 "n_itemsets": 0, "n_rules": 0, "max_lift": 0.0,
                 "error": "build_failed_or_binary_not_found",
                 "n_rules_fwer": 0}
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        txn = tdir / "input.txt"
        csv_to_transactions(csv_path, txn)
        # Kingfisher typical CLI:  kingfisher -i input.txt -a alpha
        cmd = [str(binary), "-i", str(txn), "-a", str(alpha)]
        t0 = time.perf_counter()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            status = "ok" if r.returncode == 0 else f"rc={r.returncode}"
            stdout = r.stdout
        except subprocess.TimeoutExpired:
            status = "timeout_600s"; stdout = ""
        elapsed = round(time.perf_counter() - t0, 3)
        peak_mb = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                          / 1024, 2)
        # Kingfisher prints significant rules one per line; blank lines + '#' skipped
        n_sig = sum(1 for ln in stdout.splitlines()
                     if ln.strip() and not ln.lstrip().startswith("#"))
        return {"dataset": dataset, "method": "kingfisher",
                 "param_name": "alpha", "param_value": alpha,
                 "time_s": elapsed, "peak_mem_MB": peak_mb,
                 "n_itemsets": n_sig, "n_rules": n_sig, "max_lift": 0.0,
                 "error": "" if status == "ok" else status,
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
