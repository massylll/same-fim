"""Merge per-dataset per-baseline CSVs into a single table appendable
to all_methods_combined.csv.  Usage: merge_results.py <out_dir>"""
import sys
from pathlib import Path
import pandas as pd

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
frames = []
for csv in sorted(out_dir.glob("*.csv")):
    if csv.name == "all_baselines_ext.csv":
        continue
    try:
        frames.append(pd.read_csv(csv))
    except Exception as e:
        print(f"skip {csv.name}: {e}", file=sys.stderr)

if frames:
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(out_dir / "all_baselines_ext.csv", index=False)
    print(merged)
else:
    print("no per-baseline CSVs found; did the builds run?", file=sys.stderr)
