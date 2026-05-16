"""One-command reproduction of every number and figure in the paper.

Usage:
    python experiments/reproduce.py

Runs the following in order (each step checkpoints its own CSV under
`experiments/out/`):

  1. domain benchmark            run_remaining_bench.py + run_domain_benchmark.py
  2. neuro-connectivity scaling  run_scaling_domain.py
  3. downstream ABIDE            run_downstream_classification.py
  4. downstream EEG              run_downstream_eeg.py
  5. auto-hyperparam ablation    run_auto_ablation.py

Then wires the CSVs into the sibling paper folder (set via the
SAME_PAPER_DIR env var; defaults to ../paper) and regenerates figures
(needs matplotlib). If no paper folder is found the wiring step is skipped.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Resolve the sibling paper folder for figure/CSV wiring. Override with the
# SAME_PAPER_DIR env var; fall back to common names found in the wild.
import os as _os
_candidates = [
    _os.environ.get("SAME_PAPER_DIR"),
    ROOT.parent / "paper",
    ROOT.parent / "BDMA_submission",
    ROOT.parent / "EINF-PAPER",
]
PAPER = next((Path(c) for c in _candidates if c and Path(c).exists()),
             ROOT.parent / "paper")

STEPS = [
    ("domain benchmark",          "experiments/run_domain_benchmark.py"),
    ("neuro-connectivity scaling", "experiments/run_scaling_domain.py"),
    ("downstream ABIDE",           "experiments/run_downstream_classification.py"),
    ("downstream EEG",             "experiments/run_downstream_eeg.py"),
    ("auto-hyperparam ablation",   "experiments/run_auto_ablation.py"),
]


def run(label: str, script: str) -> None:
    print(f"\n--- {label} ({script}) ---", flush=True)
    path = ROOT / script
    if not path.exists():
        print(f"  SKIP: {path} missing", flush=True)
        return
    r = subprocess.run([sys.executable, "-u", str(path)], cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"step {label!r} failed (exit {r.returncode})")


def wire_and_plot() -> None:
    if not PAPER.exists():
        print(f"\nskip paper wiring: {PAPER} missing")
        return
    print(f"\n--- wiring CSVs into {PAPER} ---", flush=True)
    subprocess.run([sys.executable, "assemble_from_bench.py"], cwd=str(PAPER),
                    check=True)
    subprocess.run([sys.executable, "make_figures_v6.py"], cwd=str(PAPER),
                    check=True)


def main() -> int:
    for label, script in STEPS:
        run(label, script)
    wire_and_plot()
    print("\nReproduction complete. CSVs in experiments/out/, figures in "
          f"{PAPER}/fig_v2/.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
