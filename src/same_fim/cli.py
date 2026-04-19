"""Command-line interface for same-fim.

Example:
    $ same-mine --input data.csv --mode dfs --alpha 0.05 --auto --out rules.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def _load(path: Path):
    import numpy as np
    import pandas as pd
    df = pd.read_csv(path).astype(np.int8)
    return df.values, list(df.columns)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="same-mine",
        description="Mine FWER-validated association rules with SAME.",
    )
    p.add_argument("--input", "-i", required=True, type=Path,
                    help="Binary CSV (0/1 columns).")
    p.add_argument("--out", "-o", required=True, type=Path,
                    help="Output CSV of ranked rules.")
    p.add_argument("--mode", choices=["dfs", "opus"], default="dfs")
    p.add_argument("--max-k", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.05,
                    help="FWER target (default 0.05).")
    p.add_argument("--auto", action="store_true",
                    help="Use auto-derived hyperparameters (parameter-free mode).")
    p.add_argument("--top-k-rules", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fwer-only", action="store_true",
                    help="Keep only rules passing Tarone-Bonferroni.")
    args = p.parse_args(argv)

    from same_fim import SAME
    X, feature_names = _load(args.input)

    est = SAME(max_k=args.max_k, fwer_alpha=args.alpha,
                search_mode=args.mode, fdr_method="tarone",
                top_k_rules=args.top_k_rules, seed=args.seed,
                auto_hyperparams=args.auto)
    est.fit(X, feature_names=feature_names)

    import pandas as pd
    rules = est.result_.rules
    if args.fwer_only:
        rules = [r for r in rules if r.passes_fwer]
    rows = [{
        "antecedent": "&".join(str(i) for i in r.antecedent),
        "consequent": "&".join(str(i) for i in r.consequent),
        "support":    round(r.support, 4),
        "confidence": round(r.confidence, 4),
        "lift":       round(r.lift, 4),
        "p_value":    r.p_value,
        "passes_fwer": r.passes_fwer,
    } for r in rules]
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {len(rows)} rules to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
