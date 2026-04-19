"""
Fixed-sigma Apriori with post-hoc Bonferroni / Holm FWER correction.

The miner itself is unchanged Apriori (mlxtend) at a user-chosen sigma. After
enumeration, each rule is scored by a Fisher exact p-value on its 2x2 table
against the independence product, and the family-wise error rate is controlled
by Bonferroni (default) or Holm step-down over the full enumerated set.

This baseline answers a reviewer question the significance-aware table leaves
open: how much of SAME's edge comes from the adaptive threshold itself, and
how much from attaching any FWER correction on top of a classical miner? Same
Fisher test, same alpha; only the support threshold differs.

Usage as a library:

    from apriori_bonferroni import mine_apriori_fwer
    rules = mine_apriori_fwer(df_binary, sigma=0.10, alpha=0.05, method="bonferroni")

From the command line:

    python apriori_bonferroni.py --csv datasets/abide.csv --sigma 0.10 --alpha 0.05
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


@dataclass
class FWERRule:
    antecedent: tuple
    consequent: tuple
    support: float
    confidence: float
    lift: float
    p_value: float
    p_adjusted: float
    passes_fwer: bool


def _fisher_p_for_rule(
    a_tids: np.ndarray, c_tids: np.ndarray, n: int
) -> float:
    """Fisher exact p-value for the 2x2 (antecedent, consequent) table."""
    a = int(np.sum(a_tids & c_tids))
    b = int(np.sum(a_tids & ~c_tids))
    c = int(np.sum(~a_tids & c_tids))
    d = int(n - a - b - c)
    try:
        _, p = fisher_exact([[a, b], [c, d]])
        return float(p)
    except ValueError:
        return 1.0


def mine_apriori_fwer(
    df: pd.DataFrame,
    sigma: float = 0.10,
    alpha: float = 0.05,
    method: Literal["bonferroni", "holm"] = "bonferroni",
    max_len: Optional[int] = 5,
) -> List[FWERRule]:
    """Run Apriori at support sigma, then Fisher + Bonferroni/Holm over the rules."""
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError as exc:
        raise SystemExit(
            "mlxtend is required. Install with: pip install mlxtend"
        ) from exc

    data_binary = df.astype(bool)
    n = len(data_binary)

    frequent = apriori(data_binary, min_support=sigma, use_colnames=True, max_len=max_len)
    if frequent.empty:
        return []

    rules_df = association_rules(frequent, metric="confidence", min_threshold=0.0)
    if rules_df.empty:
        return []

    col_tids = {col: data_binary[col].values for col in data_binary.columns}

    def tids_for(itemset) -> np.ndarray:
        out = np.ones(n, dtype=bool)
        for col in itemset:
            out &= col_tids[col]
        return out

    raw_ps: List[float] = []
    for _, row in rules_df.iterrows():
        a_tids = tids_for(row["antecedents"])
        c_tids = tids_for(row["consequents"])
        raw_ps.append(_fisher_p_for_rule(a_tids, c_tids, n))

    m = len(raw_ps)
    raw_arr = np.asarray(raw_ps)

    if method == "bonferroni":
        adj = np.minimum(raw_arr * m, 1.0)
        passes = raw_arr <= (alpha / m)
    elif method == "holm":
        order = np.argsort(raw_arr)
        adj_sorted = np.empty(m)
        running_max = 0.0
        for rank, idx in enumerate(order):
            adj_sorted[idx] = min(raw_arr[idx] * (m - rank), 1.0)
            running_max = max(running_max, adj_sorted[idx])
            adj_sorted[idx] = running_max
        adj = adj_sorted
        passes = adj <= alpha
    else:
        raise ValueError(f"unknown correction method: {method!r}")

    out: List[FWERRule] = []
    for i, (_, row) in enumerate(rules_df.iterrows()):
        out.append(
            FWERRule(
                antecedent=tuple(row["antecedents"]),
                consequent=tuple(row["consequents"]),
                support=float(row["support"]),
                confidence=float(row["confidence"]),
                lift=float(row["lift"]),
                p_value=float(raw_arr[i]),
                p_adjusted=float(adj[i]),
                passes_fwer=bool(passes[i]),
            )
        )
    return out


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--csv", required=True, help="binary CSV input")
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--method", choices=["bonferroni", "holm"], default="bonferroni"
    )
    parser.add_argument("--max-len", type=int, default=5)
    parser.add_argument("--out", default=None, help="optional CSV for the surviving rules")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if not df.isin([0, 1]).all().all():
        for col in df.columns:
            thr = df[col].median()
            df[col] = (df[col] > thr).astype(int)

    t0 = time.time()
    rules = mine_apriori_fwer(
        df, sigma=args.sigma, alpha=args.alpha,
        method=args.method, max_len=args.max_len,
    )
    elapsed = time.time() - t0

    total = len(rules)
    passing = sum(1 for r in rules if r.passes_fwer)

    print(f"dataset:    {args.csv}")
    print(f"sigma:      {args.sigma}")
    print(f"alpha:      {args.alpha}  ({args.method})")
    print(f"max_len:    {args.max_len}")
    print(f"time:       {elapsed:.3f} s")
    print(f"rules:      {total}")
    print(f"FWER-valid: {passing}  ({100.0 * passing / total:.1f}% of rules)" if total else "FWER-valid: 0")

    if args.out:
        df_out = pd.DataFrame([asdict(r) for r in rules])
        df_out.to_csv(args.out, index=False)
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(_cli())
