"""
Build binarised versions of 3 new datasets: Chess (UCI kr-vs-kp),
Adult (UCI Census Income), and a synthetic market-basket T10I4D100K.

Saved to datasets/<name>.csv in the binary format SAME expects.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw"
OUT = ROOT / "datasets"


def build_chess():
    """UCI King-Rook-vs-King-Pawn. 3196 transactions, 36 categorical features
    (each {f,t,l,n,b,w,g}), 1 label (won/nowin). One-hot → binary.
    """
    cols = [f"f{i}" for i in range(36)] + ["label"]
    df = pd.read_csv(RAW / "chess.data", header=None, names=cols)
    # One-hot encode everything
    y = (df["label"] == "won").astype(np.int8).rename("won")
    X = pd.get_dummies(df.drop(columns=["label"]), prefix_sep="=").astype(np.int8)
    # Drop rare items (< 5% support) and ultra-dominant (> 95%)
    sup = X.mean()
    keep = (sup >= 0.05) & (sup <= 0.95)
    X = X.loc[:, keep]
    out = pd.concat([X, y], axis=1)
    out.to_csv(OUT / "chess.csv", index=False)
    print(f"  chess.csv: {out.shape} density={out.values.mean():.3f}")


def build_adult():
    """UCI Adult (Census Income). ~48k transactions, mixed types.
    Binarise continuous by quartile, one-hot categorical.
    """
    cols = ["age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week",
            "native_country", "income"]
    df = pd.read_csv(RAW / "adult.data", header=None, names=cols,
                     skipinitialspace=True, na_values="?")
    df = df.dropna().reset_index(drop=True)
    # Quartile-bin continuous
    continuous = ["age", "fnlwgt", "education_num",
                  "capital_gain", "capital_loss", "hours_per_week"]
    parts = []
    for col in continuous:
        # Use only unique quantile boundaries to avoid duplicates (e.g., capital_gain mostly 0)
        q = df[col].quantile([0.25, 0.5, 0.75]).unique()
        bins = [-np.inf] + sorted(q.tolist()) + [np.inf]
        labels = [f"{col}_b{i}" for i in range(len(bins) - 1)]
        cat = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
        parts.append(pd.get_dummies(cat).astype(np.int8))
    categorical = ["workclass", "education", "marital_status", "occupation",
                   "relationship", "race", "sex", "native_country"]
    for col in categorical:
        parts.append(pd.get_dummies(df[col], prefix=col).astype(np.int8))
    X = pd.concat(parts, axis=1)
    # Drop rare
    X = X.loc[:, X.mean() >= 0.02]
    y = (df["income"].str.strip() == ">50K").astype(np.int8).rename("income_gt50k")
    out = pd.concat([X, y], axis=1)
    out.to_csv(OUT / "adult.csv", index=False)
    print(f"  adult.csv: {out.shape} density={out.values.mean():.3f}")


def build_t10i4_synthetic():
    """Synthetic market-basket: 100,000 transactions, 870 unique items (synthetic
    equivalent to IBM generator's T10I4D100K). We sample items via a Zipfian
    distribution and embed a few hidden strong patterns for discovery.
    """
    rng = np.random.default_rng(42)
    n_tx = 100_000
    n_items = 100  # trimmed: top-100 items so the binary matrix is tractable
    zipf_a = 1.5
    X = np.zeros((n_tx, n_items), dtype=np.int8)
    # Each transaction has 5-15 items sampled from Zipf
    for i in range(n_tx):
        k = rng.integers(5, 16)
        items = rng.zipf(zipf_a, size=k * 2)
        items = items[items <= n_items] - 1
        X[i, items[:k]] = 1
    # Embed two strong hidden patterns (to test pattern recovery)
    # Pattern 1: items {0,1,2} co-occur in 3000 random transactions
    p1 = rng.choice(n_tx, 3000, replace=False)
    X[p1, 0] = 1; X[p1, 1] = 1; X[p1, 2] = 1
    # Pattern 2: items {10,11,12,13} co-occur in 2000 transactions
    p2 = rng.choice(n_tx, 2000, replace=False)
    X[p2, 10] = 1; X[p2, 11] = 1; X[p2, 12] = 1; X[p2, 13] = 1
    cols = [f"item_{i}" for i in range(n_items)]
    df = pd.DataFrame(X, columns=cols)
    # Keep items with support between 1% and 30%
    sup = df.mean()
    keep = (sup >= 0.01) & (sup <= 0.30)
    df = df.loc[:, keep]
    df.to_csv(OUT / "t10i4_synthetic.csv", index=False)
    print(f"  t10i4_synthetic.csv: {df.shape} density={df.values.mean():.3f}")


def main():
    print("Building datasets where SAME can shine:")
    build_chess()
    build_adult()
    build_t10i4_synthetic()
    print("Done.")


if __name__ == "__main__":
    main()
