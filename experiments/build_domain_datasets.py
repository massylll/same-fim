"""
Build domain-appropriate binary datasets for the SAME paper.
Domains (per user spec):
  1. Neuroimaging / computational neuroscience
  2. Population genomics
  3. Protein analysis, structure, and dynamics

Already have:
  - ABIDE (neuroimaging)            — 1,112 x 14
  - EEG Eye State (neuroscience)     — 14,980 x 15
  - ClinVar (clinical genomics)      — 12,000 x 32

Adding:
  - Pfam-UniProt (protein family co-occurrence) — ~120k x top-N Pfam families
  - 1000 Genomes chrY SNP subset (pop. genomics) — attempted
  - Synthetic brain connectivity (neuroimaging) — for scaling experiments
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw"
OUT = ROOT / "datasets"


def build_pfam():
    """Convert UniProt Pfam annotations to a protein x Pfam-family binary matrix.

    Each row = one protein, each column = one Pfam family; cell = 1 if the
    protein contains a hit for that family. Top-frequency families only
    (to keep the matrix tractable).
    """
    src = RAW / "uniprot_pfam_stream.tsv"
    df = pd.read_csv(src, sep="\t")
    df.columns = ["entry", "pfam"]
    df = df.dropna(subset=["pfam"])
    # Each entry's pfam field is semicolon-separated
    df["pfam_list"] = df["pfam"].str.rstrip(";").str.split(";")
    # Flatten to protein-family pairs
    rows = []
    for entry, pfams in zip(df["entry"], df["pfam_list"]):
        for p in pfams:
            p = p.strip()
            if p:
                rows.append((entry, p))
    long_df = pd.DataFrame(rows, columns=["entry", "family"])
    print(f"  raw protein-family pairs: {len(long_df)}")

    # Pick top-60 most-frequent Pfam families
    family_counts = long_df["family"].value_counts()
    top_families = family_counts.head(60).index.tolist()
    long_df = long_df[long_df["family"].isin(top_families)]

    # Pivot to wide binary matrix
    pivot = long_df.pivot_table(index="entry", columns="family",
                                  values="family", aggfunc="count", fill_value=0)
    pivot = (pivot > 0).astype(np.int8)
    # Keep proteins with at least 1 family hit (drop empty rows)
    pivot = pivot.loc[pivot.sum(axis=1) >= 1]
    # Reset index so rows are sequential
    pivot = pivot.reset_index(drop=True)

    # Sample 30k rows if larger (for tractability)
    if len(pivot) > 30_000:
        pivot = pivot.sample(n=30_000, random_state=42).reset_index(drop=True)

    pivot.to_csv(OUT / "pfam_proteins.csv", index=False)
    print(f"  pfam_proteins.csv: {pivot.shape}  density={pivot.values.mean():.3f}")


def build_synth_neuro(n_subjects=50_000, n_regions=50, seed=42):
    """Synthetic brain-connectivity binary matrix for scaling experiments.

    Each row represents a subject; each column represents the presence of
    a significant functional connection between two brain regions under
    a threshold. We embed 3 plausible network "modules" (clusters of
    co-active regions) to generate realistic cohesion.
    """
    rng = np.random.default_rng(seed)
    X = (rng.random((n_subjects, n_regions)) < 0.25).astype(np.int8)
    # Inject 3 network modules with correlated firing
    for mod_start, mod_size, prob in [(0, 8, 0.7), (15, 6, 0.6), (30, 5, 0.65)]:
        mask = rng.random(n_subjects) < 0.35  # 35% of subjects show this module
        for col in range(mod_start, mod_start + mod_size):
            X[mask, col] = (rng.random(mask.sum()) < prob).astype(np.int8)
    cols = [f"region_{i:02d}" for i in range(n_regions)]
    df = pd.DataFrame(X, columns=cols)
    df.to_csv(OUT / "synth_neuro.csv", index=False)
    print(f"  synth_neuro.csv: {df.shape}  density={df.values.mean():.3f}")


def main():
    print("Building domain-appropriate datasets:")
    build_pfam()
    build_synth_neuro()
    print("Done.")


if __name__ == "__main__":
    main()
