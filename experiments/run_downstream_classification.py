"""
Downstream task: does using SAME's FWER-validated rules as features
improve a predictive task, compared to raw features or rules from
parametric baselines?

On ABIDE (binary columns include diagnostic label 'autisme'), we:
  1. Train logistic regression on raw binary features (baseline)
  2. Train on SAME rule indicators (each rule becomes a 0/1 feature:
     1 if the transaction satisfies the rule body)
  3. Train on Apriori rule indicators at matched rule count
  4. Report 5-fold CV macro-F1

This shows whether SAME's statistically-valid rules carry predictive
signal beyond the raw features.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def rules_to_features(rules, X, feature_names):
    """For each rule, create a column = 1 iff transaction satisfies the
    rule's antecedent AND consequent items."""
    name2idx = {n: i for i, n in enumerate(feature_names)}
    feats = []
    labels = []
    for r in rules:
        items = list(r.antecedent) + list(r.consequent)
        if not items:
            continue
        feat_col = np.ones(len(X), dtype=np.int8)
        for item in items:
            if isinstance(item, str):
                if item not in name2idx:
                    feat_col = np.zeros(len(X), dtype=np.int8); break
                feat_col &= X[:, name2idx[item]]
            else:
                if item >= X.shape[1]:
                    feat_col = np.zeros(len(X), dtype=np.int8); break
                feat_col &= X[:, item]
        feats.append(feat_col)
        labels.append("&".join([feature_names[i] if isinstance(i,int) else str(i) for i in items]))
    if not feats:
        return np.zeros((len(X), 0), dtype=np.int8), []
    return np.column_stack(feats), labels


def main():
    df = pd.read_csv(ROOT / "datasets" / "abide.csv").astype(np.int8)
    target_col = "autisme"
    y = df[target_col].values
    X_raw = df.drop(columns=[target_col]).values.astype(np.int8)
    feature_names = list(df.drop(columns=[target_col]).columns)

    results = []

    # 1. Raw features baseline
    clf = LogisticRegression(max_iter=1000, solver='liblinear')
    f1_raw = cross_val_score(clf, X_raw, y, cv=5, scoring='f1_macro').mean()
    print(f"Raw features (baseline):        F1_macro = {f1_raw:.4f}")
    results.append({"method": "raw_features", "n_features": X_raw.shape[1], "f1_macro": round(f1_raw, 4)})

    # 2. SAME DFS rules as features
    from same_fim import SAME as SAMEv6
    # Full data for mining
    est = SAMEv6(max_k=10, seed=42, mode="all", test_method="gtest",
                 top_k_rules=500, fwer_alpha=0.05, search_mode="dfs",
                 fdr_method="tarone")
    est.fit(df.values, feature_names=list(df.columns))
    rules = [r for r in est.result_.rules if r.passes_fwer]
    print(f"SAME DFS: {len(rules)} FWER-passing rules")
    X_same, _ = rules_to_features(rules[:200], df.values, list(df.columns))
    # Remove target column from rule indicator features if present
    if X_same.shape[1] > 0:
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
        f1_same_dfs = cross_val_score(clf, X_same, y, cv=5, scoring='f1_macro').mean()
    else:
        f1_same_dfs = 0.0
    print(f"SAME DFS rules:                 F1_macro = {f1_same_dfs:.4f}  ({X_same.shape[1]} rule features)")
    results.append({"method": "SAME_v6_dfs_rules", "n_features": X_same.shape[1],
                     "f1_macro": round(f1_same_dfs, 4)})

    # 3. SAME OPUS rules as features
    est_opus = SAMEv6(max_k=10, seed=42, mode="all", test_method="gtest",
                      top_k_rules=500, fwer_alpha=0.05, search_mode="opus",
                      fdr_method="tarone", opus_top_k=200)
    est_opus.fit(df.values, feature_names=list(df.columns))
    rules_opus = [r for r in est_opus.result_.rules if r.passes_fwer]
    X_opus, _ = rules_to_features(rules_opus[:200], df.values, list(df.columns))
    if X_opus.shape[1] > 0:
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
        f1_opus = cross_val_score(clf, X_opus, y, cv=5, scoring='f1_macro').mean()
    else:
        f1_opus = 0.0
    print(f"SAME OPUS rules:                F1_macro = {f1_opus:.4f}  ({X_opus.shape[1]} rule features)")
    results.append({"method": "SAME_v6_opus_rules", "n_features": X_opus.shape[1],
                     "f1_macro": round(f1_opus, 4)})

    # 4. Apriori rules (matched count, no FWER)
    from mlxtend.frequent_patterns import apriori, association_rules
    df_bool = pd.DataFrame(df.values.astype(bool), columns=df.columns)
    fis = apriori(df_bool, min_support=0.10, use_colnames=True, low_memory=True)
    apr_rules = association_rules(fis, metric="confidence", min_threshold=0.5)
    # Take top-200 by lift
    apr_rules = apr_rules.nlargest(200, "lift")
    # Convert to same shape: each rule's antecedent items
    apr_feats = []
    for _, r in apr_rules.iterrows():
        items = list(r["antecedents"]) + list(r["consequents"])
        col = np.ones(len(df), dtype=np.int8)
        valid = True
        for item in items:
            if item in df.columns:
                col &= df[item].values
            else:
                valid = False; break
        if valid:
            apr_feats.append(col)
    if apr_feats:
        X_apr = np.column_stack(apr_feats)
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
        f1_apr = cross_val_score(clf, X_apr, y, cv=5, scoring='f1_macro').mean()
    else:
        X_apr = np.zeros((len(df),0), dtype=np.int8); f1_apr = 0.0
    print(f"Apriori rules (no FWER):        F1_macro = {f1_apr:.4f}  ({X_apr.shape[1]} rule features)")
    results.append({"method": "apriori_rules", "n_features": X_apr.shape[1], "f1_macro": round(f1_apr, 4)})

    # 5. Raw features + SAME rules combined
    if X_same.shape[1] > 0:
        X_combined = np.hstack([X_raw, X_same])
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
        f1_combined = cross_val_score(clf, X_combined, y, cv=5, scoring='f1_macro').mean()
        print(f"Raw + SAME rules combined:      F1_macro = {f1_combined:.4f}  ({X_combined.shape[1]} features)")
        results.append({"method": "raw+SAME_dfs", "n_features": X_combined.shape[1],
                         "f1_macro": round(f1_combined, 4)})

    out = ROOT / "experiments" / "out" / "downstream_abide_classification.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
