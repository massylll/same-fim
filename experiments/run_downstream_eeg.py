"""
Downstream task on EEG Eye State: predict the `eyeDetection` label (eye
open/closed) from the 14 electrode channels (median-binarised).

Four feature sets are compared at matched rule count:
  1. raw binary features (14 columns)
  2. top-K SAME (DFS+Tarone) FWER-valid rules by lift
  3. top-K SAME (OPUS+Tarone) FWER-valid rules by lift
  4. top-K Apriori rules at sigma=0.20 ranked by lift (no FWER filter)

Metric: 5-fold stratified macro-F1 of sklearn LogisticRegression.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def rules_to_features(rules, X, feature_names):
    name2idx = {n: i for i, n in enumerate(feature_names)}
    feats = []
    for r in rules:
        items = list(r.antecedent) + list(r.consequent)
        if not items:
            continue
        col = np.ones(len(X), dtype=np.int8)
        ok = True
        for item in items:
            if isinstance(item, str):
                if item not in name2idx:
                    ok = False; break
                col &= X[:, name2idx[item]]
            else:
                if item >= X.shape[1]:
                    ok = False; break
                col &= X[:, item]
        if ok:
            feats.append(col)
    if not feats:
        return np.zeros((len(X), 0), dtype=np.int8)
    return np.column_stack(feats)


def main():
    df = pd.read_csv(ROOT / "datasets" / "eeg_eye_state.csv").astype(np.int8)
    target = "eyeDetection"
    y = df[target].values
    X_raw = df.drop(columns=[target]).values.astype(np.int8)
    feature_names = list(df.drop(columns=[target]).columns)
    K = 200

    results = []

    # 1. raw
    clf = LogisticRegression(max_iter=1000, solver='liblinear')
    f1 = cross_val_score(clf, X_raw, y, cv=5, scoring='f1_macro').mean()
    print(f"Raw features:        F1_macro = {f1:.4f}")
    results.append({"method": "raw_features", "n_features": X_raw.shape[1],
                     "f1_macro": round(f1, 4)})

    # 2. SAME DFS
    from same_fim import SAME as SAMEv6
    est = SAMEv6(max_k=7, seed=42, mode="all", test_method="gtest",
                  top_k_rules=5000, fwer_alpha=0.05, search_mode="dfs",
                  fdr_method="tarone")
    est.fit(df.values, feature_names=list(df.columns))
    rules = [r for r in est.result_.rules if r.passes_fwer][:K]
    X_same = rules_to_features(rules, df.values, list(df.columns))
    if X_same.shape[1] > 0:
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
        f1_same = cross_val_score(clf, X_same, y, cv=5, scoring='f1_macro').mean()
    else:
        f1_same = 0.0
    print(f"SAME DFS rules:      F1_macro = {f1_same:.4f}  ({X_same.shape[1]} rule features)")
    results.append({"method": "SAME_v6_dfs_rules", "n_features": X_same.shape[1],
                     "f1_macro": round(f1_same, 4)})

    # 3. SAME OPUS
    est_opus = SAMEv6(max_k=7, seed=42, mode="all", test_method="gtest",
                      top_k_rules=5000, fwer_alpha=0.05, search_mode="opus",
                      fdr_method="tarone", opus_top_k=500)
    est_opus.fit(df.values, feature_names=list(df.columns))
    rules_opus = [r for r in est_opus.result_.rules if r.passes_fwer][:K]
    X_opus = rules_to_features(rules_opus, df.values, list(df.columns))
    if X_opus.shape[1] > 0:
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
        f1_opus = cross_val_score(clf, X_opus, y, cv=5, scoring='f1_macro').mean()
    else:
        f1_opus = 0.0
    print(f"SAME OPUS rules:     F1_macro = {f1_opus:.4f}  ({X_opus.shape[1]} rule features)")
    results.append({"method": "SAME_v6_opus_rules", "n_features": X_opus.shape[1],
                     "f1_macro": round(f1_opus, 4)})

    # 4. Apriori at sigma=0.20
    from mlxtend.frequent_patterns import apriori, association_rules
    df_bool = pd.DataFrame(df.values.astype(bool), columns=df.columns)
    fis = apriori(df_bool, min_support=0.20, use_colnames=True, low_memory=True)
    apr = association_rules(fis, metric="confidence", min_threshold=0.5)
    apr = apr.nlargest(K, "lift")
    apr_feats = []
    for _, r in apr.iterrows():
        items = list(r["antecedents"]) + list(r["consequents"])
        col = np.ones(len(df), dtype=np.int8)
        ok = True
        for it in items:
            if it in df.columns:
                col &= df[it].values
            else:
                ok = False; break
        if ok:
            apr_feats.append(col)
    if apr_feats:
        X_apr = np.column_stack(apr_feats)
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
        f1_apr = cross_val_score(clf, X_apr, y, cv=5, scoring='f1_macro').mean()
    else:
        X_apr = np.zeros((len(df), 0), dtype=np.int8); f1_apr = 0.0
    print(f"Apriori rules:       F1_macro = {f1_apr:.4f}  ({X_apr.shape[1]} rule features)")
    results.append({"method": "apriori_rules", "n_features": X_apr.shape[1],
                     "f1_macro": round(f1_apr, 4)})

    out = ROOT / "experiments" / "out" / "downstream_eeg_classification.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
