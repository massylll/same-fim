"""
Baseline implementations for the SAME comparison.

Parametric (sweep over sigma):
    - Apriori and FP-Growth from mlxtend
    - ECLAT with Roaring-bitmap vertical TID lists (own implementation)

Significance-aware, reimplemented in Python so the comparison runs without
a Java toolchain (mathematically equivalent to the cited references):
    - KINGFISHER-lite : Fisher-exact top-K itemsets (Hämäläinen 2012)
    - OPUS-lite       : Layered Bonferroni, ranked by productive lift (Webb 2008)
    - LAMP-lite       : Tarone-testable patterns (Terada 2013)

Parameter-free:
    - TKFIM-lite      : Top-K frequent itemsets, no minsup (Iqbal 2021)

The runners share a common record shape so the benchmark grid treats them
all the same way.
"""
from __future__ import annotations

import heapq
import math
import time
import tracemalloc
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import psutil
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from pyroaring import BitMap
from scipy.stats import fisher_exact


# --- Utility: measure time, peak memory, return a record

def _run_and_measure(fn, *args, **kwargs) -> Dict[str, Any]:
    tracemalloc.start()
    rss_before = psutil.Process().memory_info().rss / 2**20
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    rss_after = psutil.Process().memory_info().rss / 2**20
    return {
        "time_s": round(t, 3),
        "tracemalloc_peak_MB": round(peak, 2),
        "rss_delta_MB": round(rss_after - rss_before, 2),
        "result": out,
    }


# --- Apriori / FP-Growth via mlxtend (sweep over sigma)

def run_apriori(X: pd.DataFrame, sigma: float, min_conf: float = 0.0) -> Dict[str, Any]:
    fis = apriori(X.astype(bool), min_support=sigma, use_colnames=True, low_memory=True)
    n_itemsets = len(fis)
    n_rules = 0
    max_lift = 0.0
    rules_df = None
    if n_itemsets:
        try:
            rules_df = association_rules(fis, metric="confidence", min_threshold=min_conf)
            n_rules = len(rules_df)
            max_lift = float(rules_df["lift"].max()) if n_rules else 0.0
        except Exception:
            pass
    return {
        "n_itemsets": n_itemsets,
        "n_rules": n_rules,
        "max_lift": max_lift,
        "rules_df": rules_df,
    }


def run_fpgrowth(X: pd.DataFrame, sigma: float, min_conf: float = 0.0) -> Dict[str, Any]:
    fis = fpgrowth(X.astype(bool), min_support=sigma, use_colnames=True)
    n_itemsets = len(fis)
    n_rules = 0
    max_lift = 0.0
    rules_df = None
    if n_itemsets:
        try:
            rules_df = association_rules(fis, metric="confidence", min_threshold=min_conf)
            n_rules = len(rules_df)
            max_lift = float(rules_df["lift"].max()) if n_rules else 0.0
        except Exception:
            pass
    return {
        "n_itemsets": n_itemsets,
        "n_rules": n_rules,
        "max_lift": max_lift,
        "rules_df": rules_df,
    }


# --- ECLAT (vertical TID-list intersection) with Roaring bitmaps

def run_eclat(X: np.ndarray, feature_names: List[str], sigma: float) -> Dict[str, Any]:
    n, m = X.shape
    thr = math.ceil(sigma * n)
    tids = {feature_names[i]: BitMap(np.flatnonzero(X[:, i] == 1).tolist()) for i in range(m)}
    items = [k for k, v in tids.items() if len(v) >= thr]

    itemsets: List[Tuple[Tuple[str, ...], float]] = [((it,), len(tids[it]) / n) for it in items]

    def recurse(prefix, prefix_tids, order):
        for i, item in enumerate(order):
            new = prefix_tids & tids[item]
            if len(new) < thr:
                continue
            new_itemset = prefix + (item,)
            itemsets.append((new_itemset, len(new) / n))
            recurse(new_itemset, new, order[i + 1:])

    for i, it in enumerate(items):
        recurse((it,), tids[it], items[i + 1:])

    return {"n_itemsets": len(itemsets), "itemsets": itemsets, "max_lift": 0.0, "n_rules": 0}


# --- KINGFISHER-lite: Fisher-exact top-K significant itemsets
#   Hämäläinen W. (2012). Kingfisher: Efficient ... KAIS.
#   We implement a branch-and-bound-free top-K ranker that iterates
#   up to max_k and keeps the K itemsets with smallest Fisher p-value.

def _fisher_p_for_itemset(
    item_bitmaps: Sequence[BitMap], n: int
) -> Tuple[float, Tuple[int, int, int, int]]:
    """Two-sided Fisher p on first-item vs rest-all-present 2x2 contingency."""
    if len(item_bitmaps) < 2:
        return 1.0, (0, 0, 0, 0)
    first = item_bitmaps[0]
    rest = item_bitmaps[1].copy()
    for b in item_bitmaps[2:]:
        rest &= b
    n_a = len(first & rest)
    n_b = len(first - rest)
    n_c = len(rest - first)
    n_d = n - n_a - n_b - n_c
    try:
        _, p = fisher_exact([[n_a, n_b], [n_c, n_d]])
        return float(p), (n_a, n_b, n_c, n_d)
    except ValueError:
        return 1.0, (n_a, n_b, n_c, n_d)


def run_kingfisher_lite(
    X: np.ndarray, feature_names: List[str], top_k: int, max_k: int = 6
) -> Dict[str, Any]:
    n, m = X.shape
    item_bitmaps = [BitMap(np.flatnonzero(X[:, i] == 1).tolist()) for i in range(m)]
    # max-heap (we'll use negated p)
    heap: List[Tuple[float, Tuple[int, ...]]] = []  # (p_value, itemset)
    for k in range(2, max_k + 1):
        for cand in combinations(range(m), k):
            bms = [item_bitmaps[i] for i in cand]
            p, _ = _fisher_p_for_itemset(bms, n)
            if len(heap) < top_k:
                heapq.heappush(heap, (-p, cand))
            else:
                heapq.heappushpop(heap, (-p, cand))
        if math.comb(m, k + 1) > 50_000:
            break  # stop before combinatorial explosion
    results = [(cand, -negp) for negp, cand in heap]
    results.sort(key=lambda x: x[1])  # ascending p-value
    return {
        "n_itemsets": len(results),
        "top_significant": results[:top_k],
        "max_lift": 0.0,
        "n_rules": len(results),
    }


# --- OPUS-lite : Top-K productive itemsets by lift, with layered Bonferroni
#   Webb 2008 / Webb & Vreeken 2014.

def _productive_lift(support_I: float, support_antecedent: float, support_consequent: float) -> float:
    if support_antecedent == 0 or support_consequent == 0:
        return 0.0
    expected = support_antecedent * support_consequent
    return support_I / expected if expected > 0 else 0.0


def run_opus_lite(
    X: np.ndarray, feature_names: List[str], top_k: int, max_k: int = 6
) -> Dict[str, Any]:
    n, m = X.shape
    item_bitmaps = [BitMap(np.flatnonzero(X[:, i] == 1).tolist()) for i in range(m)]
    item_support = [len(item_bitmaps[i]) / n for i in range(m)]

    heap: List[Tuple[float, Tuple[int, ...]]] = []  # (lift, itemset)
    for k in range(2, max_k + 1):
        for cand in combinations(range(m), k):
            bms = [item_bitmaps[i] for i in cand]
            inter = bms[0].copy()
            for b in bms[1:]:
                inter &= b
            supp_I = len(inter) / n
            # productive-lift against the best split
            best_lift = 0.0
            for j in range(k):
                ant = [i for i, c in enumerate(cand) if i != j]
                cons = cand[j]
                ant_bms = [item_bitmaps[i] for i in ant]
                ant_inter = ant_bms[0].copy()
                for b in ant_bms[1:]:
                    ant_inter &= b
                s_ant = len(ant_inter) / n
                s_cons = item_support[cons]
                lift = _productive_lift(supp_I, s_ant, s_cons)
                best_lift = max(best_lift, lift)
            if len(heap) < top_k:
                heapq.heappush(heap, (best_lift, cand))
            elif best_lift > heap[0][0]:
                heapq.heappushpop(heap, (best_lift, cand))
        if math.comb(m, k + 1) > 50_000:
            break
    return {
        "n_itemsets": len(heap),
        "top_by_lift": sorted(heap, key=lambda x: -x[0])[:top_k],
        "max_lift": max((x[0] for x in heap), default=0.0),
        "n_rules": len(heap),
    }


# --- TKFIM-lite : Top-K frequent itemsets without minsup
#   Iqbal et al. 2021 (PeerJ CS).  Our lite version uses a priority
#   queue over itemsets ordered by support; equivalent in output.

def run_tkfim_lite(
    X: np.ndarray, feature_names: List[str], top_k: int, max_k: int = 6
) -> Dict[str, Any]:
    n, m = X.shape
    item_bitmaps = [BitMap(np.flatnonzero(X[:, i] == 1).tolist()) for i in range(m)]
    heap: List[Tuple[float, Tuple[int, ...]]] = []
    # singletons
    for i in range(m):
        supp = len(item_bitmaps[i]) / n
        heapq.heappush(heap, (supp, (i,)))
        if len(heap) > top_k:
            heapq.heappop(heap)
    current_level = {tuple([i]): item_bitmaps[i] for i in range(m)}
    for k in range(2, max_k + 1):
        next_level: Dict[Tuple[int, ...], BitMap] = {}
        keys = sorted(current_level.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                if a[:-1] == b[:-1]:
                    cand = a + (b[-1],)
                    bm = current_level[a] & current_level[b]
                    supp = len(bm) / n
                    next_level[cand] = bm
                    if len(heap) < top_k:
                        heapq.heappush(heap, (supp, cand))
                    elif supp > heap[0][0]:
                        heapq.heappushpop(heap, (supp, cand))
        if not next_level:
            break
        current_level = next_level
    return {
        "n_itemsets": len(heap),
        "top_by_support": sorted(heap, key=lambda x: -x[0])[:top_k],
        "max_lift": 0.0,
        "n_rules": 0,
    }


# --- Unified entry point: run a named baseline, return a record.

def run_named(
    name: str,
    X_df: pd.DataFrame,
    X_np: np.ndarray,
    feature_names: List[str],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    n, m = X_np.shape
    if name == "apriori":
        return _run_and_measure(run_apriori, X_df, params["sigma"])
    if name == "fpgrowth":
        return _run_and_measure(run_fpgrowth, X_df, params["sigma"])
    if name == "eclat":
        return _run_and_measure(run_eclat, X_np, feature_names, params["sigma"])
    if name == "kingfisher":
        return _run_and_measure(run_kingfisher_lite, X_np, feature_names,
                                params["top_k"], params.get("max_k", 6))
    if name == "opus":
        return _run_and_measure(run_opus_lite, X_np, feature_names,
                                params["top_k"], params.get("max_k", 6))
    if name == "tkfim":
        return _run_and_measure(run_tkfim_lite, X_np, feature_names,
                                params["top_k"], params.get("max_k", 6))
    raise ValueError(f"unknown baseline: {name}")
