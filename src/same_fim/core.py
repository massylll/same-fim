"""
SAME v6 — production miner backing the `same-fim` package.

Extends v5 with an exact LAMP testable count, column-permutation
Westfall-Young, an OPUS-style best-first search for top-K lift, and a
Hoeffding-triggered `partial_fit` for streaming updates. Everything else
carries over: vectorised G-test, chunked evaluation, ECLAT DFS, optional
closure, Webb's layered Bonferroni allocation for gamma_k.
"""
from __future__ import annotations

import heapq
import json
import math
import os
import signal
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import psutil
from pyroaring import BitMap
from scipy.stats import chi2 as chi2_dist
from scipy.stats import fisher_exact
from sklearn.base import BaseEstimator


# ---------------------------------------------------------------------
# Information gain (unified, [0, 1])
# ---------------------------------------------------------------------

def monotone_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0 if p <= 0 else 2.0
    h = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    return h if p <= 0.5 else 2 - h


def monotone_entropy_vec(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return np.where(p <= 0.5, h, 2 - h)


def information_gain(p: float, p_bg: float = 0.5) -> float:
    m_max = 2.0
    m_bg = monotone_entropy(p_bg)
    denom = max(m_max - m_bg, 1e-9)
    return max(0.0, min(1.0, (monotone_entropy(p) - m_bg) / denom))


def information_gain_upper_envelope(supp_max: float, p_bg: float = 0.5) -> float:
    """Largest IG reachable given supp <= supp_max. Used for sound early-exit."""
    candidates = [0.0, supp_max]
    p_star = 1 - p_bg
    if 0 <= p_star <= supp_max:
        candidates.append(p_star)
    return max(information_gain(p, p_bg) for p in candidates)


# ---------------------------------------------------------------------
# Cohesion measures on the 2x2 table (phi is null-invariant; see Thm 4)
# ---------------------------------------------------------------------

def cohesion_phi(n11: int, n10: int, n01: int, n00: int) -> float:
    denom = math.sqrt(
        max((n11 + n10) * (n11 + n01) * (n00 + n10) * (n00 + n01), 1)
    )
    phi = (n11 * n00 - n10 * n01) / denom
    return (phi + 1) / 2


def cohesion_symmetric(n11: int, n10: int, n01: int, n00: int) -> float:
    n = n11 + n10 + n01 + n00
    return (n11 + n00) / n if n else 0.0


def cohesion_ochiai(n11: int, n10: int, n01: int, n00: int) -> float:
    denom = math.sqrt(max((n11 + n10) * (n11 + n01), 1))
    return n11 / denom if denom else 0.0


def cohesion_pmi(n11: int, n10: int, n01: int, n00: int) -> float:
    n = n11 + n10 + n01 + n00
    if n == 0 or n11 == 0:
        return 0.0
    p11 = n11 / n
    p1_ = (n11 + n10) / n
    p_1 = (n11 + n01) / n
    if p1_ == 0 or p_1 == 0 or p11 == 0:
        return 0.0
    pmi = math.log(p11 / (p1_ * p_1))
    npmi = pmi / (-math.log(p11))
    return max(min((npmi + 1) / 2, 1.0), 0.0)


COHESION_FUNCS = {
    "phi":       cohesion_phi,
    "symmetric": cohesion_symmetric,
    "ochiai":    cohesion_ochiai,
    "pmi":       cohesion_pmi,
}


# ---------------------------------------------------------------------
# G-test (vectorised and single-table) + Fisher fallback
# ---------------------------------------------------------------------

def g_test_batch(contingency_tables: np.ndarray) -> np.ndarray:
    if len(contingency_tables) == 0:
        return np.array([], dtype=np.float64)
    O = contingency_tables.astype(np.float64) + 0.5
    row_sums = O.sum(axis=2, keepdims=True)
    col_sums = O.sum(axis=1, keepdims=True)
    total = O.sum(axis=(1, 2), keepdims=True)
    E = row_sums * col_sums / total
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(E > 0, O / E, 1.0)
        G = 2.0 * np.sum(O * np.log(ratio), axis=(1, 2))
    G = np.maximum(G, 0.0)
    p_values = chi2_dist.sf(G, df=1)
    return p_values


def g_test_single_table(table_2x2: np.ndarray) -> float:
    O = table_2x2.astype(np.float64) + 0.5
    row_s = O.sum(axis=1, keepdims=True)
    col_s = O.sum(axis=0, keepdims=True)
    total = O.sum()
    if total == 0:
        return 1.0
    E = row_s * col_s / total
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(E > 0, O / E, 1.0)
        G = 2.0 * float(np.sum(O * np.log(ratio)))
    G = max(G, 0.0)
    return float(chi2_dist.sf(G, df=1))


def fisher_p(n11: int, n10: int, n01: int, n00: int) -> float:
    try:
        _, p = fisher_exact([[n11, n10], [n01, n00]])
        return float(p)
    except ValueError:
        return 1.0


def needs_fisher_fallback(table: np.ndarray) -> bool:
    """True when any expected cell count falls below 5 — use Fisher instead of G."""
    O = table.astype(np.float64)
    row_sums = O.sum(axis=1, keepdims=True)
    col_sums = O.sum(axis=0, keepdims=True)
    total = O.sum()
    if total == 0:
        return True
    E = row_sums * col_sums / total
    return bool(np.any(E < 5))


# ---------------------------------------------------------------------
# Tarone-Bonferroni, Westfall-Young, Benjamini-Hochberg
# ---------------------------------------------------------------------

def minimum_achievable_p(n1_: int, n_1: int, n: int) -> float:
    """Smallest attainable p-value given the row/column margins (Tarone 1990)."""
    a_star = min(n1_, n_1)
    b = n1_ - a_star
    c = n_1 - a_star
    d = n - n1_ - n_1 + a_star
    table = np.array([[a_star, b], [c, d]], dtype=np.float64) + 0.5
    row_s = table.sum(axis=1, keepdims=True)
    col_s = table.sum(axis=0, keepdims=True)
    total = table.sum()
    if total == 0:
        return 1.0
    E = row_s * col_s / total
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(E > 0, table / E, 1.0)
        G = 2.0 * float(np.sum(table * np.log(ratio)))
    G = max(G, 0.0)
    return float(chi2_dist.sf(G, df=1))


def minimum_achievable_p_batch(n1_arr: np.ndarray, n_1_arr: np.ndarray,
                                n: int) -> np.ndarray:
    n1 = n1_arr.astype(np.float64)
    n_1 = n_1_arr.astype(np.float64)
    a_star = np.minimum(n1, n_1)
    b = n1 - a_star
    c = n_1 - a_star
    d = n - n1 - n_1 + a_star
    tables = np.stack([
        np.stack([a_star, b], axis=-1),
        np.stack([c, d], axis=-1),
    ], axis=-2) + 0.5
    return g_test_batch(tables)


def tarone_threshold(min_ps: Sequence[float], alpha: float = 0.05) -> Tuple[float, int]:
    """Highest k such that the k smallest min-p-values still satisfy p <= alpha/k.
    Returns (alpha/k, k). When no k qualifies, the threshold is the conservative
    alpha/n fallback.
    """
    if not min_ps:
        return alpha, 0
    min_ps_sorted = sorted(min_ps)
    n = len(min_ps_sorted)
    best_k = 0
    for k in range(1, n + 1):
        if min_ps_sorted[k - 1] <= alpha / k:
            best_k = k
        else:
            continue
    if best_k == 0:
        return alpha / n, n
    return alpha / best_k, best_k


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(p_values)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p)
    thresh = np.arange(1, n + 1) * alpha / n
    passed_in_rank_order = p[order] <= thresh
    mask = np.zeros(n, dtype=bool)
    if passed_in_rank_order.any():
        last = np.where(passed_in_rank_order)[0].max()
        mask[order[: last + 1]] = True
    return mask


# ---------------------------------------------------------------------
# LAMP exact min-attainable p-value (unsupervised variant)
# ---------------------------------------------------------------------

def min_attainable_p(support_count: int, n: int, n1: int = None) -> float:
    """Min p-value attainable for an itemset with this support count.

    Default row marginal is n/2 (symmetric case — the smallest possible
    min-p given no label).
    """
    if n1 is None:
        n1 = n // 2
    if support_count <= 0 or n <= 0:
        return 1.0
    a_star = min(support_count, n1)
    table = np.array([
        [a_star, support_count - a_star],
        [n1 - a_star, n - n1 - support_count + a_star]
    ], dtype=np.float64) + 0.5
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    E = row_sums * col_sums / table.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(E > 0, table / E, 1.0)
        G = 2.0 * float(np.sum(table * np.log(ratio)))
    return float(chi2_dist.sf(max(G, 0), df=1))


# ---------------------------------------------------------------------
# Roaring-bitmap TID wrapper
# ---------------------------------------------------------------------

class RoaringTID:
    __slots__ = ("bitmap", "n_total")

    def __init__(self, tid_iter: Sequence[int], n_total: int):
        self.bitmap = BitMap(tid_iter)
        self.n_total = n_total

    @classmethod
    def from_column(cls, col: np.ndarray) -> "RoaringTID":
        return cls(np.flatnonzero(col == 1).tolist(), len(col))

    def support(self) -> float:
        return len(self.bitmap) / self.n_total if self.n_total else 0.0

    def size(self) -> int:
        return len(self.bitmap)

    def __and__(self, other: "RoaringTID") -> "RoaringTID":
        out = RoaringTID([], self.n_total)
        out.bitmap = self.bitmap & other.bitmap
        return out

    def __or__(self, other: "RoaringTID") -> "RoaringTID":
        out = RoaringTID([], self.n_total)
        out.bitmap = self.bitmap | other.bitmap
        return out

    def __invert__(self) -> "RoaringTID":
        universe = BitMap(range(self.n_total))
        out = RoaringTID([], self.n_total)
        out.bitmap = universe - self.bitmap
        return out

    def __len__(self) -> int:
        return len(self.bitmap)


# ---------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------

@dataclass
class Itemset:
    items: Tuple[int, ...]
    support: float
    ig: float
    cohesion: float
    p_value: float
    min_p_value: float


@dataclass
class AssociationRule:
    antecedent: Tuple[int, ...]
    consequent: Tuple[int, ...]
    support: float
    confidence: float
    lift: float
    p_value: float
    passes_fwer: bool = False

    def to_dict(self, feature_names: Sequence[str]) -> Dict[str, Any]:
        return {
            "antecedent": [feature_names[i] for i in self.antecedent],
            "consequent": [feature_names[i] for i in self.consequent],
            "support": self.support,
            "confidence": self.confidence,
            "lift": self.lift,
            "p_value": self.p_value,
            "passes_fwer": self.passes_fwer,
        }


@dataclass
class SAMEResult:
    frequent_itemsets: Dict[int, List[Itemset]] = field(default_factory=dict)
    rules: List[AssociationRule] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    gamma_history: List[float] = field(default_factory=list)
    delta_history: List[float] = field(default_factory=list)
    cohesion_history: List[float] = field(default_factory=list)
    persistence_history: List[float] = field(default_factory=list)
    tarone_threshold: float = 0.0
    n_testable: int = 0
    performance: Dict[str, Any] = field(default_factory=dict)
    stopped_early: bool = False
    stop_reason: str = ""
    fdr_method_used: str = ""
    lamp_m_testable: int = 0

    def to_json(self, path: str, rule_cap: int = 2000) -> None:
        top = sorted(self.rules, key=lambda r: r.lift, reverse=True)[:rule_cap]
        payload = {
            "frequent_itemsets": {
                str(k): [
                    {"items": [self.feature_names[i] for i in it.items],
                     "support": it.support, "ig": it.ig, "cohesion": it.cohesion,
                     "p_value": it.p_value, "min_p_value": it.min_p_value}
                    for it in lst
                ]
                for k, lst in self.frequent_itemsets.items()
            },
            "rules_total": len(self.rules),
            "rules_fwer_passing": sum(1 for r in self.rules if r.passes_fwer),
            "rules_top": [r.to_dict(self.feature_names) for r in top],
            "gamma_history": self.gamma_history,
            "delta_history": self.delta_history,
            "cohesion_history": self.cohesion_history,
            "persistence_history": self.persistence_history,
            "tarone_threshold": self.tarone_threshold,
            "n_testable": self.n_testable,
            "lamp_m_testable": self.lamp_m_testable,
            "fdr_method_used": self.fdr_method_used,
            "performance": self.performance,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=float)


CHUNK_SIZE = 500


def is_closed(itemset_bitmap: BitMap, extension_bitmaps: List[BitMap],
              n: int) -> bool:
    """True iff no extension preserves the itemset's TID count (closure test)."""
    itemset_count = len(itemset_bitmap)
    for ext_bitmap in extension_bitmaps:
        ext_count = len(itemset_bitmap & ext_bitmap)
        if ext_count == itemset_count:
            return False
    return True


# ---------------------------------------------------------------------
# Timeout context (cross-platform; signals are POSIX-only)
# ---------------------------------------------------------------------

class _TimeoutError(Exception):
    pass


class _Timeout:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self._start = None
        self._timer = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self._timer is not None:
            self._timer.cancel()
        return False

    def check(self):
        """Call inside hot loops to enforce the deadline cooperatively."""
        if time.perf_counter() - self._start > self.seconds:
            raise _TimeoutError(
                f"Operation exceeded {self.seconds:.0f}s timeout")


# ---------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------

class SAMEv6(BaseEstimator):
    """SAME miner: entropy-driven support thresholds + Tarone-Bonferroni FWER.

    Parameters
    ----------
    alpha           : Hoeffding-margin fraction of s0 (default 0.1).
    cohesion        : one of {'phi', 'symmetric', 'ochiai', 'pmi'}.
    fwer_alpha      : target FWER (default 0.05).
    fdr_method      : {'tarone', 'wy', 'bh', None} — multiple-testing correction.
    search_mode     : {'dfs', 'opus'}.
    wy_permutations : permutations for Westfall-Young (default 200).
    opus_top_k      : K for OPUS top-K lift search.
    top_k_rules     : cap on returned rules; None keeps them all.
    minsup_floor    : lower bound on the dynamic threshold.
    persistence_threshold, patience : early-stop controls.
    max_k           : maximum itemset cardinality.
    mode            : 'all' or 'closed'.
    test_method     : 'gtest' or 'fisher'.
    seed, min_confidence, min_lift : as in v5.
    timeout_s       : per-phase wall-clock timeout (0 disables).
    auto_hyperparams: derive alpha and persistence_threshold from the data.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        cohesion: str = "phi",
        fwer_alpha: float = 0.05,
        fdr_method: str = "tarone",
        search_mode: str = "dfs",
        wy_permutations: int = 200,
        opus_top_k: int = 1000,
        top_k_rules: Optional[int] = 10_000,
        minsup_floor: float = 0.01,
        persistence_threshold: float = 0.1,
        patience: int = 2,
        max_k: int = 20,
        seed: int = 42,
        min_confidence: float = 0.0,
        min_lift: float = 0.0,
        mode: str = "all",
        test_method: str = "gtest",
        timeout_s: float = 0,
        auto_hyperparams: bool = False,
    ) -> None:
        self.alpha = alpha
        self.cohesion = cohesion
        self.fwer_alpha = fwer_alpha
        self.fdr_method = fdr_method
        self.search_mode = search_mode
        self.wy_permutations = wy_permutations
        self.opus_top_k = opus_top_k
        self.top_k_rules = top_k_rules
        self.minsup_floor = minsup_floor
        self.persistence_threshold = persistence_threshold
        self.patience = patience
        self.max_k = max_k
        self.seed = seed
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.mode = mode
        self.test_method = test_method
        self.timeout_s = timeout_s
        self.auto_hyperparams = auto_hyperparams

    # -----------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------

    def fit(self, X, feature_names=None):
        np.random.seed(self.seed)
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns) if feature_names is None else feature_names
            X = X.values
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
        X = np.asarray(X, dtype=np.int8)
        n, m = X.shape
        density = float(X.sum()) / max(n * m, 1)

        self._X = X
        self._feature_names = list(feature_names)

        s0 = self._compute_s0(m, n, density)

        if self.auto_hyperparams:
            # Pick delta equal to the Bonferroni-corrected FWER target at s0,
            # which makes alpha a function of the data rather than a user knob.
            # alpha is the Hoeffding-margin fraction: eps_k = alpha * s_bar,
            # so delta_k = exp(-2n (alpha s_bar)^2) from Hoeffding 1963.
            k_eff = min(m, self.max_k)
            m_testable = sum(math.comb(m, k) for k in range(1, k_eff + 1))
            delta_target = max(self.fwer_alpha / max(m_testable, 1), 1e-300)
            margin = math.sqrt(math.log(1.0 / delta_target) / max(2 * n, 1))
            self.alpha = max(min(margin / max(s0, 1e-9), 1.0), 1e-4)
            self.persistence_threshold = 1.0 / max(self.max_k, 1)

        tracemalloc.start()
        t0 = time.perf_counter()

        # Phase 1: per-item Roaring bitmaps and supports
        item_reps: Dict[int, RoaringTID] = {
            i: RoaringTID.from_column(X[:, i]) for i in range(m)
        }
        item_supports: Dict[int, float] = {
            i: item_reps[i].support() for i in range(m)
        }

        result = SAMEResult(feature_names=list(feature_names))
        result.performance["s0"] = s0
        result.performance["density"] = density
        result.fdr_method_used = self.fdr_method or "none"

        # Phase 2: singletons pass the dynamic k=1 threshold.
        f1 = self._phase_singletons(item_reps, item_supports, s0, n)
        result.frequent_itemsets[1] = f1

        if not f1:
            result.stopped_early = True
            result.stop_reason = "no_frequent_singletons"
        else:
            if self.search_mode == "opus":
                # OPUS needs an initial significance threshold for pruning.
                # Start from the naive Bonferroni over the upper-bound testable
                # count; LAMP exact replaces this in phase 4 below.
                k_eff = min(m, self.max_k)
                m_upper = sum(math.comb(m, k) for k in range(1, k_eff + 1))
                alpha_star_init = self.fwer_alpha / max(m_upper, 1)
                self._mine_opus_topk_lift(
                    result, item_reps, item_supports, n, alpha_star_init, X
                )
            else:
                self._mine_dfs(result, item_reps, item_supports, s0, n, f1, m)

        # Phase 4: LAMP exact testable count on the frequent set.
        self._compute_lamp_exact(result, n)

        # Phase 5: rule generation + FWER correction.
        result.rules = self._generate_rules(result, n)
        self._apply_fwer_control(result, item_reps, n, X)

        peak_trace = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop()
        result.performance["total_s"] = round(time.perf_counter() - t0, 3)
        result.performance["tracemalloc_peak_MB"] = round(peak_trace, 2)
        result.performance["rss_MB"] = round(
            psutil.Process().memory_info().rss / 2**20, 1
        )
        self.result_ = result
        return self

    def get_rules(self, only_fwer_passing: bool = False) -> List[AssociationRule]:
        if only_fwer_passing:
            return [r for r in self.result_.rules if r.passes_fwer]
        return self.result_.rules

    # -----------------------------------------------------------------
    # Phase 1 — base threshold s0
    # -----------------------------------------------------------------

    def _compute_s0(self, m: int, n: int, density: float) -> float:
        """Base support floor derived from Bonferroni-corrected chi-square."""
        k_eff = min(m, self.max_k)
        m_testable = sum(math.comb(m, k) for k in range(1, k_eff + 1))
        corrected_alpha = max(self.fwer_alpha / max(m_testable, 1), 1e-300)
        chi2_crit = chi2_dist.ppf(1 - corrected_alpha, df=1)
        s0 = math.sqrt(chi2_crit / max(n * density * (1 - density), 1e-9))
        return max(0.01, min(s0, 0.5))

    def _phase_singletons(
        self,
        item_reps: Dict[int, RoaringTID],
        item_supports: Dict[int, float],
        s0: float,
        n: int,
    ) -> List[Itemset]:
        gamma1 = -math.log2(s0) if s0 > 0 else 1.0
        delta1 = math.exp(-2 * n * (self.alpha * s0) ** 2)
        delta1 = max(min(delta1, 1.0), 1e-12)
        epsilon1 = math.sqrt(math.log(2 / delta1) / (2 * n))

        frequent: List[Itemset] = []
        for i, supp in item_supports.items():
            ig = information_gain(supp, p_bg=0.5)
            tau = s0 * (2 ** (-gamma1 * ig)) * (2 - 1) + epsilon1
            tau = max(tau, self.minsup_floor)
            if supp >= tau:
                frequent.append(Itemset(
                    items=(i,), support=supp, ig=ig, cohesion=1.0,
                    p_value=1.0, min_p_value=0.0,
                ))
        return frequent

    # -----------------------------------------------------------------
    # ECLAT-style DFS
    # -----------------------------------------------------------------

    def _mine_dfs(
        self,
        result: SAMEResult,
        item_reps: Dict[int, RoaringTID],
        item_supports: Dict[int, float],
        s0: float,
        n: int,
        f1: List[Itemset],
        m: int,
    ) -> None:
        cohesion_fn = COHESION_FUNCS[self.cohesion]
        frequent_items = sorted([it.items[0] for it in f1])
        n_freq = len(frequent_items)
        use_closed = (self.mode == "closed")
        use_gtest = (self.test_method == "gtest")

        # Webb (2007) layered Bonferroni allocation: share alpha across
        # levels in proportion to the candidate-count at each level.
        N_total_est = 0
        cand_counts_by_k: Dict[int, int] = {}
        for kk in range(2, self.max_k + 1):
            if kk > n_freq:
                break
            ck = math.comb(n_freq, kk)
            cand_counts_by_k[kk] = ck
            N_total_est += ck
        N_total_est = max(N_total_est, 1)

        gamma_base = -math.log2(s0) if s0 > 0 else 1.0

        gamma_by_k: Dict[int, float] = {}
        chi2_crit_base = None
        for kk in range(2, self.max_k + 1):
            if kk not in cand_counts_by_k:
                break
            alpha_k = self.fwer_alpha * cand_counts_by_k[kk] / N_total_est
            alpha_k = max(alpha_k, 1e-300)
            chi2_crit_k = chi2_dist.ppf(1 - alpha_k, df=1)
            if chi2_crit_base is None:
                chi2_crit_base = max(chi2_crit_k, 1e-9)
            gamma_by_k[kk] = gamma_base * (chi2_crit_k / chi2_crit_base)

        stagnation = 0
        prev_frequent_items_set = set(frequent_items)
        level_counts: Dict[int, int] = {}

        _item_supp_arr = np.array(
            [item_supports.get(i, 0.0)
             for i in range(max(item_supports.keys()) + 1)]
        )

        _timeout = _Timeout(self.timeout_s) if self.timeout_s > 0 else None
        if _timeout:
            _timeout.__enter__()
        _timed_out = False

        def recurse(prefix: Tuple[int, ...], prefix_bitmap: BitMap,
                    extensions: List[int], k: int):
            nonlocal stagnation, _timed_out

            if k > self.max_k:
                return
            if _timed_out:
                return

            gamma_k = gamma_by_k.get(k, gamma_base)

            s_bar = len(prefix_bitmap) / n if n > 0 else 0.01
            s_bar = max(s_bar, 0.01)
            delta_k = max(min(math.exp(-2 * n * (self.alpha * s_bar) ** 2), 1.0), 1e-12)
            epsilon_k = math.sqrt(math.log(2 / delta_k) / (2 * n))

            prefix_supp_prod = 1.0
            for ii in prefix:
                prefix_supp_prod *= _item_supp_arr[ii]
            prefix_min_supp = min(_item_supp_arr[ii] for ii in prefix)

            new_frequent: List[Tuple[int, BitMap, Itemset]] = []

            for chunk_start in range(0, len(extensions), CHUNK_SIZE):
                if _timed_out:
                    break
                chunk_exts = extensions[chunk_start:chunk_start + CHUNK_SIZE]
                chunk_size = len(chunk_exts)

                tables = np.zeros((chunk_size, 2, 2), dtype=np.int64)
                bitmaps = [None] * chunk_size
                supps = np.zeros(chunk_size, dtype=np.float64)
                igs = np.zeros(chunk_size, dtype=np.float64)
                cohesions = np.zeros(chunk_size, dtype=np.float64)
                taus = np.zeros(chunk_size, dtype=np.float64)
                valid_mask = np.zeros(chunk_size, dtype=bool)

                for j, ext_item in enumerate(chunk_exts):
                    if _timeout and j % 50 == 0:
                        try:
                            _timeout.check()
                        except _TimeoutError:
                            _timed_out = True
                            break

                    ext_bm = item_reps[ext_item].bitmap
                    new_bitmap = prefix_bitmap & ext_bm
                    n11 = len(new_bitmap)
                    new_supp = n11 / n

                    p_bg = prefix_supp_prod * _item_supp_arr[ext_item]
                    supp_max = min(prefix_min_supp, _item_supp_arr[ext_item])
                    ig_up = information_gain_upper_envelope(supp_max, p_bg)
                    tau_env = max(
                        s0 * (2 ** (-gamma_k * ig_up)) * (2 - 1) + epsilon_k,
                        self.minsup_floor,
                    )
                    if supp_max < tau_env:
                        continue

                    new_items = prefix + (ext_item,)
                    first_bm = item_reps[new_items[0]].bitmap
                    if len(new_items) == 2:
                        rest_bm = (ext_bm if new_items[1] == ext_item
                                   else item_reps[new_items[1]].bitmap)
                    else:
                        rest_bm = item_reps[new_items[1]].bitmap
                        for ii_idx in range(2, len(new_items)):
                            rest_bm = rest_bm & item_reps[new_items[ii_idx]].bitmap

                    n_a = len(first_bm & rest_bm)
                    n_b = len(first_bm) - n_a
                    n_c = len(rest_bm) - n_a
                    n_d = n - n_a - n_b - n_c

                    ig = information_gain(new_supp, p_bg)
                    coh = cohesion_fn(n_a, n_b, n_c, n_d)
                    tau = s0 * (2 ** (-gamma_k * ig)) * (2 - coh) + epsilon_k
                    tau = max(tau, self.minsup_floor)

                    tables[j] = [[n_a, n_b], [n_c, n_d]]
                    bitmaps[j] = new_bitmap
                    supps[j] = new_supp
                    igs[j] = ig
                    cohesions[j] = coh
                    taus[j] = tau
                    valid_mask[j] = True

                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    valid_tables = tables[valid_indices]

                    if use_gtest:
                        p_values = g_test_batch(valid_tables)
                    else:
                        p_values = np.array([
                            fisher_p(int(tables[j, 0, 0]), int(tables[j, 0, 1]),
                                     int(tables[j, 1, 0]), int(tables[j, 1, 1]))
                            for j in valid_indices
                        ])

                    n1_arr = valid_tables[:, 0, 0] + valid_tables[:, 0, 1]
                    n_1_arr = valid_tables[:, 0, 0] + valid_tables[:, 1, 0]
                    min_p_values = minimum_achievable_p_batch(
                        n1_arr.astype(np.int64),
                        n_1_arr.astype(np.int64), n
                    )

                    for idx_v, orig_j in enumerate(valid_indices):
                        orig_j = int(orig_j)
                        new_supp_j = supps[orig_j]
                        tau_j = taus[orig_j]
                        if new_supp_j >= tau_j:
                            ext_item = chunk_exts[orig_j]
                            new_items = prefix + (ext_item,)
                            itemset_obj = Itemset(
                                items=tuple(sorted(new_items)),
                                support=float(new_supp_j),
                                ig=float(igs[orig_j]),
                                cohesion=float(cohesions[orig_j]),
                                p_value=float(p_values[idx_v]),
                                min_p_value=float(min_p_values[idx_v]),
                            )
                            new_frequent.append(
                                (ext_item, bitmaps[orig_j], itemset_obj))

                del tables, bitmaps

            if use_closed and new_frequent:
                closed_frequent: List[Tuple[int, BitMap, Itemset]] = []
                all_ext_bitmaps = [bm for _, bm, _ in new_frequent]
                for idx, (item_j, bitmap_j, iset) in enumerate(new_frequent):
                    other_bitmaps = (all_ext_bitmaps[:idx]
                                     + all_ext_bitmaps[idx+1:])
                    if is_closed(bitmap_j, other_bitmaps, n):
                        closed_frequent.append((item_j, bitmap_j, iset))
                stored = closed_frequent
            else:
                stored = new_frequent

            for _, _, iset in stored:
                result.frequent_itemsets.setdefault(k, []).append(iset)

            level_counts[k] = level_counts.get(k, 0) + len(stored)

            for j_idx in range(len(new_frequent)):
                item_j, bitmap_j, _ = new_frequent[j_idx]
                remaining_items = [
                    itm for idx2 in range(j_idx + 1, len(new_frequent))
                    for itm in [new_frequent[idx2][0]]
                ]
                if remaining_items and k < self.max_k:
                    recurse(prefix + (item_j,), bitmap_j, remaining_items,
                            k + 1)

        try:
            for i_idx, item in enumerate(frequent_items):
                if _timed_out:
                    break
                remaining = frequent_items[i_idx + 1:]
                if remaining:
                    recurse((item,), item_reps[item].bitmap, remaining, 2)
        except _TimeoutError:
            _timed_out = True

        if _timeout:
            _timeout.__exit__(None, None, None)

        if _timed_out:
            result.stopped_early = True
            result.stop_reason = "dfs_timeout"

        for kk in sorted(level_counts.keys()):
            gamma_k = gamma_by_k.get(kk, gamma_base)
            result.gamma_history.append(gamma_k)

            level_itemsets = result.frequent_itemsets.get(kk, [])
            if level_itemsets:
                avg_coh = float(np.mean([it.cohesion for it in level_itemsets]))
            else:
                avg_coh = 0.0
            result.cohesion_history.append(avg_coh)

            s_bar = (float(np.mean([it.support for it in level_itemsets]))
                     if level_itemsets else 0.01)
            delta_k = max(min(
                math.exp(-2 * n * (self.alpha * s_bar) ** 2), 1.0), 1e-12)
            result.delta_history.append(delta_k)

            cur_items = set()
            for it in level_itemsets:
                cur_items.update(it.items)
            persistence = (
                len(cur_items & prev_frequent_items_set) /
                len(cur_items | prev_frequent_items_set)
                if (cur_items | prev_frequent_items_set) else 0.0
            )
            result.persistence_history.append(persistence)

            if not level_itemsets:
                result.stop_reason = "no_frequent_at_level_k"
                result.stopped_early = True
                break
            if persistence < self.persistence_threshold:
                stagnation += 1
            else:
                stagnation = 0
            if stagnation >= self.patience:
                result.stop_reason = (
                    f"persistence<{self.persistence_threshold}")
                result.stopped_early = True
                break

            prev_frequent_items_set = cur_items

    # -----------------------------------------------------------------
    # LAMP exact testable count
    # -----------------------------------------------------------------

    def _compute_lamp_exact(self, result: SAMEResult, n: int) -> None:
        """LAMP testable count over the mined frequent set (Terada 2013)."""
        all_min_ps = []
        for k, itemsets in result.frequent_itemsets.items():
            if k < 2:
                continue
            for it in itemsets:
                sup_count = max(int(round(it.support * n)), 1)
                mp = min_attainable_p(sup_count, n)
                all_min_ps.append(mp)

        if not all_min_ps:
            result.lamp_m_testable = 0
            return

        sorted_ps = sorted(all_min_ps)
        m_testable = 0
        for k in range(1, len(sorted_ps) + 1):
            if sorted_ps[k - 1] <= self.fwer_alpha / k:
                m_testable = k

        result.lamp_m_testable = m_testable
        result.performance["lamp_m_testable"] = m_testable
        result.performance["lamp_alpha_star"] = (
            self.fwer_alpha / max(m_testable, 1))

    # -----------------------------------------------------------------
    # Westfall-Young column permutation
    # -----------------------------------------------------------------

    def _westfall_young(
        self,
        X: np.ndarray,
        item_reps: Dict[int, RoaringTID],
        result: SAMEResult,
        n: int,
        J: int = 200,
    ) -> float:
        """WY-corrected significance threshold via column-permutation FDR.

        Each permutation: shuffle every column independently, re-evaluate
        every frequent itemset of size >= 2, and record the smallest
        p-value. The WY threshold is the floor(alpha * J)-th smallest
        min-p across permutations.
        """
        all_itemsets = []
        for k, lst in result.frequent_itemsets.items():
            if k < 2:
                continue
            for it in lst:
                all_itemsets.append(it.items)

        if not all_itemsets:
            return self.fwer_alpha

        min_pvals = []
        rng = np.random.default_rng(self.seed)
        m_cols = X.shape[1]

        max_itemsets_per_perm = min(len(all_itemsets), 5000)
        if len(all_itemsets) > max_itemsets_per_perm:
            sample_idx = rng.choice(len(all_itemsets), max_itemsets_per_perm,
                                    replace=False)
            sampled_itemsets = [all_itemsets[i] for i in sample_idx]
        else:
            sampled_itemsets = all_itemsets

        timeout = _Timeout(self.timeout_s) if self.timeout_s > 0 else None
        if timeout:
            timeout.__enter__()

        try:
            for j in range(J):
                if timeout:
                    timeout.check()

                X_perm = X.copy()
                for col in range(m_cols):
                    rng.shuffle(X_perm[:, col])

                perm_reps = {}
                for i in range(m_cols):
                    perm_reps[i] = BitMap(
                        np.flatnonzero(X_perm[:, i] == 1).tolist())

                min_p = 1.0
                for items in sampled_itemsets:
                    if len(items) < 2:
                        continue
                    perm_bitmap = perm_reps[items[0]].copy()
                    for idx in items[1:]:
                        perm_bitmap &= perm_reps[idx]
                    perm_sup = len(perm_bitmap)

                    first_bm = perm_reps[items[0]]
                    if len(items) == 2:
                        rest_bm = perm_reps[items[1]]
                    else:
                        rest_bm = perm_reps[items[1]].copy()
                        for ii in items[2:]:
                            rest_bm &= perm_reps[ii]

                    n_a = len(first_bm & rest_bm)
                    n_b = len(first_bm) - n_a
                    n_c = len(rest_bm) - n_a
                    n_d = n - n_a - n_b - n_c

                    table = np.array([[n_a, n_b], [n_c, n_d]],
                                     dtype=np.float64)
                    p = g_test_single_table(table)
                    if p < min_p:
                        min_p = p

                min_pvals.append(min_p)
        except _TimeoutError:
            pass
        finally:
            if timeout:
                timeout.__exit__(None, None, None)

        if not min_pvals:
            return self.fwer_alpha

        min_pvals.sort()
        idx = max(0, int(self.fwer_alpha * len(min_pvals)) - 1)
        return min_pvals[idx] if idx < len(min_pvals) else self.fwer_alpha

    # -----------------------------------------------------------------
    # OPUS best-first top-K lift
    # -----------------------------------------------------------------

    def _mine_opus_topk_lift(
        self,
        result: SAMEResult,
        item_reps: Dict[int, RoaringTID],
        item_supports: Dict[int, float],
        n: int,
        alpha_star: float,
        X: np.ndarray,
    ) -> None:
        """Best-first branch-and-bound for top-K itemsets by lift.

        Items are sorted by ascending support (rarest first) so the lift
        upper bound used for pruning is tightest earliest.
        """
        top_k = self.opus_top_k
        cohesion_fn = COHESION_FUNCS[self.cohesion]

        freq_items = sorted(
            [it.items[0] for it in result.frequent_itemsets.get(1, [])],
            key=lambda i: item_supports[i]
        )

        heap: list = []
        counter = [0]

        _timeout = _Timeout(self.timeout_s) if self.timeout_s > 0 else None
        if _timeout:
            _timeout.__enter__()
        _timed_out = False

        def search(current_items: Tuple[int, ...],
                   current_bitmap: BitMap,
                   available: List[int],
                   current_prod_sup: float):
            nonlocal _timed_out

            if _timed_out:
                return

            for i, e in enumerate(available):
                if _timed_out:
                    return

                if _timeout and counter[0] % 200 == 0:
                    try:
                        _timeout.check()
                    except _TimeoutError:
                        _timed_out = True
                        return
                counter[0] += 1

                ext_bitmap = current_bitmap & item_reps[e].bitmap
                ext_sup_count = len(ext_bitmap)
                if ext_sup_count == 0:
                    continue

                ext_sup = ext_sup_count / n
                ext_items = current_items + (e,)
                ext_prod = current_prod_sup * item_supports[e]
                ext_lift = ext_sup / ext_prod if ext_prod > 0 else 0

                min_lift_in_heap = (
                    -heap[0][0] if len(heap) >= top_k else 0)

                if ext_lift > min_lift_in_heap and len(ext_items) >= 2:
                    sorted_items = tuple(sorted(ext_items))
                    first_bm = item_reps[sorted_items[0]].bitmap
                    if len(sorted_items) == 2:
                        rest_bm = item_reps[sorted_items[1]].bitmap
                    else:
                        rest_bm = item_reps[sorted_items[1]].bitmap
                        for ii in sorted_items[2:]:
                            rest_bm = rest_bm & item_reps[ii].bitmap

                    n_a = len(first_bm & rest_bm)
                    n_b = len(first_bm) - n_a
                    n_c = len(rest_bm) - n_a
                    n_d = n - n_a - n_b - n_c

                    table = np.array([[n_a, n_b], [n_c, n_d]],
                                     dtype=np.float64)
                    p_val = g_test_single_table(table)
                    ig = information_gain(ext_sup, ext_prod)
                    coh = cohesion_fn(n_a, n_b, n_c, n_d)

                    n1_ = n_a + n_b
                    n_1 = n_a + n_c
                    mp = minimum_achievable_p(n1_, n_1, n)

                    if p_val <= alpha_star:
                        entry = (-ext_lift, counter[0],
                                 sorted_items, p_val, ext_sup,
                                 ig, coh, mp)
                        if len(heap) < top_k:
                            heapq.heappush(heap, entry)
                        else:
                            heapq.heappushpop(heap, entry)

                # Prune: upper bound on lift reachable by any further extension.
                remaining = available[i + 1:]
                if remaining and len(ext_items) < self.max_k:
                    min_remaining_sup = min(
                        item_supports[r] for r in remaining)
                    ub_denom = ext_prod * min_remaining_sup
                    ub_lift = (ext_sup / ub_denom
                               if ub_denom > 0 else 0)
                    new_min_lift = (
                        -heap[0][0] if len(heap) >= top_k else 0)
                    if ub_lift > new_min_lift:
                        search(ext_items, ext_bitmap, remaining, ext_prod)

        try:
            for i, item in enumerate(freq_items):
                if _timed_out:
                    break
                remaining = freq_items[i + 1:]
                search(
                    (item,),
                    item_reps[item].bitmap,
                    remaining,
                    item_supports[item],
                )
        except _TimeoutError:
            _timed_out = True

        if _timeout:
            _timeout.__exit__(None, None, None)

        if _timed_out:
            result.stopped_early = True
            result.stop_reason = "opus_timeout"

        for neg_lift, _, items, p_val, sup, ig, coh, mp in heap:
            k = len(items)
            result.frequent_itemsets.setdefault(k, []).append(
                Itemset(
                    items=items,
                    support=sup,
                    ig=ig,
                    cohesion=coh,
                    p_value=p_val,
                    min_p_value=mp,
                )
            )

    # -----------------------------------------------------------------
    # Rule generation
    # -----------------------------------------------------------------

    def _generate_rules(self, result: SAMEResult,
                        n: int) -> List[AssociationRule]:
        supports: Dict[Tuple[int, ...], float] = {}
        for lst in result.frequent_itemsets.values():
            for it in lst:
                supports[it.items] = it.support

        top_k = self.top_k_rules or 0
        use_heap = top_k > 0
        heap: List[Tuple[float, int, Any]] = []
        rule_counter = 0
        rules_list: List[AssociationRule] = []
        min_conf = self.min_confidence
        min_lift_val = self.min_lift
        _sup_get = supports.get

        all_itemsets = []
        for k, lst in result.frequent_itemsets.items():
            if k < 2:
                continue
            for it in lst:
                all_itemsets.append(it)
        all_itemsets.sort(key=lambda x: x.support)

        for it in all_itemsets:
            items = it.items
            it_supp = it.support
            it_pval = it.p_value
            k_items = len(items)

            if use_heap and len(heap) >= top_k:
                max_possible_lift = 1.0 / max(it_supp, 1e-12)
                if max_possible_lift <= heap[0][0]:
                    continue

            heap_min_lift = (heap[0][0]
                            if (use_heap and len(heap) >= top_k) else 0.0)

            if k_items <= 5:
                total_masks = (1 << k_items) - 1
                masks = range(1, total_masks)
            else:
                masks = [1 << b for b in range(k_items)]

            for mask in masks:
                ant = tuple(items[b] for b in range(k_items)
                            if mask & (1 << b))
                ant_s = _sup_get(ant)
                if not ant_s:
                    continue
                conf = it_supp / ant_s
                if conf < min_conf:
                    continue
                cons = tuple(items[b] for b in range(k_items)
                             if not (mask & (1 << b)))
                cons_s = _sup_get(cons)
                if not cons_s:
                    continue
                lift = conf / cons_s
                if lift < min_lift_val:
                    continue
                if use_heap:
                    if len(heap) < top_k:
                        heapq.heappush(heap, (
                            lift, rule_counter,
                            ant, cons, it_supp, conf, lift, it_pval))
                        rule_counter += 1
                    elif lift > heap_min_lift:
                        heapq.heapreplace(heap, (
                            lift, rule_counter,
                            ant, cons, it_supp, conf, lift, it_pval))
                        rule_counter += 1
                        heap_min_lift = heap[0][0]
                else:
                    rules_list.append(AssociationRule(
                        antecedent=ant, consequent=cons,
                        support=it_supp, confidence=conf, lift=lift,
                        p_value=it_pval, passes_fwer=False,
                    ))

        if use_heap:
            return [
                AssociationRule(
                    antecedent=ant, consequent=cons,
                    support=supp, confidence=conf, lift=lift,
                    p_value=pval, passes_fwer=False,
                )
                for _, _, ant, cons, supp, conf, lift, pval
                in sorted(heap, key=lambda x: -x[0])
            ]
        return rules_list

    # -----------------------------------------------------------------
    # FWER control (Tarone / Westfall-Young / Benjamini-Hochberg)
    # -----------------------------------------------------------------

    def _apply_fwer_control(
        self,
        result: SAMEResult,
        item_reps: Dict[int, RoaringTID],
        n: int,
        X: np.ndarray,
    ) -> None:
        if self.fdr_method is None or not result.rules:
            return

        if self.fdr_method == "tarone":
            if result.lamp_m_testable > 0:
                thr = self.fwer_alpha / result.lamp_m_testable
                n_test = result.lamp_m_testable
            else:
                min_ps = [
                    it.min_p_value
                    for lst in result.frequent_itemsets.values()
                    for it in lst if len(it.items) >= 2
                ]
                thr, n_test = tarone_threshold(min_ps, self.fwer_alpha)
            result.tarone_threshold = thr
            result.n_testable = n_test
            for r in result.rules:
                r.passes_fwer = r.p_value <= thr

        elif self.fdr_method == "wy":
            wy_thr = self._westfall_young(
                X, item_reps, result, n,
                J=self.wy_permutations,
            )
            result.tarone_threshold = wy_thr
            result.n_testable = sum(
                1 for lst in result.frequent_itemsets.values()
                for it in lst if len(it.items) >= 2
            )
            for r in result.rules:
                r.passes_fwer = r.p_value <= wy_thr

        elif self.fdr_method == "bh":
            pvals = [r.p_value for r in result.rules]
            mask = benjamini_hochberg(pvals, self.fwer_alpha)
            for r, keep in zip(result.rules, mask):
                r.passes_fwer = bool(keep)

        result.performance["n_rules_total"] = len(result.rules)
        result.performance["n_rules_fwer_passing"] = sum(
            1 for r in result.rules if r.passes_fwer
        )

    # -----------------------------------------------------------------
    # partial_fit — Hoeffding-triggered incremental update
    # -----------------------------------------------------------------

    def partial_fit(self, X_new, feature_names=None):
        """Append new transactions; re-mine only when Hoeffding says we must.

        A full re-mine triggers when any existing itemset's support sits
        inside an epsilon-band around the refreshed s0: that is the regime
        where the concentration inequality can no longer rule out a threshold
        crossing.
        """
        if isinstance(X_new, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X_new.columns)
            X_new = X_new.values
        X_new = np.asarray(X_new, dtype=np.int8)

        if not hasattr(self, '_X_accumulated'):
            self._X_accumulated = X_new.copy()
            return self.fit(X_new, feature_names)

        X_old = self._X_accumulated
        self._X_accumulated = np.vstack([X_old, X_new])
        n_old = X_old.shape[0]
        n_new = self._X_accumulated.shape[0]

        epsilon_new = math.sqrt(math.log(2 / 0.01) / (2 * n_new))

        needs_remine = False
        if hasattr(self, 'result_') and self.result_.frequent_itemsets:
            density = float(self._X_accumulated.sum()) / max(
                n_new * self._X_accumulated.shape[1], 1)
            s0_new = self._compute_s0(
                self._X_accumulated.shape[1], n_new, density)

            for k, itemsets in self.result_.frequent_itemsets.items():
                if k < 2:
                    continue
                for it in itemsets:
                    new_count = 0
                    for row_idx in range(X_new.shape[0]):
                        if all(X_new[row_idx, i] == 1 for i in it.items):
                            new_count += 1
                    old_count = int(round(it.support * n_old))
                    updated_sup = (old_count + new_count) / n_new

                    if abs(updated_sup - s0_new) < epsilon_new:
                        needs_remine = True
                        break
                if needs_remine:
                    break
        else:
            needs_remine = True

        if needs_remine:
            fn = (feature_names or self._feature_names
                  if hasattr(self, '_feature_names')
                  else [f"x{i}"
                        for i in range(self._X_accumulated.shape[1])])
            return self.fit(self._X_accumulated, fn)
        else:
            # No re-mine: just refresh support counts on the retained itemsets.
            if hasattr(self, 'result_'):
                for k, itemsets in self.result_.frequent_itemsets.items():
                    for it in itemsets:
                        new_count = 0
                        for row_idx in range(X_new.shape[0]):
                            if all(X_new[row_idx, i] == 1 for i in it.items):
                                new_count += 1
                        old_count = int(round(it.support * n_old))
                        it.support = (old_count + new_count) / n_new
            return self

    # -----------------------------------------------------------------
    # Persistence helper kept for v5 compatibility
    # -----------------------------------------------------------------

    @staticmethod
    def _persistence(cur: List[Itemset], prev: List[Itemset]) -> float:
        if not cur or not prev:
            return 0.0
        a = set().union(*[it.items for it in cur])
        b = set().union(*[it.items for it in prev])
        return len(a & b) / len(a | b) if (a | b) else 0.0


# ---------------------------------------------------------------------
# CLI / benchmark harness
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import csv
    import gc
    import traceback

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "datasets")
    OUT_DIR = os.path.join(BASE_DIR, "experiments", "out")
    os.makedirs(OUT_DIR, exist_ok=True)

    datasets = [
        ("abide", 10),
        ("eeg_eye_state", 7),
        ("mushroom", 3),
        ("clinvar_sample", 6),
    ]

    configs = [
        {"name": "v6_dfs_tarone",
         "search_mode": "dfs", "fdr_method": "tarone"},
        {"name": "v6_dfs_wy",
         "search_mode": "dfs", "fdr_method": "wy", "wy_permutations": 100},
        {"name": "v6_opus_tarone",
         "search_mode": "opus", "fdr_method": "tarone"},
        {"name": "v6_opus_wy",
         "search_mode": "opus", "fdr_method": "wy", "wy_permutations": 100},
    ]

    TIMEOUT_PER_RUN = 300

    rows = []
    header = [
        "config", "dataset", "max_k", "time_s", "mem_MB",
        "n_itemsets", "n_rules", "n_fwer", "max_lift", "p90_lift",
        "lamp_m_testable", "tarone_thr", "status",
    ]

    print("=" * 80)
    print("SAME v6 Benchmark")
    print("=" * 80)

    for ds_name, max_k in datasets:
        csv_path = os.path.join(DATA_DIR, f"{ds_name}.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {ds_name}: file not found at {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        X = df.values.astype(np.int8)
        feature_names = list(df.columns)
        print(f"\nDataset: {ds_name}  ({X.shape[0]} x {X.shape[1]})  max_k={max_k}")

        for cfg in configs:
            cfg_name = cfg["name"]
            gc.collect()
            print(f"  Config: {cfg_name} ... ", end="", flush=True)

            status = "ok"
            try:
                est = SAMEv6(
                    max_k=max_k,
                    search_mode=cfg["search_mode"],
                    fdr_method=cfg["fdr_method"],
                    wy_permutations=cfg.get("wy_permutations", 200),
                    opus_top_k=1000,
                    top_k_rules=10_000,
                    timeout_s=TIMEOUT_PER_RUN,
                )
                est.fit(X, feature_names=feature_names)
                r = est.result_

                n_itemsets = sum(len(v) for v in r.frequent_itemsets.values())
                n_rules = len(r.rules)
                n_fwer = sum(1 for x in r.rules if x.passes_fwer)
                lifts = [x.lift for x in r.rules] if r.rules else [0.0]
                max_lift = max(lifts)
                p90_lift = float(np.percentile(lifts, 90)) if lifts else 0.0
                time_s = r.performance.get("total_s", 0)
                mem_MB = r.performance.get("tracemalloc_peak_MB", 0)
                lamp_m = r.lamp_m_testable
                tarone_thr = r.tarone_threshold

                if r.stopped_early and "timeout" in r.stop_reason:
                    status = f"timeout({r.stop_reason})"

                row = [
                    cfg_name, ds_name, max_k,
                    round(time_s, 3), round(mem_MB, 2),
                    n_itemsets, n_rules, n_fwer,
                    round(max_lift, 4), round(p90_lift, 4),
                    lamp_m, f"{tarone_thr:.2e}", status,
                ]
                rows.append(row)

                print(f"{time_s:.1f}s  {n_itemsets} itemsets  "
                      f"{n_rules} rules  {n_fwer} fwer  "
                      f"lift_max={max_lift:.3f}  p90={p90_lift:.3f}  "
                      f"LAMP_m={lamp_m}")

            except Exception as e:
                tb = traceback.format_exc()
                print(f"ERROR: {e}")
                print(tb[:500])
                status = f"error: {str(e)[:80]}"
                row = [
                    cfg_name, ds_name, max_k,
                    0, 0, 0, 0, 0, 0, 0, 0, "N/A", status,
                ]
                rows.append(row)

    print("\n" + "=" * 80)
    print("partial_fit smoke test on abide")
    print("=" * 80)

    abide_path = os.path.join(DATA_DIR, "abide.csv")
    if os.path.exists(abide_path):
        df_abide = pd.read_csv(abide_path)
        X_abide = df_abide.values.astype(np.int8)
        fn_abide = list(df_abide.columns)

        half = X_abide.shape[0] // 2
        X1, X2 = X_abide[:half], X_abide[half:]

        est_inc = SAMEv6(max_k=5, timeout_s=60)
        print(f"  partial_fit(X1) [{X1.shape[0]} rows] ... ", end="",
              flush=True)
        est_inc.partial_fit(X1, fn_abide)
        r1 = est_inc.result_
        n1_items = sum(len(v) for v in r1.frequent_itemsets.values())
        print(f"done. {n1_items} itemsets, {len(r1.rules)} rules")

        print(f"  partial_fit(X2) [{X2.shape[0]} rows] ... ", end="",
              flush=True)
        est_inc.partial_fit(X2, fn_abide)
        r2 = est_inc.result_
        n2_items = sum(len(v) for v in r2.frequent_itemsets.values())
        print(f"done. {n2_items} itemsets, {len(r2.rules)} rules")

        row = [
            "v6_partial_fit", "abide", 5,
            round(r2.performance.get("total_s", 0), 3),
            round(r2.performance.get("tracemalloc_peak_MB", 0), 2),
            n2_items, len(r2.rules),
            sum(1 for x in r2.rules if x.passes_fwer),
            round(max((x.lift for x in r2.rules), default=0), 4),
            round(float(np.percentile(
                [x.lift for x in r2.rules], 90)) if r2.rules else 0, 4),
            r2.lamp_m_testable,
            f"{r2.tarone_threshold:.2e}",
            "ok",
        ]
        rows.append(row)

    out_path = os.path.join(OUT_DIR, "same_v6_comparison.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nResults written to {out_path}")
    print("=" * 80)
    print("Done.")
