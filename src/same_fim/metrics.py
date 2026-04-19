"""
Statistical utilities for method comparison.

Provides:
    - Cliff's delta effect size (Cliff 1993)
    - Bonferroni/Holm corrections
    - Friedman + Nemenyi post-hoc for critical-difference diagrams
      (Demsar 2006)
    - Pareto-front dominance test
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import friedmanchisquare, rankdata


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta; positive means a tends to be greater than b."""
    a = np.asarray(a); b = np.asarray(b)
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0
    more = np.sum(a[:, None] > b[None, :])
    less = np.sum(a[:, None] < b[None, :])
    return (more - less) / (n_a * n_b)


def cliff_magnitude(d: float) -> str:
    d = abs(d)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def friedman_test(score_matrix: np.ndarray) -> Tuple[float, float]:
    """Run Friedman test on a (datasets x methods) score matrix.
    Returns (statistic, p_value)."""
    columns = [score_matrix[:, j] for j in range(score_matrix.shape[1])]
    stat, p = friedmanchisquare(*columns)
    return float(stat), float(p)


def nemenyi_cd(n_methods: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Critical-difference threshold for Nemenyi post-hoc test.
    Uses the Studentized range for alpha=0.05, 0.10 (tabulated).
    For non-listed n_methods we interpolate linearly."""
    # Table of q values at alpha=0.05, 0.10 from Demsar 2006 / standard tables
    q_table = {
        0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
               12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391},
        0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589,
               7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920, 11: 2.978,
               12: 3.030, 13: 3.077, 14: 3.120, 15: 3.159},
    }
    q = q_table[alpha].get(n_methods, q_table[alpha][min(n_methods, 15)])
    return q * math.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))


def nemenyi_posthoc(score_matrix: np.ndarray,
                    method_names: Sequence[str],
                    alpha: float = 0.05) -> Dict[str, float]:
    """Return average ranks per method; smaller = better (if scores are 'lower is better').
    Here we assume higher = better, so we rank descending: best gets rank 1.
    """
    # Rank each row (dataset) in descending order (best = 1)
    ranks = np.zeros_like(score_matrix)
    for i in range(score_matrix.shape[0]):
        ranks[i] = rankdata(-score_matrix[i], method="average")
    avg_ranks = ranks.mean(axis=0)
    return {m: float(r) for m, r in zip(method_names, avg_ranks)}


def pareto_dominates(a: Tuple[float, ...], b: Tuple[float, ...],
                     higher_is_better: Sequence[bool]) -> bool:
    """Does point a Pareto-dominate point b?"""
    strict_better = False
    for ai, bi, hib in zip(a, b, higher_is_better):
        if hib:
            if ai < bi:
                return False
            if ai > bi:
                strict_better = True
        else:
            if ai > bi:
                return False
            if ai < bi:
                strict_better = True
    return strict_better


def pareto_front(points: Sequence[Tuple[float, ...]],
                 higher_is_better: Sequence[bool]) -> List[int]:
    """Indices of Pareto-efficient points."""
    n = len(points)
    non_dominated: List[int] = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i != j and pareto_dominates(q, p, higher_is_better):
                dominated = True
                break
        if not dominated:
            non_dominated.append(i)
    return non_dominated
