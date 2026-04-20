"""same-fim: Similarity-Adaptive Monotonic Entropy frequent-itemset mining.

See README.md or the companion paper (Necir & Benarab, 2026) for the method
and evaluation. The public API exposes SAME (the miner) and a handful of
helper functions used by the paper's baselines.

Example:
    >>> from same_fim import SAME
    >>> import pandas as pd
    >>> df = pd.read_csv("mydata.csv").astype("int8")
    >>> est = SAME(auto_hyperparams=True, search_mode="dfs", max_k=5)
    >>> est.fit(df.values, feature_names=list(df.columns))
    >>> for r in est.result_.rules[:10]:
    ...     if r.passes_fwer:
    ...         print(r)
"""
from __future__ import annotations

from .core import (
    SAMEv6 as SAME,
    Itemset, AssociationRule, SAMEResult, RoaringTID,
    monotone_entropy, information_gain, cohesion_phi, cohesion_symmetric,
    cohesion_ochiai, cohesion_pmi, g_test_batch, fisher_p,
    tarone_threshold, benjamini_hochberg, min_attainable_p,
)

Rule = AssociationRule  # public alias

__version__ = "0.7.2"

__all__ = [
    "SAME", "SAMEResult", "Itemset", "AssociationRule", "Rule",
    "RoaringTID",
    "monotone_entropy", "information_gain",
    "cohesion_phi", "cohesion_symmetric", "cohesion_ochiai", "cohesion_pmi",
    "g_test_batch", "fisher_p",
    "tarone_threshold", "benjamini_hochberg", "min_attainable_p",
    "__version__",
]
