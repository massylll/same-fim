"""
Benchmark on 5 domain-appropriate datasets:

  Neuroimaging / neuroscience:   abide, eeg_eye_state, synth_neuro
  Population genomics:           clinvar_sample
  Protein analysis:              pfam_proteins

Runs:
  - SAME (DFS + Tarone)
  - SAME (OPUS + Tarone)
  - mlxtend Apriori (sigma sweep)
  - mlxtend FP-Growth (sigma sweep)
  - PAMI FPGrowth, ECLAT, CHARM (sigma sweep)
  - scikit-mine SLIM (parameter-free)

LAMP is not runnable here: the C binary compiles, but the Python driver
returns None after the call due to a Windows / Python 2-to-3 issue
(noted in the paper).
"""
from __future__ import annotations
import sys, time, tempfile, os, traceback, tracemalloc
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent

OUT = ROOT / "experiments" / "out"
OUT.mkdir(exist_ok=True, parents=True)

DATASETS = [
    ("abide",          "neuroimaging",      10),
    ("eeg_eye_state",  "neuroscience",       7),
    ("synth_neuro",    "neuroimaging (synth)", 5),
    ("clinvar_sample", "pop. genomics",      6),
    ("pfam_proteins",  "protein analysis",   4),
]
SIGMAS = [0.50, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05]

PROCESS_TIMEOUT = 120


def measure(fn, *args, **kw):
    """Run fn; return {time_s, peak_mem_MB, ok, err, **result}."""
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        r = fn(*args, **kw)
        ok = True; err = ""
    except Exception as e:
        r = {}; ok = False; err = f"{type(e).__name__}: {str(e)[:100]}"
    t = round(time.perf_counter() - t0, 3)
    peak = round(tracemalloc.get_traced_memory()[1] / 2**20, 2)
    tracemalloc.stop()
    out = {"time_s": t, "peak_mem_MB": peak, "ok": ok, "error": err}
    if ok and isinstance(r, dict):
        out.update(r)
    return out


def csv_to_pami(csv_path, out_path):
    df = pd.read_csv(csv_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            items = [str(col) for col, val in row.items() if val == 1]
            f.write("\t".join(items) + "\n")


def run_mlxtend_apriori(df_bool, sigma):
    from mlxtend.frequent_patterns import apriori, association_rules
    fis = apriori(df_bool, min_support=sigma, use_colnames=True, low_memory=True)
    n = len(fis)
    n_rules = 0; max_lift = 0.0
    if n > 0:
        try:
            r = association_rules(fis, metric="confidence", min_threshold=0.0)
            n_rules = len(r)
            max_lift = float(r["lift"].max()) if n_rules else 0.0
        except Exception:
            pass
    return {"n_itemsets": n, "n_rules": n_rules, "max_lift": max_lift, "n_rules_fwer": 0}


def run_mlxtend_fpgrowth(df_bool, sigma):
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    fis = fpgrowth(df_bool, min_support=sigma, use_colnames=True)
    n = len(fis)
    n_rules = 0; max_lift = 0.0
    if n > 0:
        try:
            r = association_rules(fis, metric="confidence", min_threshold=0.0)
            n_rules = len(r)
            max_lift = float(r["lift"].max()) if n_rules else 0.0
        except Exception:
            pass
    return {"n_itemsets": n, "n_rules": n_rules, "max_lift": max_lift, "n_rules_fwer": 0}


def run_pami(algo, pami_file, sigma):
    if algo == "fpgrowth":
        from PAMI.frequentPattern.basic import FPGrowth as A
        obj = A.FPGrowth(iFile=pami_file, minSup=sigma, sep='\t')
    elif algo == "eclat":
        from PAMI.frequentPattern.basic import ECLAT as A
        obj = A.ECLAT(iFile=pami_file, minSup=sigma, sep='\t')
    elif algo == "charm":
        from PAMI.frequentPattern.closed import CHARM as A
        obj = A.CHARM(iFile=pami_file, minSup=sigma, sep='\t')
    obj.mine()
    n = len(obj.getPatterns())
    return {"n_itemsets": n, "n_rules": 0, "max_lift": 0.0, "n_rules_fwer": 0}


def run_slim(csv_path):
    from skmine.itemsets import SLIM
    df = pd.read_csv(csv_path)
    transactions = []
    for _, row in df.iterrows():
        items = frozenset(str(c) for c, v in row.items() if v == 1)
        if items:
            transactions.append(items)
    slim = SLIM(pruning=True, max_time=60)
    if not hasattr(slim, "_validate_data"):
        slim._validate_data = lambda X, **kw: X
    slim.fit(transactions)
    n = len(getattr(slim, "codetable_", [])) or 0
    return {"n_itemsets": n, "n_rules": 0, "max_lift": 0.0, "n_rules_fwer": 0}


def run_same_v6(X, fn, mk, mode):
    from same_fim import SAME as SAMEv6
    est = SAMEv6(max_k=mk, seed=42, mode="all", test_method="gtest",
                 top_k_rules=10_000, fwer_alpha=0.05,
                 search_mode=mode, fdr_method="tarone", opus_top_k=1000)
    est.fit(X, feature_names=fn)
    r = est.result_
    n_it = sum(len(v) for v in r.frequent_itemsets.values())
    lifts = [rr.lift for rr in r.rules]
    return {
        "n_itemsets": n_it,
        "n_rules": len(r.rules),
        "max_lift": float(max(lifts, default=0)),
        "n_rules_fwer": sum(1 for rr in r.rules if rr.passes_fwer),
    }


def main():
    rows = []
    for ds_name, domain, max_k in DATASETS:
        csv = ROOT / "datasets" / f"{ds_name}.csv"
        if not csv.exists():
            print(f"SKIP {ds_name}: file missing")
            continue
        df = pd.read_csv(csv).astype(np.int8)
        X = df.values; fn = list(df.columns)
        n, m = X.shape
        density = X.mean()
        print(f"\n--- {ds_name} ({domain})  n={n}  m={m}  density={density:.3f}  max_k={max_k}", flush=True)
        # checkpoint intermediate results
        if rows:
            pd.DataFrame(rows).to_csv(OUT / "domain_benchmark_final.csv", index=False)

        pami_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False).name
        csv_to_pami(csv, pami_file)

        # SAME
        for mode in ("dfs", "opus"):
            r = measure(run_same_v6, X, fn, max_k, mode)
            row = {"dataset": ds_name, "domain": domain,
                   "method": f"SAME_v6_{mode}", "param_name": "none", "param_value": None,
                   **r}
            rows.append(row)
            print(f"  SAME_{mode}: t={r['time_s']}s mem={r['peak_mem_MB']}MB "
                  f"items={r.get('n_itemsets',0)} rules={r.get('n_rules',0)} "
                  f"fwer={r.get('n_rules_fwer',0)} lift={r.get('max_lift',0):.2f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "domain_benchmark_final.csv", index=False)

        # mlxtend + PAMI at sigma sweep
        df_bool = pd.DataFrame(df.values.astype(bool), columns=fn)
        for sigma in SIGMAS:
            if m > 50 and sigma < 0.15:
                continue
            # apriori
            r = measure(run_mlxtend_apriori, df_bool, sigma)
            rows.append({"dataset": ds_name, "domain": domain, "method": "mlxtend_apriori",
                          "param_name": "sigma", "param_value": sigma, **r})
            # fpgrowth
            r = measure(run_mlxtend_fpgrowth, df_bool, sigma)
            rows.append({"dataset": ds_name, "domain": domain, "method": "mlxtend_fpgrowth",
                          "param_name": "sigma", "param_value": sigma, **r})
            # pami fpg
            r = measure(run_pami, "fpgrowth", pami_file, sigma)
            rows.append({"dataset": ds_name, "domain": domain, "method": "pami_fpgrowth",
                          "param_name": "sigma", "param_value": sigma, **r})
            # pami eclat
            r = measure(run_pami, "eclat", pami_file, sigma)
            rows.append({"dataset": ds_name, "domain": domain, "method": "pami_eclat",
                          "param_name": "sigma", "param_value": sigma, **r})
            # pami charm
            r = measure(run_pami, "charm", pami_file, sigma)
            rows.append({"dataset": ds_name, "domain": domain, "method": "pami_charm",
                          "param_name": "sigma", "param_value": sigma, **r})
            print(f"  sigma={sigma}: done", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "domain_benchmark_final.csv", index=False)

        # SLIM (skip on very large/dense)
        if ds_name not in ("eeg_eye_state", "synth_neuro", "pfam_proteins"):
            r = measure(run_slim, csv)
            rows.append({"dataset": ds_name, "domain": domain, "method": "slim",
                          "param_name": "none", "param_value": None, **r})
            print(f"  slim: t={r['time_s']}s  items={r.get('n_itemsets',0)}")

        try: os.unlink(pami_file)
        except: pass

    df = pd.DataFrame(rows)
    out_csv = OUT / "domain_benchmark_final.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
