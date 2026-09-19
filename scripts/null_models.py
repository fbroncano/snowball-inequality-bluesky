"""
Comparison against NULL MODELS with uncertainty, over the five replications.

For each replication the in-degree Gini of the empirical graph is compared against two
nulls of the same n and m: Erdos-Renyi G(n,m) (the trivial baseline) and Barabasi-Albert
(preferential attachment, the informative contrast). R realisations of each null give a
mean, a standard deviation and a z-score for the empirical-null gap. The five
replications are then aggregated.

These are from-scratch nulls. The paper's point is precisely that this comparison is
invalid; the like-for-like version is in decisive_null.py.

Output: results/null_models_agg.json, plus a table on stdout.

Usage:
  ./venv/bin/python null_models.py --reps 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"


def gini(x):
    x = np.sort(np.asarray(x, dtype=float)); x = x[x >= 0]
    if x.sum() == 0:
        return float("nan")
    n = len(x); idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def top_share(x, frac=0.01):
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    k = max(1, int(len(x) * frac))
    return float(x[:k].sum() / x.sum())


def per_run(edges_path, reps, rng):
    edges = pd.read_parquet(edges_path)
    G = nx.from_pandas_edgelist(edges, "src", "dst", create_using=nx.DiGraph)
    n, m = G.number_of_nodes(), G.number_of_edges()
    emp_gini = gini(np.array([d for _, d in G.in_degree()]))
    emp_top1 = top_share(np.array([d for _, d in G.in_degree()]))

    er = [gini(np.array([d for _, d in
          nx.gnm_random_graph(n, m, directed=True, seed=int(rng.integers(1e9))).in_degree()]))
          for _ in range(reps)]
    mm = max(1, round(m / n))
    ba = [gini(np.array([d for _, d in
          nx.barabasi_albert_graph(n, mm, seed=int(rng.integers(1e9))).degree()]))
          for _ in range(reps)]

    def z(emp, vals):
        sd = np.std(vals, ddof=1)
        return float((emp - np.mean(vals)) / sd) if sd > 0 else float("inf")

    return {
        "n": n, "m": m,
        "empirical_gini": round(emp_gini, 4), "empirical_top1": round(emp_top1, 4),
        "er_mean": round(float(np.mean(er)), 4), "er_sd": round(float(np.std(er, ddof=1)), 4),
        "ba_mean": round(float(np.mean(ba)), 4), "ba_sd": round(float(np.std(ba, ddof=1)), 4),
        "z_vs_er": round(z(emp_gini, er), 1), "z_vs_ba": round(z(emp_gini, ba), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30, help="Realisations per null and per replication")
    args = ap.parse_args()
    rng = np.random.default_rng(42)

    runs = sorted((ROOT / "runs").glob("run_*/data/edges.parquet"))
    rows = []
    print(f"{'run':>4} {'n':>5} {'m':>7} {'emp':>6} {'ER(m±sd)':>14} {'BA(m±sd)':>14} {'z_BA':>7}")
    for ep in runs:
        r = per_run(ep, args.reps, rng)
        r["run"] = ep.parent.parent.name
        rows.append(r)
        print(f"{r['run'][-1]:>4} {r['n']:>5} {r['m']:>7} {r['empirical_gini']:>6.3f} "
              f"{r['er_mean']:>6.3f}±{r['er_sd']:<5.3f} {r['ba_mean']:>6.3f}±{r['ba_sd']:<5.3f} "
              f"{r['z_vs_ba']:>7.1f}")

    def agg(key):
        v = [r[key] for r in rows]
        return round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4)

    summary = {
        "n_runs": len(rows), "reps_per_null": args.reps,
        "empirical_gini": agg("empirical_gini"),
        "er_gini": agg("er_mean"), "ba_gini": agg("ba_mean"),
        "z_vs_er_mean": round(float(np.mean([r["z_vs_er"] for r in rows])), 1),
        "z_vs_ba_mean": round(float(np.mean([r["z_vs_ba"] for r in rows])), 1),
        "per_run": rows,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "null_models_agg.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    eg, es = summary["empirical_gini"]; bg, bs = summary["ba_gini"]; rg, rs = summary["er_gini"]
    print(f"\nAGGREGATE ({len(rows)} replications):")
    print(f"  empirical Gini(in) = {eg:.3f} +/- {es:.3f}")
    print(f"  BA        Gini     = {bg:.3f} +/- {bs:.3f}   (mean z = {summary['z_vs_ba_mean']})")
    print(f"  ER        Gini     = {rg:.3f} +/- {rs:.3f}   (mean z = {summary['z_vs_er_mean']})")
    print(f"  -> {OUT / 'null_models_agg.json'}")


if __name__ == "__main__":
    main()
