"""
Additional robustness checks.

  B3: the gap between the PRESCRIBED degree sequence (lognormal) and the REALISED one,
      after collapsing parallel edges and removing self-loops in the configuration model.
  B2: sensitivity of the bias to the out-degree MODEL (a permutation of the in-degree,
      against an independent Poisson sequence, against an independent lognormal one).
  A4: tail fit (Clauset) on the near-unbiased Bluesky follower counters, to characterise
      the family (power law against lognormal).

Output: results/extra_checks.json

Usage:
  ./venv/bin/python extra_checks.py
"""

from __future__ import annotations

import json
import random
import warnings
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"


def gini(x):
    x = np.sort(np.asarray(x, dtype=float)); x = x[x >= 0]
    if x.sum() == 0:
        return float("nan")
    n = len(x); idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def sample_gini(G, N, K, s, D, rng):
    nodes = list(G.nodes()); seeds = rng.sample(nodes, min(s, len(nodes)))
    visited, queue, queued = set(), deque((x, 0) for x in seeds), set(seeds)
    while queue and len(visited) < N:
        u, d = queue.popleft(); visited.add(u)
        if d >= D:
            continue
        outs = list(G.successors(u))
        if len(outs) > K:
            outs = rng.sample(outs, K)
        for v in outs:
            if v not in queued and len(visited) + len(queue) < N * 3:
                queue.append((v, d + 1)); queued.add(v)
    return gini(np.array([dg for _, dg in G.subgraph(visited).in_degree()]))


def build(indeg, outdeg, np_rng):
    d = int(indeg.sum() - outdeg.sum())
    if d > 0:
        outdeg[:d] += 1
    elif d < 0:
        indeg[:(-d)] += 1
    g = nx.DiGraph(nx.directed_configuration_model(indeg.tolist(), outdeg.tolist(),
                                                   seed=int(np_rng.integers(1e9))))
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def main():
    rng = random.Random(0); np_rng = np.random.default_rng(0)
    pop, sigma, mean_deg = 40000, 1.4, 15
    mu = np.log(mean_deg) - sigma ** 2 / 2
    report = {}

    # --- B3: prescribed against realised ---
    indeg = np.maximum(1, np.round(np_rng.lognormal(mu, sigma, pop)).astype(int))
    outdeg = np_rng.permutation(indeg).copy()
    G = build(indeg.copy(), outdeg.copy(), np_rng)
    realized_in = np.array([d for _, d in G.in_degree()])
    report["B3_prescribed_vs_realized"] = {
        "prescribed_gini": round(gini(indeg), 4),
        "realized_gini": round(gini(realized_in), 4),
        "prescribed_max": int(indeg.max()), "realized_max": int(realized_in.max()),
    }
    print("B3 prescribed vs realised:", report["B3_prescribed_vs_realized"])

    # --- B2: sensitivity to the out-degree model ---
    print("\nB2 out-degree sensitivity (in-degree lognormal held fixed):")
    res = {}
    for name, outd in [
        ("permutation_of_in", np_rng.permutation(indeg).copy()),
        ("independent_poisson", np_rng.poisson(mean_deg, pop).clip(min=1)),
        ("independent_lognormal", np.maximum(1, np.round(np_rng.lognormal(mu, sigma, pop)).astype(int))),
    ]:
        Gx = build(indeg.copy(), outd.astype(int).copy(), np_rng)
        tg = gini(np.array([d for _, d in Gx.in_degree()]))
        sv = [sample_gini(Gx, 3000, 1000, 10, 4, rng) for _ in range(6)]
        res[name] = {"true": round(tg, 4), "sampled": round(float(np.mean(sv)), 4),
                     "sd": round(float(np.std(sv, ddof=1)), 4)}
        print(f"  {name:22s} true={tg:.3f} sampled={np.mean(sv):.3f}±{np.std(sv,ddof=1):.3f}")
    report["B2_outdegree_sensitivity"] = res

    # --- A4: tail test on the global follower counts (near-unbiased) ---
    import powerlaw
    nodes = pd.concat([pd.read_parquet(p) for p in (ROOT / "runs").glob("run_*/data/nodes.parquet")],
                      ignore_index=True).drop_duplicates(subset="did")
    fc = nodes["followers_count"].dropna().to_numpy()
    fc = fc[fc > 0]
    fit = powerlaw.Fit(fc, discrete=True, verbose=False)
    R, p = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
    report["A4_tail_fit_followers"] = {
        "n": int(len(fc)), "alpha": round(float(fit.alpha), 3), "xmin": float(fit.xmin),
        "powerlaw_vs_lognormal_R": round(float(R), 3), "p": round(float(p), 4),
        "verdict": ("lognormal preferred" if R < 0 else "power-law preferred") if p < 0.05 else "inconclusive",
    }
    print("\nA4 tail fit (followers):", report["A4_tail_fit_followers"])

    OUT.mkdir(exist_ok=True)
    (OUT / "extra_checks.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n-> {OUT/'extra_checks.json'}")


if __name__ == "__main__":
    main()
