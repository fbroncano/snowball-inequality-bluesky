"""
The like-for-like null comparison.

The problem: the in-degree Gini is measured on a subgraph built by snowball sampling
plus densification, which over-represents hubs. Comparing that sampled Gini against
Erdos-Renyi or preferential-attachment nulls generated FROM SCRATCH is not a valid
comparison. The correct test applies THE SAME sampling procedure to a synthetic
population graph of known mechanism, and compares subgraph-Gini against subgraph-Gini.

Procedure:
  1. Generate a large directed "population" graph of known mechanism:
     - directed scale-free (directed preferential attachment), and
     - directed Erdos-Renyi (pure chance).
  2. Measure the POPULATION in-degree Gini (the ground truth).
  3. Apply the same snowball (random seeds, expansion along follow edges, cap K, up to N
     nodes) plus densification (induced subgraph) and measure the SUBGRAPH Gini.

What this shows:
  - How much the sampling INFLATES the Gini (subgraph against population) under each
    mechanism.
  - Whether the empirical Bluesky Gini exceeds the subgraph Gini of the sampled
    preferential-attachment null, which is the only comparison that can support a claim
    of excess concentration over rich-get-richer growth.

Output: results/decisive_null.json

Usage:
  ./venv/bin/python decisive_null.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results"
EMPIRICAL_GINI = 0.616  # mean in-degree Gini over the 5 real replications


def gini(x):
    x = np.sort(np.asarray(x, dtype=float)); x = x[x >= 0]
    if x.sum() == 0:
        return float("nan")
    n = len(x); idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def population(model, pop, rng_seed):
    """Large directed graph. A->B = A follows B."""
    if model == "scale_free":  # directed preferential attachment (Bollobas et al.)
        g = nx.scale_free_graph(pop, seed=rng_seed)
        return nx.DiGraph(g)  # colapsa multigrafo a simple
    if model == "er":
        m = pop * 15  # densidad comparable, grado medio ~15
        return nx.gnm_random_graph(pop, m, directed=True, seed=rng_seed)
    raise ValueError(model)


def snowball_densify(G, N, K, s, rng):
    """The same procedure as the real pipeline, applied to the population graph G."""
    nodes = list(G.nodes())
    seeds = rng.sample(nodes, min(s, len(nodes)))
    visited, queue, queued = set(), deque((x, 0) for x in seeds), set(seeds)
    while queue and len(visited) < N:
        u, d = queue.popleft()
        visited.add(u)
        if d >= 4:
            continue
        outs = list(G.successors(u))  # whom u follows (outgoing follow edges)
        if len(outs) > K:
            outs = rng.sample(outs, K)
        for v in outs:
            if v not in queued and len(visited) + len(queue) < N * 3:
                queue.append((v, d + 1)); queued.add(v)
    sub = G.subgraph(visited)  # densification = the complete induced subgraph
    return gini(np.array([dg for _, dg in sub.in_degree()])), sub.number_of_nodes(), sub.number_of_edges()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=50000)
    ap.add_argument("--N", type=int, default=3000)
    ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    rng = random.Random(42)

    report = {"params": vars(args), "empirical_subgraph_gini": EMPIRICAL_GINI, "models": {}}
    print(f"Empirical (Bluesky, subgraph): Gini(in) = {EMPIRICAL_GINI}\n")
    for model in ["scale_free", "er"]:
        G = population(model, args.pop, 7)
        pop_gini = gini(np.array([d for _, d in G.in_degree()]))
        subs = [snowball_densify(G, args.N, args.K, args.seeds, rng) for _ in range(args.reps)]
        g_sub = [s[0] for s in subs]
        report["models"][model] = {
            "population_gini": round(pop_gini, 4),
            "sampled_subgraph_gini_mean": round(float(np.mean(g_sub)), 4),
            "sampled_subgraph_gini_sd": round(float(np.std(g_sub, ddof=1)), 4),
            "inflation": round(float(np.mean(g_sub)) - pop_gini, 4),
            "sub_nodes_mean": int(np.mean([s[1] for s in subs])),
            "sub_edges_mean": int(np.mean([s[2] for s in subs])),
        }
        m = report["models"][model]
        print(f"[{model}]")
        print(f"  population  Gini(in)        = {m['population_gini']:.3f}")
        print(f"  subgraph    Gini(in) sampled = {m['sampled_subgraph_gini_mean']:.3f} "
              f"+/- {m['sampled_subgraph_gini_sd']:.3f}  (inflation {m['inflation']:+.3f})")
        print(f"  subgraph    n={m['sub_nodes_mean']} m={m['sub_edges_mean']}\n")

    sf = report["models"]["scale_free"]["sampled_subgraph_gini_mean"]
    verdict = ("the excess SURVIVES: empirical > sampled-PA"
               if EMPIRICAL_GINI > sf + 0.03 else
               "WARNING: empirical ~ sampled-PA -> the excess may be a sampling artefact")
    report["verdict"] = verdict
    print(f"VERDICT: {verdict}")
    OUT.mkdir(exist_ok=True)
    (OUT / "decisive_null.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"-> {OUT / 'decisive_null.json'}")


if __name__ == "__main__":
    main()
