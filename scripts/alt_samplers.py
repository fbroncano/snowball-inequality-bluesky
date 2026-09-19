"""
Alternative samplers: is the problem the traversal, or the estimand?

Compares THREE sampling designs over the same synthetic populations:
  - snowball : BFS along follow edges, with cap K (Algorithm 1 of the paper)
  - mhrw     : Metropolis-Hastings random walk over the undirected view
               (Gjoka et al.), whose stationary distribution is uniform over nodes
  - uniform  : uniform node sampling (the ideal, unreachable without a census)

and for each it measures TWO different estimands:
  - gini_induced : Gini of the in-degree INSIDE the induced subgraph
                   (what a crawl conventionally reports)
  - gini_true    : Gini of the true population degrees of the sampled nodes
                   (what a per-node counter gives)

The number of walk steps is also recorded, as a proxy for the requests a real crawl
would have to make.

Output: results/alt_samplers.json

Usage:
  ./venv/bin/python alt_samplers.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np

from bias_full import gini, top_share
from bias_study import lognormal_population, reference_population

OUT = Path(__file__).resolve().parent.parent / "results"


def undirected_adj(G) -> dict:
    """Undirected view: u knows v if u follows v or v follows u.
    This is what a crawler can reconstruct with getFollows + getFollowers."""
    adj = {}
    for u in G.nodes():
        nb = set(G.successors(u)) | set(G.predecessors(u))
        nb.discard(u)
        adj[u] = tuple(nb)
    return adj


def sample_snowball(G, N, K, s, D, rng):
    nodes = list(G.nodes())
    seeds = rng.sample(nodes, min(s, len(nodes)))
    visited, queue, queued = set(), deque((x, 0) for x in seeds), set(seeds)
    cost = 0
    while queue and len(visited) < N:
        u, d = queue.popleft(); visited.add(u); cost += 1
        if d >= D:
            continue
        outs = list(G.successors(u))
        if len(outs) > K:
            outs = rng.sample(outs, K)
        for v in outs:
            if v not in queued and len(visited) + len(queue) < N * 3:
                queue.append((v, d + 1)); queued.add(v)
    return visited, cost


def sample_mhrw(G, N, rng, adj):
    """Metropolis-Hastings random walk, stationary uniform over nodes.
    `cost` counts walk steps, that is API requests, not distinct nodes."""
    nodes = list(G.nodes())
    cur = rng.choice(nodes)
    visited = {cur}
    cost = 0
    max_steps = N * 300
    while len(visited) < N and cost < max_steps:
        cost += 1
        nb = adj[cur]
        if not nb:                       # nodo aislado: reinicio
            cur = rng.choice(nodes); visited.add(cur); continue
        v = nb[rng.randrange(len(nb))]
        dv = len(adj[v])
        if dv and rng.random() < min(1.0, len(nb) / dv):
            cur = v
        visited.add(cur)
    return visited, cost


def sample_uniform(G, N, rng):
    nodes = list(G.nodes())
    v = set(rng.sample(nodes, min(N, len(nodes))))
    return v, len(v)


def measure(G, true_indeg, visited):
    sub = G.subgraph(visited)
    ind = np.array([d for _, d in sub.in_degree()])
    td = np.array([true_indeg[u] for u in visited])
    return {
        "n": int(len(visited)),
        "m_induced": int(sub.number_of_edges()),
        "gini_induced": round(gini(ind), 4),
        "top1_induced": round(top_share(ind), 4) if ind.sum() else None,
        "gini_truedeg": round(gini(td), 4),
        "top1_truedeg": round(top_share(td), 4) if td.sum() else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=40000)
    ap.add_argument("--N", type=int, default=3000)
    ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=4)
    args = ap.parse_args()
    # The same seeds and the same generation order as bias_study.py, so that the
    # populations are IDENTICAL to those of Table 1 and the true Gini values agree
    # exactly across the two tables.
    rng = random.Random(42); np_rng = np.random.default_rng(42)

    pops = [("lognormal", s) for s in (0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0)] + [("scalefree", None)]
    rows = []
    for kind, sigma in pops:
        G = lognormal_population(args.pop, sigma, 15, np_rng) if kind == "lognormal" \
            else reference_population("scale_free", args.pop, 7)
        true_indeg = dict(G.in_degree())
        tg = gini(np.array(list(true_indeg.values())))
        tt = top_share(np.array(list(true_indeg.values())))
        adj = undirected_adj(G)
        row = {"pop": kind, "sigma": sigma, "true_gini": round(tg, 4),
               "true_top1": round(tt, 4), "f": round(args.N / args.pop, 5),
               "samplers": {}}
        for name in ("snowball", "mhrw", "uniform"):
            reps = []
            for _ in range(args.reps):
                if name == "snowball":
                    v, c = sample_snowball(G, args.N, args.K, 10, 4, rng)
                elif name == "mhrw":
                    v, c = sample_mhrw(G, args.N, rng, adj)
                else:
                    v, c = sample_uniform(G, args.N, rng)
                m = measure(G, true_indeg, v); m["cost"] = c
                reps.append(m)
            agg = {k: round(float(np.mean([r[k] for r in reps if r[k] is not None])), 4)
                   for k in ("n", "m_induced", "gini_induced", "gini_truedeg",
                             "top1_induced", "top1_truedeg", "cost")
                   if any(r[k] is not None for r in reps)}
            row["samplers"][name] = agg
            print(f"  {kind}{'' if sigma is None else f' s={sigma}'} "
                  f"(true {tg:.3f}) {name:9}: induced={agg.get('gini_induced')} "
                  f"truedeg={agg.get('gini_truedeg')} m={agg.get('m_induced'):.0f} "
                  f"cost={agg.get('cost'):.0f}", flush=True)
        rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "alt_samplers.json").write_text(json.dumps(
        {"pop": args.pop, "N": args.N, "K": args.K, "reps": args.reps, "rows": rows}, indent=2))
    print(f"\n[saved] {OUT / 'alt_samplers.json'}")


if __name__ == "__main__":
    main()
