"""
Validation of the bias on a REAL follow graph with known ground truth.

Source: data/bsky.db (andrewconner/bluesky_profiles, Bluesky snapshot of 2023-04-23),
table `follows` (sourceDid -> targetDid). This is a real Bluesky graph from the beta
period (~35k nodes, 2.1M edges), used here as a POPULATION whose true Gini is known, to
check whether the bias of snowball plus densification measured on synthetic structure
reproduces on real structure.

  - TRUE in-degree Gini over the complete graph.
  - Snowball plus densification at several fractions f = N/N_pop -> sampled Gini.
  - The sampled-against-f curve on real data, compared with the synthetic prediction.

Output: results/validate_fullgraph.json, results/fig_realvalidation.png

Usage:
  ./venv/bin/python validate_fullgraph.py --reps 5
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results"
DB = Path(__file__).resolve().parent.parent / "data" / "bsky.db"

plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 10,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})


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


def snowball_densify(G, N, K, s, D, rng):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    rng = random.Random(0)

    print("Loading the real follow graph...")
    con = sqlite3.connect(str(DB))
    G = nx.DiGraph()
    G.add_edges_from(con.execute("SELECT sourceDid, targetDid FROM follows"))
    npop = G.number_of_nodes()
    indeg = np.array([d for _, d in G.in_degree()])
    true = {"n": npop, "m": G.number_of_edges(), "mean_degree": round(G.number_of_edges() / npop, 1),
            "gini_indegree": round(gini(indeg), 4), "top1": round(top_share(indeg), 4)}
    print(f"  real population: n={true['n']}, m={true['m']}, mean degree={true['mean_degree']}")
    print(f"  TRUE in-degree Gini = {true['gini_indegree']}  (top1%={true['top1']})")

    # Snowball at several fractions.
    print("\nSnowball + densification at several f:")
    rows = []
    for N in [1000, 2000, 3000, 6000, 12000, 24000]:
        if N >= npop:
            continue
        vals = [snowball_densify(G, N, 1000, 10, 4, rng) for _ in range(args.reps)]
        f = N / npop
        rows.append({"N": N, "f": round(f, 4), "sampled_gini": round(float(np.mean(vals)), 4),
                     "sd": round(float(np.std(vals, ddof=1)), 4), "bias": round(float(np.mean(vals)) - true["gini_indegree"], 4)})
        print(f"  N={N:6d} f={f*100:5.1f}% sampled={np.mean(vals):.3f}±{np.std(vals,ddof=1):.3f} "
              f"bias={np.mean(vals)-true['gini_indegree']:+.3f}")

    report = {"source": "andrewconner/bluesky_profiles (2023-04-23 snapshot)",
              "true_population": true, "sampled_by_fraction": rows}
    OUT.mkdir(exist_ok=True)
    (OUT / "validate_fullgraph.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Figure: sampled against f on real data, with the ground truth.
    fs = np.array([r["f"] for r in rows]) * 100
    ss = np.array([r["sampled_gini"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.errorbar(fs, ss, yerr=[r["sd"] for r in rows], fmt="o-", capsize=3, color="#1f77b4",
                label="snowball + densification (real graph)")
    ax.axhline(true["gini_indegree"], color="grey", ls="--", lw=1,
               label=f"true Gini = {true['gini_indegree']:.2f}")
    import matplotlib.ticker as mticker
    ax.set_xscale("log"); ax.set_xlabel("sampled fraction $f = N/N_{pop}$ (\\%)")
    ax.set_xticks(fs); ax.get_xaxis().set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel("in-degree Gini"); ax.set_ylim(0, max(0.8, true["gini_indegree"] + 0.05))
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_realvalidation.png"); plt.close(fig)
    print(f"\n-> {OUT/'validate_fullgraph.json'}, fig_realvalidation.png")


if __name__ == "__main__":
    main()
