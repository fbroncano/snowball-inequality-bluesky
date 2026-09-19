"""
Scaling the bias experiment to large populations (N up to 100k) with an efficient
numpy/CSR engine (no networkx), in order to:
  (1) confirm scale invariance at fixed f up to N=100k, and
  (2) extend the bias-against-f curve to smaller fractions, with populations of up to
      several million nodes.

Engine: directed configuration model by stub matching; adjacency in CSR format; snowball
BFS over the CSR; in-degree by bincount; Gini over the in-degree array.

Output: results/scale_up.json

Usage:
  ./venv/bin/python scale_up.py
"""

from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results"


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64)); x = x[x >= 0]
    s = x.sum()
    if s == 0:
        return float("nan")
    n = len(x); idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * s) - (n + 1) / n)


def build_population_csr(pop, sigma, mean_deg, rng):
    """Directed configuration model -> CSR of successors (whom each node follows)."""
    mu = np.log(mean_deg) - sigma ** 2 / 2
    indeg = np.maximum(1, np.round(rng.lognormal(mu, sigma, pop)).astype(np.int64))
    outdeg = rng.permutation(indeg)
    # equalise the stub sums
    diff = int(indeg.sum() - outdeg.sum())
    if diff > 0:
        outdeg[:diff] += 1
    elif diff < 0:
        indeg[:(-diff)] += 1
    m = int(outdeg.sum())
    src = np.repeat(np.arange(pop, dtype=np.int64), outdeg)            # out-stubs
    dst = np.repeat(np.arange(pop, dtype=np.int64), indeg)             # in-stubs
    rng.shuffle(dst)
    # CSR ordered by source (src is already ordered by construction of repeat)
    indptr = np.zeros(pop + 1, dtype=np.int64)
    np.add.at(indptr, np.arange(1, pop + 1), outdeg)  # cumulativo
    indptr = np.cumsum(np.concatenate([[0], outdeg]))
    true_in = np.bincount(dst, minlength=pop)
    return indptr, dst, true_in, m


def successors(indptr, dst, u):
    return dst[indptr[u]:indptr[u + 1]]


def snowball_gini(indptr, dst, pop, N, K, s, D, rng):
    seeds = rng.sample(range(pop), min(s, pop))
    visited = set(); queue = deque((x, 0) for x in seeds); queued = set(seeds)
    while queue and len(visited) < N:
        u, d = queue.popleft(); visited.add(u)
        if d >= D:
            continue
        outs = successors(indptr, dst, u)
        if len(outs) > K:
            outs = rng.sample(list(outs), K)
        for v in outs:
            v = int(v)
            if v not in queued and len(visited) + len(queue) < N * 3:
                queue.append((v, d + 1)); queued.add(v)
    # in-degree within the induced subgraph
    vis = np.fromiter(visited, dtype=np.int64)
    vset = visited
    indeg = np.zeros(len(vis), dtype=np.int64)
    pos = {int(n): i for i, n in enumerate(vis)}
    for u in vis:
        for v in successors(indptr, dst, int(u)):
            v = int(v)
            if v in vset:
                indeg[pos[v]] += 1
    return gini(indeg)


def main():
    rng = np.random.default_rng(0); prng = random.Random(0)
    report = {"scale_invariance_f5pct": [], "f_extension_largescale": []}
    sigma = 1.4

    # (1) scale invariance at f=5% up to N=100k
    print("=== scale invariance (f=5%) up to N=100k ===")
    for N, pop in [(3000, 60000), (20000, 400000), (50000, 1000000), (100000, 2000000)]:
        indptr, dst, true_in, m = build_population_csr(pop, sigma, 15, rng)
        tg = gini(true_in)
        sv = [snowball_gini(indptr, dst, pop, N, 1000, 10, 4, prng) for _ in range(3)]
        row = {"N": N, "pop": pop, "f": round(N / pop, 4), "true": round(tg, 4),
               "sampled": round(float(np.mean(sv)), 4), "sd": round(float(np.std(sv, ddof=1)), 4)}
        report["scale_invariance_f5pct"].append(row)
        print(f"  N={N:6d} pop={pop:8d} f=5% true={tg:.3f} sampled={np.mean(sv):.3f}±{np.std(sv,ddof=1):.3f}", flush=True)

    # (2) extend bias-against-f to small f with N=100k (large populations)
    print("\n=== bias at small f with N=100k ===")
    for N, pop in [(100000, 4000000), (100000, 10000000)]:
        indptr, dst, true_in, m = build_population_csr(pop, sigma, 15, rng)
        tg = gini(true_in)
        sv = [snowball_gini(indptr, dst, pop, N, 1000, 10, 4, prng) for _ in range(2)]
        row = {"N": N, "pop": pop, "f": round(N / pop, 5), "true": round(tg, 4),
               "sampled": round(float(np.mean(sv)), 4)}
        report["f_extension_largescale"].append(row)
        print(f"  N={N} pop={pop} f={N/pop*100:.3f}% true={tg:.3f} sampled={np.mean(sv):.3f}", flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "scale_up.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n-> {OUT/'scale_up.json'}")


if __name__ == "__main__":
    main()
