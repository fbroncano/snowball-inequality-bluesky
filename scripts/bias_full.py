"""
Extended experiments on the sampling bias.

  (1) Bias in SEVERAL concentration measures: the in-degree Gini, the top-1% share and
      the PageRank Gini, as a function of the true Gini. Shows the phenomenon is general.
  (2) Dependence of the bias on the sampling DESIGN: depth D, number of seeds s and
      sampled fraction (N), on a fixed population of medium-to-high concentration.
  (3) CALIBRATION with uncertainty: bootstrap inversion of the bias curve, mapping the
      Gini observed on Bluesky to a true Gini with an interval. Note that the paper
      reports this inversion as unsound to extrapolate across fractions; it is kept here
      because that argument rests on it.

Output: results/bias_full.json, results/fig_bias_metrics.png, results/fig_design.png

Usage:
  ./venv/bin/python bias_full.py --pop 40000 --reps 4
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results"
EMP_GINI, EMP_SD = 0.616, 0.051

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 200,
})
C1, C2, C3 = "#1f77b4", "#d62728", "#2ca02c"


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


def lognormal_pop(pop, sigma, mean_deg, np_rng):
    mu = np.log(mean_deg) - sigma ** 2 / 2
    indeg = np.maximum(1, np.round(np_rng.lognormal(mu, sigma, pop)).astype(int))
    outdeg = np_rng.permutation(indeg)
    d = indeg.sum() - outdeg.sum()
    if d != 0:
        outdeg[0] = max(1, outdeg[0] + d)
    g = nx.DiGraph(nx.directed_configuration_model(indeg.tolist(), outdeg.tolist(),
                                                   seed=int(np_rng.integers(1e9))))
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def ref_pop(model, pop, seed):
    if model == "er":
        return nx.gnm_random_graph(pop, pop * 15, directed=True, seed=seed)
    return nx.DiGraph(nx.scale_free_graph(pop, seed=seed))


def sample_subgraph(G, N, K, s, D, rng):
    nodes = list(G.nodes())
    seeds = rng.sample(nodes, min(s, len(nodes)))
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
    return G.subgraph(visited)


def metrics(sub):
    indeg = np.array([d for _, d in sub.in_degree()])
    pr = np.array(list(nx.pagerank(sub, alpha=0.85).values())) if sub.number_of_edges() else np.array([0.0])
    return gini(indeg), top_share(indeg), gini(pr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=40000)
    ap.add_argument("--N", type=int, default=3000)
    ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=4)
    args = ap.parse_args()
    rng = random.Random(42); np_rng = np.random.default_rng(42)
    report = {}

    # (1) Multi-metric bias curve --------------------------------------------------------
    print("=== (1) Multi-metric bias ===")
    sigmas = [0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0]
    rows = []
    for sg in sigmas:
        G = lognormal_pop(args.pop, sg, 15, np_rng)
        tg, tt, tp = metrics(G)  # the "true" values, on the population
        samp = [metrics(sample_subgraph(G, args.N, args.K, 10, 4, rng)) for _ in range(args.reps)]
        sg_g, sg_t, sg_p = (np.mean([s[i] for s in samp]) for i in range(3))
        rows.append({"true_gini": round(tg, 4), "true_top1": round(tt, 4), "true_prgini": round(tp, 4),
                     "samp_gini": round(float(sg_g), 4), "samp_top1": round(float(sg_t), 4),
                     "samp_prgini": round(float(sg_p), 4)})
        print(f"  trueG={tg:.3f}: Gini {tg:.3f}->{sg_g:.3f} | top1 {tt:.3f}->{sg_t:.3f} | PRGini {tp:.3f}->{sg_p:.3f}")
    report["multi_metric"] = rows

    # (2) Design sensitivity (fixed population, true Gini ~0.67) --------------------------
    print("\n=== (2) Design dependence ===")
    Gd = lognormal_pop(args.pop, 1.4, 15, np_rng)
    base_true = gini(np.array([d for _, d in Gd.in_degree()]))

    def samp_gini(N, K, s, D):
        v = [gini(np.array([d for _, d in sample_subgraph(Gd, N, K, s, D, rng).in_degree()]))
             for _ in range(args.reps)]
        return round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4)

    design = {"true_gini": round(base_true, 4), "depth": {}, "seeds": {}, "fraction": {}}
    for D in [2, 3, 4, 5]:
        design["depth"][D] = samp_gini(args.N, args.K, 10, D)
    for s in [1, 5, 10, 20, 50]:
        design["seeds"][s] = samp_gini(args.N, args.K, s, 4)
    for N in [1000, 2000, 4000, 8000]:
        design["fraction"][N] = samp_gini(N, args.K, 10, 4)
    report["design"] = design
    print(f"  true={base_true:.3f}; depth={ {k:v[0] for k,v in design['depth'].items()} }")
    print(f"  seeds={ {k:v[0] for k,v in design['seeds'].items()} }")
    print(f"  fraction(N)={ {k:v[0] for k,v in design['fraction'].items()} }")

    # (3) Calibration with bootstrap -----------------------------------------------------
    ts = np.array([r["true_gini"] for r in rows]); ss = np.array([r["samp_gini"] for r in rows])
    order = np.argsort(ss); ss_s, ts_s = ss[order], ts[order]
    boot = []
    for _ in range(2000):
        obs = np.random.normal(EMP_GINI, EMP_SD)
        if ss_s.min() <= obs <= ss_s.max():
            boot.append(float(np.interp(obs, ss_s, ts_s)))
    boot = np.array(boot)
    calib = {"observed": EMP_GINI, "true_median": round(float(np.median(boot)), 3),
             "true_ci95": [round(float(np.percentile(boot, 2.5)), 3),
                           round(float(np.percentile(boot, 97.5)), 3)],
             "note": "lower bound: estimator saturates above true~0.8"}
    report["calibration"] = calib
    print(f"\n=== (3) Calibration ===\n  observed {EMP_GINI} -> true {calib['true_median']} "
          f"IC95 {calib['true_ci95']}")

    OUT.mkdir(exist_ok=True)
    (OUT / "bias_full.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # --- Figure 1: multi-metric ---
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="identity")
    ax.plot(ts, ss, "o-", color=C1, label="Gini in-degree")
    ax.plot([r["true_top1"] for r in rows], [r["samp_top1"] for r in rows], "s-", color=C2, label="top-1\\% share")
    ax.plot([r["true_prgini"] for r in rows], [r["samp_prgini"] for r in rows], "^-", color=C3, label="Gini PageRank")
    ax.set_xlabel("true population value"); ax.set_ylabel("sampled subgraph value")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig_bias_metrics.png"); plt.close(fig)

    # --- Figure 2: design ---
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, key, xlab in [(ax[0], "depth", "BFS depth $D$"), (ax[1], "seeds", "number of seeds $s$"),
                         (ax[2], "fraction", "sample size $N$")]:
        xs = sorted(design[key]); ys = [design[key][x][0] for x in xs]; es = [design[key][x][1] for x in xs]
        a.errorbar(xs, ys, yerr=es, fmt="o-", color=C1, capsize=3)
        a.axhline(base_true, color="grey", ls="--", lw=1, label=f"true Gini ({base_true:.2f})")
        a.set_xlabel(xlab); a.set_ylabel("sampled Gini"); a.set_ylim(0, 0.7); a.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_design.png"); plt.close(fig)
    print(f"-> {OUT/'bias_full.json'}, fig_bias_metrics.png, fig_design.png")


if __name__ == "__main__":
    main()
