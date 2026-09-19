"""
The sampling bias of the concentration measures: the bias curve.

Question: what is the relation between the POPULATION in-degree Gini (the ground truth)
and the value obtained after snowball sampling plus densification? If that relation can
be characterised, an observed sampled Gini constrains the population one.

Design:
  - Directed populations of CONTROLLED true Gini are generated with a directed
    configuration model whose degree sequence is lognormal with increasing dispersion
    (more dispersion -> more concentration). An Erdos-Renyi and a scale-free population
    are added as references at the extremes.
  - The true in-degree Gini is measured on each population.
  - The same sampling (snowball over follow edges, cap K, up to N nodes) plus
    densification (induced subgraph) is applied, and the subgraph Gini is measured
    (mean over R samples).
  - Result: the curve "sampled Gini against true Gini", drawn with the identity line and
    the empirical Bluesky value, so that one can read off which true Gini is compatible
    with what was observed.

Output: results/bias_study.json, results/fig_bias.png

Usage:
  ./venv/bin/python bias_study.py
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
EMPIRICAL_GINI = 0.616

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 200,
})


def gini(x):
    x = np.sort(np.asarray(x, dtype=float)); x = x[x >= 0]
    if x.sum() == 0:
        return float("nan")
    n = len(x); idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def lognormal_population(pop, sigma, mean_deg, np_rng):
    """Directed graph with lognormal in-degree (sigma controls the concentration)."""
    mu = np.log(mean_deg) - sigma ** 2 / 2
    indeg = np.maximum(1, np.round(np_rng.lognormal(mu, sigma, pop)).astype(int))
    outdeg = np_rng.permutation(indeg)            # same sum -> valid configuration model
    diff = indeg.sum() - outdeg.sum()
    if diff != 0:
        outdeg[0] = max(1, outdeg[0] + diff)
    g = nx.directed_configuration_model(indeg.tolist(), outdeg.tolist(),
                                        seed=int(np_rng.integers(1e9)))
    g = nx.DiGraph(g); g.remove_edges_from(nx.selfloop_edges(g))
    return g


def reference_population(model, pop, seed):
    if model == "scale_free":
        return nx.DiGraph(nx.scale_free_graph(pop, seed=seed))
    if model == "er":
        return nx.gnm_random_graph(pop, pop * 15, directed=True, seed=seed)
    raise ValueError(model)


def snowball_densify(G, N, K, s, rng):
    nodes = list(G.nodes())
    seeds = rng.sample(nodes, min(s, len(nodes)))
    visited, queue, queued = set(), deque((x, 0) for x in seeds), set(seeds)
    while queue and len(visited) < N:
        u, d = queue.popleft(); visited.add(u)
        if d >= 4:
            continue
        outs = list(G.successors(u))
        if len(outs) > K:
            outs = rng.sample(outs, K)
        for v in outs:
            if v not in queued and len(visited) + len(queue) < N * 3:
                queue.append((v, d + 1)); queued.add(v)
    sub = G.subgraph(visited)
    return gini(np.array([dg for _, dg in sub.in_degree()]))


def evaluate(G, N, K, s, reps, rng):
    true_g = gini(np.array([d for _, d in G.in_degree()]))
    sub = [snowball_densify(G, N, K, s, rng) for _ in range(reps)]
    return round(true_g, 4), round(float(np.mean(sub)), 4), round(float(np.std(sub, ddof=1)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=40000)
    ap.add_argument("--N", type=int, default=3000)
    ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--reps", type=int, default=4)
    args = ap.parse_args()
    rng = random.Random(42); np_rng = np.random.default_rng(42)

    points = []
    print(f"{'model':>14} {'true':>6} {'sampled':>9} {'bias':>7}")
    # Concentration sweep with the lognormal configuration model.
    for sigma in [0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0]:
        G = lognormal_population(args.pop, sigma, 15, np_rng)
        t, sm, sd = evaluate(G, args.N, args.K, args.seeds, args.reps, rng)
        points.append({"model": f"lognormal_s{sigma}", "true": t, "sampled": sm, "sd": sd})
        print(f"{'logn s='+str(sigma):>14} {t:>6.3f} {sm:>6.3f}±{sd:<4.3f} {sm-t:>+7.3f}")
    # Reference populations.
    for model in ["er", "scale_free"]:
        G = reference_population(model, args.pop, 7)
        t, sm, sd = evaluate(G, args.N, args.K, args.seeds, args.reps, rng)
        points.append({"model": model, "true": t, "sampled": sm, "sd": sd})
        print(f"{model:>14} {t:>6.3f} {sm:>6.3f}±{sd:<4.3f} {sm-t:>+7.3f}")

    # Range of true Gini compatible with the observed one (interpolated over the lognormals).
    logn = sorted([p for p in points if p["model"].startswith("logn")], key=lambda p: p["true"])
    ts = np.array([p["true"] for p in logn]); ss = np.array([p["sampled"] for p in logn])
    consistent_true = float(np.interp(EMPIRICAL_GINI, ss, ts)) if ss.min() <= EMPIRICAL_GINI <= ss.max() else None

    report = {"params": vars(args), "empirical_sampled_gini": EMPIRICAL_GINI,
              "points": points,
              "true_gini_consistent_with_empirical": round(consistent_true, 3) if consistent_true else None}
    OUT.mkdir(exist_ok=True)
    (OUT / "bias_study.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Figure: sampled against true.
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="identity (no bias)")
    ax.errorbar(ts, ss, yerr=[p["sd"] for p in logn], fmt="o-", color="#1f77b4",
                capsize=3, label="lognormal populations")
    for p in points:
        if p["model"] in ("er", "scale_free"):
            ax.scatter(p["true"], p["sampled"], marker="^", s=70, zorder=5,
                       label=p["model"].replace("_", "-"))
    ax.axhline(EMPIRICAL_GINI, color="grey", ls=":", lw=1.2, label=f"Bluesky observed ({EMPIRICAL_GINI})")
    ax.set_xlabel("true population Gini (in-degree)")
    ax.set_ylabel("sampled subgraph Gini (in-degree)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_bias.png"); plt.close(fig)

    print(f"\nTrue Gini compatible with the empirical value ({EMPIRICAL_GINI}): "
          f"{report['true_gini_consistent_with_empirical']}")
    print(f"-> {OUT / 'bias_study.json'}  and  {OUT / 'fig_bias.png'}")


if __name__ == "__main__":
    main()
