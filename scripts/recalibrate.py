"""
Recalibration by the SAMPLED FRACTION f = N/N_pop.

The bias depends on f. A calibration built at one fraction cannot be carried to another,
and the synthetic curves sit at a few per cent whereas a crawl of a platform of tens of
millions operates at f of order 1e-4. This script establishes that:
  (1) f-dependence: at a fixed true Gini, vary f (raising N_pop with N fixed) and watch
      the sampled Gini move.
  (2) scale invariance: at fixed f, vary the absolute size (N, N_pop) and check that the
      sampled Gini depends on f and not on absolute size.
  (3) fit of the sampled Gini against log10(f), and what extrapolating it to the real f
      of Bluesky would imply. Taken literally the fit predicts a negative Gini, which is
      the paper's argument for why the bias must be addressed empirically instead.

Output: results/recalibrate.json, results/fig_fraction.png

Usage:
  ./venv/bin/python recalibrate.py --reps 8
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
EMP_GINI = 0.616
# Approximate population size of Bluesky (order of magnitude, ~3.7e7 accounts in 2026).
BLUESKY_NPOP = 3.7e7
BLUESKY_N = 3300  # mean nodes per replication

plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 10,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})


def gini(x):
    x = np.sort(np.asarray(x, dtype=float)); x = x[x >= 0]
    if x.sum() == 0:
        return float("nan")
    n = len(x); idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


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


def sample_gini(G, N, K, s, D, rng):
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
    sub = G.subgraph(visited)
    return gini(np.array([dg for _, dg in sub.in_degree()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=1.4)  # true Gini ~0.67
    args = ap.parse_args()
    rng = random.Random(0); np_rng = np.random.default_rng(0)
    report = {}

    # (1) f-dependence: N fixed at 3000, raise N_pop -> lower f.
    print("=== (1) f-dependence (true Gini held fixed) ===")
    N = 3000
    fdep = []
    for pop in [20000, 40000, 80000, 160000, 320000]:
        G = lognormal_pop(pop, args.sigma, 15, np_rng)
        tg = gini(np.array([d for _, d in G.in_degree()]))
        vals = [sample_gini(G, N, 1000, 10, 4, rng) for _ in range(args.reps)]
        f = N / pop
        fdep.append({"pop": pop, "f": f, "true": round(tg, 4),
                     "sampled": round(float(np.mean(vals)), 4),
                     "sd": round(float(np.std(vals, ddof=1)), 4)})
        print(f"  pop={pop:6d} f={f*100:5.2f}% true={tg:.3f} sampled={np.mean(vals):.3f}±{np.std(vals,ddof=1):.3f}")
    report["f_dependence"] = fdep

    # (2) scale invariance: f held at ~5%, vary the absolute size.
    print("\n=== (2) scale invariance (f held at ~5%) ===")
    scale = []
    for n_, pop in [(1500, 30000), (3000, 60000), (6000, 120000)]:
        G = lognormal_pop(pop, args.sigma, 15, np_rng)
        vals = [sample_gini(G, n_, 1000, 10, 4, rng) for _ in range(args.reps)]
        scale.append({"N": n_, "pop": pop, "f": n_ / pop,
                      "sampled": round(float(np.mean(vals)), 4),
                      "sd": round(float(np.std(vals, ddof=1)), 4)})
        print(f"  N={n_:5d} pop={pop:6d} f={n_/pop*100:.1f}% sampled={np.mean(vals):.3f}±{np.std(vals,ddof=1):.3f}")
    report["scale_invariance"] = scale

    # (3) fit sampled ~ a + b*log10(f) and extrapolate to Bluesky.
    fs = np.array([d["f"] for d in fdep]); ss = np.array([d["sampled"] for d in fdep])
    b, a = np.polyfit(np.log10(fs), ss, 1)
    f_bsky = BLUESKY_N / BLUESKY_NPOP
    sampled_pred_at_bsky_f = a + b * np.log10(f_bsky)
    # At the calibration f (7.5%) the truth is tg; the model says how far it would deflate at the real f.
    tg = fdep[0]["true"]
    report["extrapolation"] = {
        "fit_slope_per_decade": round(float(b), 4), "fit_intercept": round(float(a), 4),
        "true_gini_of_this_population": round(tg, 4),
        "bluesky_f_estimate": f_bsky,
        "predicted_sampled_gini_at_bluesky_f_if_true_is_%.2f" % tg: round(float(sampled_pred_at_bsky_f), 4),
        "interpretation": ("at the real f the same truth %.2f would measure lower still; "
                           "the observed 0.62 therefore implies a true Gini HIGHER than "
                           "the one estimated at f=7.5%%" % tg),
    }
    print(f"\n=== (3) extrapolation ===")
    print(f"  fit: sampled = {a:.3f} {b:+.3f}*log10(f)")
    print(f"  f Bluesky ~ {f_bsky:.2e}")
    print(f"  at that f, a truth of {tg:.2f} would measure ~{sampled_pred_at_bsky_f:.3f} "
          f"(vs {fdep[0]['sampled']:.3f} a f=15%)")

    OUT.mkdir(exist_ok=True)
    (OUT / "recalibrate.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Figure: sampled Gini against f (log x).
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.errorbar(fs * 100, ss, yerr=[d["sd"] for d in fdep], fmt="o-", capsize=3, color="#1f77b4",
                label=f"true Gini = {tg:.2f}")
    ax.axhline(tg, color="grey", ls="--", lw=1, label="true value")
    xs = np.logspace(np.log10(f_bsky), np.log10(0.16), 50)
    ax.plot(xs * 100, a + b * np.log10(xs), ":", color="#d62728", label="log-linear fit / extrapolation")
    ax.axvline(f_bsky * 100, color="green", ls=":", lw=1.2, label=f"Bluesky f $\\approx${f_bsky*100:.1e}\\%")
    ax.set_xscale("log"); ax.set_xlabel("sampled fraction $f = N/N_{pop}$ (\\%)")
    ax.set_ylabel("sampled in-degree Gini"); ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig_fraction.png"); plt.close(fig)
    print(f"-> {OUT/'recalibrate.json'}, fig_fraction.png")


if __name__ == "__main__":
    main()
