"""
Lorenz curves: the central result of the paper in a single figure.

Three curves are overlaid on the same September 2026 data:
  (a) in-degree within the snowball subgraph              -> Gini 0.62
  (b) global follower counters of THOSE SAME accounts     -> Gini 0.88
  (c) global follower counters on a uniform census sample -> Gini 0.93

The separation (a)->(b) is the ESTIMAND bias: the same node set, a different quantity
measured. The separation (b)->(c) is the NODE SET bias. That (a)->(b) is by far the
larger of the two is the thesis of the paper.

Output: results/fig_lorenz.png
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11, "legend.fontsize": 9.5,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 200,
})


def gini(x):
    x = np.sort(np.asarray(x, dtype=float)); x = x[x >= 0]
    n = len(x); idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def lorenz(x):
    x = np.sort(np.asarray(x, dtype=float)); x = x[x >= 0]
    c = np.cumsum(x) / x.sum()
    return np.linspace(0, 1, len(x) + 1), np.concatenate([[0.0], c])


def main():
    # (a) and (b): the snowball subgraph and the counters of those same nodes
    indeg, foll_sn = [], []
    for run in sorted(glob.glob(str(ROOT / "runs" / "run_*"))):
        nodes = pd.read_parquet(Path(run) / "data" / "nodes.parquet")
        edges = pd.read_parquet(Path(run) / "data" / "edges.parquet")
        deg = edges.iloc[:, 1].value_counts()
        indeg.append(nodes["did"].map(deg).fillna(0).to_numpy())
        foll_sn.append(nodes["followers_count"].fillna(0).to_numpy())
    indeg = np.concatenate(indeg); foll_sn = np.concatenate(foll_sn)

    # (c): uniform sample drawn from the census
    foll_un = np.load(OUT / "uniform_big_raw.npz")["followers"]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="equality")
    for x, lab, col, ls in (
        (indeg,   "snowball subgraph, in-degree",       "#d62728", "-"),
        (foll_sn, "snowball node set, follower counts", "#ff7f0e", "-"),
        (foll_un, "uniform sample, follower counts",    "#1f77b4", "-"),
    ):
        p, c = lorenz(x)
        ax.plot(p, c, ls, color=col, lw=2, label=f"{lab} (G = {gini(x):.2f})")
    ax.set_xlabel("cumulative share of accounts (poorest first)")
    ax.set_ylabel("cumulative share of attention")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_lorenz.png", dpi=200)
    print(f"  in-degree (snowball)      G={gini(indeg):.4f}  n={len(indeg)}")
    print(f"  followers (snowball set)  G={gini(foll_sn):.4f}  n={len(foll_sn)}")
    print(f"  followers (uniform)       G={gini(foll_un):.4f}  n={len(foll_un)}")
    print(f"-> {OUT / 'fig_lorenz.png'}")


if __name__ == "__main__":
    main()
