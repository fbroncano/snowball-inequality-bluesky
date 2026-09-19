"""
Descriptive statistics of the dataset (for the data section of the paper) and the
supporting figures: follower counts by order of magnitude, account ages, and the
robustness panel built from results/sensitivity.json.

Output: results/descriptive.json, results/fig_dataset.png, results/fig_robustness.png

Usage:
  ./venv/bin/python descriptive.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 11, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})


def load_all_nodes():
    frames = [pd.read_parquet(p) for p in (ROOT / "runs").glob("run_*/data/nodes.parquet")]
    nodes = pd.concat(frames, ignore_index=True).drop_duplicates(subset="did")
    return nodes


def main():
    OUT.mkdir(exist_ok=True)
    nodes = load_all_nodes()
    fc = nodes["followers_count"].dropna()
    pc = nodes["posts_count"].dropna()
    age = nodes["days_active"].dropna()

    desc = {
        "n_unique_nodes": int(len(nodes)),
        "followers": {"median": float(fc.median()), "mean": round(float(fc.mean()), 1),
                      "p90": float(fc.quantile(0.9)), "max": float(fc.max())},
        "posts": {"median": float(pc.median()), "mean": round(float(pc.mean()), 1)},
        "account_age_days": {"median": float(age.median()), "mean": round(float(age.mean()), 1)},
    }
    (OUT / "descriptive.json").write_text(json.dumps(desc, indent=2, ensure_ascii=False))
    print("Descriptives:", json.dumps(desc, ensure_ascii=False))

    # --- Dataset figure: followers by order of magnitude + account age ---
    bins = [0, 10, 100, 1_000, 10_000, 100_000, 1_000_000, np.inf]
    labels = ["<10", "10-10$^2$", "10$^2$-10$^3$", "10$^3$-10$^4$",
              "10$^4$-10$^5$", "10$^5$-10$^6$", ">10$^6$"]
    counts = pd.cut(fc, bins=bins, right=False, labels=labels).value_counts().sort_index()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(range(len(counts)), counts.values, color="#1f77b4", edgecolor="white")
    ax[0].set_xticks(range(len(counts)))
    ax[0].set_xticklabels(labels, rotation=45, ha="right")
    ax[0].set_ylabel("number of accounts")
    ax[0].set_xlabel("followers (order of magnitude)")
    ax[0].set_title("(a) Follower count distribution")

    ax[1].hist((age / 365.25).clip(upper=age.max() / 365.25), bins=30,
               color="#d62728", edgecolor="white")
    ax[1].set_xlabel("account age (years)")
    ax[1].set_ylabel("number of accounts")
    ax[1].set_title("(b) Account age distribution")
    fig.tight_layout()
    fig.savefig(OUT / "fig_dataset.png", dpi=150)
    plt.close(fig)

    # --- Robustness figure: Gini against K and Gini against N ---
    sens = json.loads((OUT / "sensitivity.json").read_text())
    A = sens["A_sensitivity_K"]
    Ks = sorted(int(k) for k in A)
    gin = [A[str(k)]["gini_indegree"] for k in Ks]
    gpr = [A[str(k)]["gini_pagerank"] for k in Ks]
    B = sens["B_sensitivity_N"]
    Ns = [r["N"] for r in B]
    gin_n = [r["gini_indegree"] for r in B]
    gpr_n = [r["gini_pagerank"] for r in B]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogx(Ks, gin, "o-", label="Gini in-degree")
    ax[0].semilogx(Ks, gpr, "s--", label="Gini PageRank")
    ax[0].set_xlabel("follow cap per node, $K$")
    ax[0].set_ylabel("Gini")
    ax[0].set_ylim(0, 0.7)
    ax[0].set_title("(a) Sensitivity to the hub cap $K$")
    ax[0].legend()

    ax[1].scatter(Ns, gin_n, label="Gini in-degree")
    ax[1].scatter(Ns, gpr_n, marker="s", label="Gini PageRank")
    ax[1].set_xlabel("sample size, $N$")
    ax[1].set_ylabel("Gini")
    ax[1].set_ylim(0, 0.7)
    ax[1].set_title("(b) Stability across sample size $N$")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_robustness.png", dpi=150)
    plt.close(fig)
    print("Figures: fig_dataset.png, fig_robustness.png")


if __name__ == "__main__":
    main()
