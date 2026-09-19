"""
Analysis of the collected graph.

  RQ1 inequality of PRODUCTION (posts)        -> Gini, Lorenz, 90-9-1, power law
  RQ2 inequality of ATTENTION (in-degree)     -> Gini, Lorenz, power law
  RQ3 do producers and receivers coincide?    -> Spearman(posts, in-degree)
  RQ4 concentration of attention              -> share held by the top 1%, hubs

The power-law fit uses the Clauset-Shalizi-Newman (2009) method through the `powerlaw`
package, and ALWAYS compares against a lognormal by likelihood ratio.

Usage:
  ./venv/bin/python analyze.py
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
DATA_DIR = Path(os.environ.get("RRSS_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
OUT_DIR = Path(os.environ.get("RRSS_OUT_DIR", Path(__file__).resolve().parent.parent / "results"))


def gini(x: np.ndarray) -> float:
    """Gini coefficient (0 = perfect equality, 1 = maximal concentration)."""
    x = np.sort(np.asarray(x, dtype=float))
    x = x[x >= 0]
    if x.sum() == 0:
        return float("nan")
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def top_share(x: np.ndarray, frac: float) -> float:
    """Share of the total held by the top `frac` (e.g. 0.01 = top 1%)."""
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    k = max(1, int(len(x) * frac))
    return float(x[:k].sum() / x.sum())


def rule_90_9_1(x: np.ndarray) -> dict:
    """% of the total produced by the top 1%, top 10% and bottom 90% (Nielsen's rule)."""
    return {
        "share_top_1pct": round(top_share(x, 0.01), 4),
        "share_top_10pct": round(top_share(x, 0.10), 4),
        "share_bottom_90pct": round(1 - top_share(x, 0.10), 4),
    }


def fit_powerlaw(x: np.ndarray, label: str) -> dict:
    """
    Power-law fit (Clauset et al. 2009) and likelihood-ratio comparison with a lognormal.
    R>0 favorece power-law; R<0 favorece lognormal; p indica significancia.
    """
    import powerlaw

    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    if len(x) < 50 or np.unique(x).size < 10:
        return {"variable": label, "note": "sample too small or too flat for a reliable fit"}
    try:
        fit = powerlaw.Fit(x, discrete=True, verbose=False)
        R_ln, p_ln = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
        R_exp, p_exp = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)
    except (ValueError, RuntimeError) as e:
        return {"variable": label, "note": f"fit not feasible on this sample ({e})"}
    if R_ln > 0 and p_ln < 0.05:
        verdict = "power law preferred over lognormal"
    elif R_ln < 0 and p_ln < 0.05:
        verdict = "LOGNORMAL preferred over power law"
    else:
        verdict = "inconclusive (power law not distinguishable from lognormal)"
    return {
        "variable": label,
        "alpha": round(float(fit.alpha), 3),
        "xmin": float(fit.xmin),
        "n_tail": int((x >= fit.xmin).sum()),
        "vs_lognormal_R": round(float(R_ln), 3),
        "vs_lognormal_p": round(float(p_ln), 4),
        "vs_exponential_R": round(float(R_exp), 3),
        "vs_exponential_p": round(float(p_exp), 4),
        "verdict": verdict,
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    nodes = pd.read_parquet(DATA_DIR / "nodes.parquet")
    edges = pd.read_parquet(DATA_DIR / "edges.parquet")
    meta = json.loads((DATA_DIR / "meta.json").read_text())

    # Directed graph: src follows dst -> in-degree(dst) = attention received within the sample.
    G = nx.from_pandas_edgelist(edges, "src", "dst", create_using=nx.DiGraph)
    in_deg = dict(G.in_degree())
    nodes["in_degree"] = nodes["did"].map(in_deg).fillna(0).astype(int)

    # Production adjusted for account age (control).
    nodes["posts_per_day"] = nodes["posts_count"] / nodes["days_active"].clip(lower=1)

    report: dict = {"meta": meta, "n_nodes_analyzed": len(nodes)}

    # --- RQ1: inequality of production ---
    posts = nodes["posts_count"].dropna().to_numpy()
    report["RQ1_production"] = {
        "gini_posts": round(gini(posts), 4),
        **rule_90_9_1(posts),
        "powerlaw": fit_powerlaw(posts, "posts_count"),
    }

    # --- RQ2: inequality of attention (in-degree of the sampled graph) ---
    indeg = nodes["in_degree"].to_numpy()
    report["RQ2_attention"] = {
        "gini_in_degree": round(gini(indeg), 4),
        **rule_90_9_1(indeg),
        "powerlaw": fit_powerlaw(indeg, "in_degree"),
        "note": "in_degree = attention INSIDE the sample; not the global followers_count",
    }
    # Global follower counts (from the API) as a complementary reference.
    fc = nodes["followers_count"].dropna().to_numpy()
    report["RQ2_attention"]["gini_followers_global"] = round(gini(fc), 4)

    # --- RQ3: do the big producers and the big receivers of attention coincide? ---
    sub = nodes.dropna(subset=["posts_count", "in_degree"])
    rho, p = spearmanr(sub["posts_count"], sub["in_degree"], nan_policy="omit")
    rho2, p2 = spearmanr(sub["posts_per_day"], sub["in_degree"], nan_policy="omit")
    report["RQ3_overlap"] = {
        "spearman_posts_vs_indegree": {"rho": round(float(rho), 4), "p": round(float(p), 6)},
        "spearman_postsperday_vs_indegree": {"rho": round(float(rho2), 4), "p": round(float(p2), 6)},
    }

    # --- RQ2b: GLOBAL follower distribution (few with many, many with few) ---
    # followers_count is the platform's own counter, not the in-degree within the sample.
    fc_series = nodes["followers_count"].dropna()
    fc = fc_series.to_numpy()
    # Histogram by order of magnitude, which shows the head-and-tail shape at a glance.
    bins = [0, 10, 100, 1_000, 10_000, 100_000, 1_000_000, float("inf")]
    labels = ["0-9", "10-99", "100-999", "1k-9.9k", "10k-99k", "100k-999k", "1M+"]
    buckets = pd.cut(fc_series, bins=bins, right=False, labels=labels).value_counts().sort_index()
    report["RQ2b_follower_distribution"] = {
        "gini_followers": round(gini(fc), 4),
        "median": float(np.median(fc)),
        "mean": round(float(np.mean(fc)), 1),
        "max": float(np.max(fc)),
        "pct_profiles_under_1000_followers": round(float((fc < 1000).mean()), 4),
        "histogram_by_order_of_magnitude": {k: int(v) for k, v in buckets.items()},
        "powerlaw": fit_powerlaw(fc, "followers_count"),
        "reading": "mean >> median with a long tail = a few nodes hold almost all the audience",
    }

    # --- RQ4: concentration of attention, and the hubs ---
    top_hubs = (nodes.sort_values("in_degree", ascending=False)
                .head(10)[["handle", "in_degree", "followers_count", "posts_count"]])
    report["RQ4_concentration"] = {
        "attention_top_1pct": round(top_share(indeg, 0.01), 4),
        "attention_top_10pct": round(top_share(indeg, 0.10), 4),
        "top_10_hubs": top_hubs.to_dict("records"),
    }

    # --- RQ_power: centrality / influence in the graph, beyond in-degree ---
    # PageRank: recursive standing (being followed by accounts that are themselves followed).
    # Betweenness: control of the bridges along which information flows between communities.
    pr = nx.pagerank(G, alpha=0.85)
    nodes["pagerank"] = nodes["did"].map(pr).fillna(0)
    # Betweenness is costly on large graphs -> approximate it by sampling k pivots.
    k_piv = min(500, G.number_of_nodes())
    btw = nx.betweenness_centrality(G, k=k_piv, seed=42) if G.number_of_nodes() > 1 else {}
    nodes["betweenness"] = nodes["did"].map(btw).fillna(0)

    top_power = (nodes.sort_values("pagerank", ascending=False)
                 .head(10)[["handle", "pagerank", "in_degree", "followers_count", "betweenness"]])
    # Does standing (PageRank) track raw audience (the global follower count)?
    sub_p = nodes.dropna(subset=["followers_count"])
    rho_pf, p_pf = spearmanr(sub_p["pagerank"], sub_p["followers_count"], nan_policy="omit")
    report["RQ_power"] = {
        "gini_pagerank": round(gini(nodes["pagerank"].to_numpy()), 4),
        "pagerank_top_1pct_share": round(top_share(nodes["pagerank"].to_numpy(), 0.01), 4),
        "spearman_pagerank_vs_followers": {"rho": round(float(rho_pf), 4), "p": round(float(p_pf), 6)},
        "top_10_by_power": top_power.round(6).to_dict("records"),
        "note": "in-degree measures direct audience; PageRank measures recursive standing (Cha et al. 2010)",
    }

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Human-readable summary on stdout.
    print("=" * 64)
    print(f"SAMPLE: {len(nodes)} nodes, {G.number_of_edges()} edges "
          f"(collected {meta['collected_at'][:10]})")
    print("=" * 64)
    r1 = report["RQ1_production"]
    print(f"\n[RQ1] PRODUCTION (posts)")
    print(f"  Gini = {r1['gini_posts']}  | top 10% produce {r1['share_top_10pct']:.0%}")
    print(f"  Power-law: {r1['powerlaw'].get('verdict', r1['powerlaw'].get('note'))}")
    r2 = report["RQ2_attention"]
    print(f"\n[RQ2] ATTENTION (in-degree within sample)")
    print(f"  Gini = {r2['gini_in_degree']}  | top 10% receive {r2['share_top_10pct']:.0%}")
    print(f"  Power-law: {r2['powerlaw'].get('verdict', r2['powerlaw'].get('note'))}")
    r3 = report["RQ3_overlap"]["spearman_posts_vs_indegree"]
    print(f"\n[RQ3] producing vs being attended to: Spearman rho={r3['rho']} (p={r3['p']})")
    rb = report["RQ2b_follower_distribution"]
    print(f"\n[RQ2b] FOLLOWER DISTRIBUTION (global)")
    print(f"  median={rb['median']:.0f} vs mean={rb['mean']:.0f}  "
          f"| {rb['pct_profiles_under_1000_followers']:.0%} have <1000 followers")
    print(f"  by order of magnitude: {rb['histogram_by_order_of_magnitude']}")
    print(f"  Power-law: {rb['powerlaw'].get('verdict', rb['powerlaw'].get('note'))}")
    r4 = report["RQ4_concentration"]
    print(f"\n[RQ4] the top 1% receive {r4['attention_top_1pct']:.0%} of the attention")
    rp = report["RQ_power"]
    print(f"\n[POWER] Gini(PageRank)={rp['gini_pagerank']} | "
          f"top 1% hold {rp['pagerank_top_1pct_share']:.0%} of the standing")
    print(f"  highest-standing nodes: "
          f"{[h['handle'] for h in rp['top_10_by_power'][:5]]}")
    print(f"  -> {OUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
