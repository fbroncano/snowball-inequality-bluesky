"""
Robustness and sensitivity analysis.

(A) Sensitivity to the per-node follow cap K. The follows of one replication's nodes are
    re-collected with a high cap (2000) and smaller values are then simulated by
    subsampling each node's out-edges. If the concentration measures settle as K grows,
    the finding does not depend on how completely the hubs were read. Very low values of
    K additionally reproduce a sparse graph, of the kind a snowball crawl returns without
    densification, and show why densification is necessary.

(B) Sensitivity to the sample size N. The five replications (of varying N) are read back
    and the measures are checked not to depend on N.

Usage:
  ./venv/bin/python sensitivity.py
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from collect import BlueskyClient

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "run_0"
OUT = ROOT / "results"


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


def metrics(edges):
    G = nx.from_pandas_edgelist(edges, "src", "dst", create_using=nx.DiGraph)
    indeg = np.array([d for _, d in G.in_degree()])
    pr = np.array(list(nx.pagerank(G, alpha=0.85).values()))
    return {
        "n_edges": G.number_of_edges(),
        "mean_degree": round(G.number_of_edges() / G.number_of_nodes(), 2),
        "gini_indegree": round(gini(indeg), 3),
        "gini_pagerank": round(gini(pr), 3),
        "attention_top1pct": round(top_share(indeg), 3),
    }


def recapture_full(k_high=2000):
    """Re-collects internal edges with a high K, so smaller caps can be subsampled."""
    nodes = pd.read_parquet(RUN / "data" / "nodes.parquet")
    node_dids = set(nodes["did"])
    actors = nodes.dropna(subset=["handle"])[["did", "handle"]].values.tolist()
    client = BlueskyClient()
    from concurrent.futures import ThreadPoolExecutor
    import threading
    edges, lock = [], threading.Lock()

    def fetch(item):
        did, handle = item
        local = [(did, f["did"]) for f in client.get_follows(handle, k_high)
                 if f.get("did") in node_dids]
        with lock:
            edges.extend(local)

    print(f"Re-collecting follows (K={k_high}) for {len(actors)} nodes...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(fetch, actors))
    df = pd.DataFrame(edges, columns=["src", "dst"]).drop_duplicates()
    df.to_parquet(RUN / "data" / "edges_k2000.parquet", index=False)
    print(f"  {len(df)} internal edges re-collected")
    return df


def subsample_by_k(edges_full, k, rng):
    """Keeps at most k out-edges per node, simulating a cap of K."""
    by_src = defaultdict(list)
    for s, d in edges_full.itertuples(index=False):
        by_src[s].append(d)
    rows = []
    for s, dsts in by_src.items():
        keep = dsts if len(dsts) <= k else rng.sample(dsts, k)
        rows.extend((s, d) for d in keep)
    return pd.DataFrame(rows, columns=["src", "dst"])


def main():
    rng = random.Random(42)
    OUT.mkdir(exist_ok=True)
    report = {}

    # (A) Sensitivity to K
    full_path = RUN / "data" / "edges_k2000.parquet"
    edges_full = pd.read_parquet(full_path) if full_path.exists() else recapture_full()
    report["A_sensitivity_K"] = {}
    print("\n=== (A) Sensitivity to the cap K ===")
    for k in [5, 50, 250, 500, 1000, 2000]:
        m = metrics(subsample_by_k(edges_full, k, rng))
        report["A_sensitivity_K"][k] = m
        print(f"  K={k:5d}: edges={m['n_edges']:6d} degree={m['mean_degree']:5.2f} "
              f"Gini(in)={m['gini_indegree']:.3f} Gini(PR)={m['gini_pagerank']:.3f} "
              f"top1%={m['attention_top1pct']:.3f}")

    # (B) Sensitivity to N (existing replications)
    report["B_sensitivity_N"] = []
    print("\n=== (B) Sensitivity to N (5 replications) ===")
    for i in range(5):
        rp = ROOT / "runs" / f"run_{i}" / "results" / "report.json"
        if not rp.exists():
            continue
        r = json.loads(rp.read_text())
        row = {
            "run": i,
            "N": r.get("n_nodes_analyzed"),
            "gini_indegree": r["RQ2_atencion"]["gini_in_degree"],
            "gini_pagerank": r["RQ_poder"]["gini_pagerank"],
        }
        report["B_sensitivity_N"].append(row)
        print(f"  run {i}: N={row['N']:5d} Gini(in)={row['gini_indegree']:.3f} "
              f"Gini(PR)={row['gini_pagerank']:.3f}")

    (OUT / "sensitivity.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n-> {OUT / 'sensitivity.json'}")


if __name__ == "__main__":
    main()
