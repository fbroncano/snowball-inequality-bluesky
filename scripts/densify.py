"""
Densification of the induced subgraph.

The problem: the BFS in collect.py only retrieves the follows of the nodes it expands,
so the resulting graph is close to a tree and the structural measures (PageRank,
in-degree) come out degenerate.

The fix: given the node SET already collected (data/nodes.parquet), request the follows
of EVERY node and keep only the edges whose two endpoints are in the set, that is the
complete induced subgraph. Only then do in-degree and PageRank reflect standing within
the sampled community.

Usage:
  ./venv/bin/python densify.py --k 2000
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from collect import BlueskyClient, DATA_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2000,
                    help="Cap on follows requested per node (sampling of hubs)")
    ap.add_argument("--workers", type=int, default=16,
                    help="Number of concurrent threads for the API calls")
    args = ap.parse_args()

    nodes = pd.read_parquet(DATA_DIR / "nodes.parquet")
    node_dids = set(nodes["did"])
    # Map did -> handle for querying (the API accepts either).
    actors = nodes.dropna(subset=["handle"])[["did", "handle"]].values.tolist()

    client = BlueskyClient()
    edges: list[tuple[str, str]] = []
    lock = threading.Lock()
    done = {"n": 0}
    print(f"Densifying: requesting follows of {len(actors)} nodes "
          f"(k={args.k}, workers={args.workers})...")
    t0 = time.time()

    def fetch(item):
        did, handle = item
        local = [(did, f["did"]) for f in client.get_follows(handle, args.k)
                 if f.get("did") in node_dids]   # only edges INTERNAL to the set
        with lock:
            edges.extend(local)
            done["n"] += 1
            if done["n"] % 250 == 0:
                print(f"  ... {done['n']}/{len(actors)} nodes, {len(edges)} edges "
                      f"internal ({done['n']/(time.time()-t0):.1f} nodes/s)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(fetch, actors))

    edges_df = pd.DataFrame(edges, columns=["src", "dst"]).drop_duplicates()
    edges_df.to_parquet(DATA_DIR / "edges.parquet", index=False)

    # Update meta.
    meta = json.loads((DATA_DIR / "meta.json").read_text())
    meta["densified_at"] = datetime.now(timezone.utc).isoformat()
    meta["densify_k"] = args.k
    meta["n_edges"] = len(edges_df)
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    elapsed = time.time() - t0
    n = len(node_dids)
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  internal edges: {len(edges_df)}  (mean degree {len(edges_df)/n:.2f})")
    print(f"  -> {DATA_DIR / 'edges.parquet'}")


if __name__ == "__main__":
    main()
